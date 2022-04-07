#!/bin/bash python
import torch 
import torch.nn.functional as F
from torch.utils import tensorboard
import torch.optim as optim
import pandas as pd
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from einops import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os
from transformer import Transformer, block_to_patch, patch_to_block
from utils import *


class early_stopping():
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.best_loss = None
        self.counter = 0
        self.best_model = None
    
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss-self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model
        else:
            self.counter += 1 
            if self.counter == self.patience:
                self.early_stop = True
                print('Early stopping')
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----')


def evaluate(model,data_loader,criterion, patch_size,scaler,noise_std,predict_res):
    patch_length = 4*4*4
    model.eval()
    test_result = torch.Tensor(0) 
    truth = torch.Tensor(0) 
    test_ts = torch.Tensor(0)
    test_coord = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    
    with torch.no_grad():
        for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(data_loader):

            src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), src_coord.to(device),\
                                                                            tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
            if i==0:
                B, N, C = src.shape
                enc_in = src
                test_rollout = src
            else:
                enc_in = test_rollout[:,-N:,:]

            # dec_rollout = reduce(enc_in.view(B,window_size,patch_length,-1), 'b n p c -> b p c', 'mean')
            # dec_rollout = torch.zeros((enc_in.shape[0],patch_length,enc_in.shape[2])).float().to(device)
            if predict_res:
                res = torch.roll(enc_in,-patch_length,1)-enc_in
                res_rollout = torch.cat([res[:,:-patch_length,:],res[:,-patch_length*2:-patch_length,:]],dim=1) + (torch.empty(enc_in.shape).normal_(mean=0,std=noise_std/10)).to(device)###repeat the second to the last patch of residuals
                output = model(enc_in,res_rollout,src_coord,tgt_coord,src_ts,tgt_ts,{})
                output = output + enc_in
            else:
                dec_rollout = enc_in[:,-patch_length:,:]
                dec_in = torch.cat([enc_in[:,patch_length:,:], dec_rollout], dim=1).float()
                dec_in = dec_in + (torch.empty(dec_in.shape).normal_(mean=0,std=noise_std)).to(device)
                output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts, {})
            test_rollout = torch.cat([test_rollout,output[:,-patch_length:,:]], dim=1)
            truth = torch.cat((truth, tgt[:,-patch_length:,:].flatten().detach().cpu()), 0)
            test_result = torch.cat((test_result, output[:,-patch_length:,:].flatten().detach().cpu()), 0)

    test_result = torch.Tensor(scaler.inverse_transform(test_result.unsqueeze(-1)))
    truth = torch.Tensor(scaler.inverse_transform(truth.unsqueeze(-1)))
    prediction, truth = test_result.numpy(), truth.numpy()
    val_loss = mean_squared_error(truth, prediction)
    r2 = r2_score(truth, prediction)
    explained_variance = explained_variance_score(truth, prediction)

    return val_loss, r2, explained_variance


def train(config, checkpoint_dir):
    root_dir = '/scratch/yd1008/nyu_capstone_2/notebooks/turbulence_16_yd/tune_results_2/'
    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)
    patch_length = 4*4*4
    ### UNet
    unet_num_layer = config['unet_num_layer']
    unet_start_filts = config['unet_start_filts']
    ### TMSA 
    # '22221000', '24441000', '24441222'
    if config['tmsa_window_shift_size'] == '22221000':
        tmsa_window_patch_size = (2,2,2,2)
        tmsa_shift_size = (1,0,0,0)
    elif config['tmsa_window_shift_size'] == '24441000':
        tmsa_window_patch_size = (2,4,4,4)
        tmsa_shift_size = (1,0,0,0)
    elif config['tmsa_window_shift_size'] == '24441222':
        tmsa_window_patch_size = (2,4,4,4)
        tmsa_shift_size = (1,2,2,2)
    tmsa_depth = config['tmsa_depth']
    ### Data
    noise_std = config['noise_std']
    window_size = config['window_size']
    scaler_type = config['scaler_type']
    ### Transformer
    feature_size = config['feature_size']
    num_enc_layers = config['num_enc_layers']
    num_dec_layers = config['num_dec_layers']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    dropout = config['dropout']
    lr = config['lr']
    lr_decay = config['lr_decay']
    loss_type = config['loss_type']
    delta = config['delta']
    reg_var = config['reg_var']
    predict_res = config['predict_res']


    conv_config = {'num_layer':unet_num_layer, 'start_filts':unet_start_filts, 'conv_type': 'UNet'}
    tmsa_config = {'use_tmsa':True, 'use_tgt_tmsa':True, 'window_patch_size': tmsa_window_patch_size, 'shift_size': tmsa_shift_size, 'depth': tmsa_depth, 'num_heads':num_heads}
    data_config = {'scale': True, 'noise_std':noise_std, 'window_size': window_size, 'option': 'patch', 'predict_res': predict_res, 'scaler_type':scaler_type,'patch_size': (4,4,4),}
    best_config = {'epochs':40, 'pe_type': '3d_temporal', 'batch_size': 64, 'feature_size': feature_size, 'num_enc_layers': num_enc_layers\
                , 'num_dec_layers': num_dec_layers, 'num_head': num_heads, 'd_ff': d_ff, 'dropout': dropout, 'lr': lr, 'lr_decay': lr_decay, 'loss_type':loss_type, 'delta': delta\
                , 'mask_type':'patch','decoder_only':False, 'reg_var':reg_var}

    pe_type = best_config['pe_type']
    batch_size = best_config['batch_size']
    feature_size = best_config['feature_size']
    num_enc_layers = best_config['num_enc_layers']
    num_dec_layers = best_config['num_dec_layers']
    d_ff = best_config['d_ff']
    num_head = best_config['num_head']
    dropout = best_config['dropout']
    lr = best_config['lr']
    lr_decay = best_config['lr_decay']
    mask_type = best_config['mask_type']
    decoder_only = best_config['decoder_only']
    delta = best_config['delta']
    reg_var = best_config['reg_var']
    loss_type = best_config['loss_type']

    patch_size = data_config['patch_size']
    noise_std = data_config['noise_std']
    scale = data_config['scale']
    option = data_config['option']
    predict_res = data_config['predict_res']
    window_size = data_config['window_size']
    scaler_type = data_config['scaler_type']

    patch_length = np.prod(data_config['patch_size'])
    # dataset parameters
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    pred_size = 1
    grid_size = 16
    
    skip_training = False
    save_model = True

    print('-'*20 + ' Config ' + '-'*20, flush=True)
    print(best_config, flush=True)
    print(data_config, flush=True)
    print(tmsa_config, flush=True)
    print(conv_config, flush=True)
    print('-'*50, flush=True)
    best_config.update(data_config)

    ### SPECIFYING use_coords=True WILL RETURN DATALOADERS FOR COORDS
    train_loader, val_loader, _, scaler, data = get_data_loaders(train_proportion, test_proportion, val_proportion,\
        pred_size = pred_size, batch_size = batch_size, num_workers = 1, pin_memory = False, use_coords = True, use_time = True,\
        test_mode = False, scale = scale, window_size = window_size, patch_size = patch_size, option = option, predict_res = False,\
        noise_std = noise_std, scaler_type = scaler_type)
    
    model = Transformer(data, feature_size=feature_size,num_enc_layers=num_enc_layers,num_dec_layers = num_dec_layers,\
        d_ff = d_ff, dropout=dropout,num_head=num_head,pe_type=pe_type,grid_size=(grid_size,)*3,mask_type=mask_type,\
        patch_size=patch_size,window_size=window_size,pred_size=pred_size,decoder_only=decoder_only, tmsa_config = tmsa_config, conv_config = conv_config)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print('Using device: ',device)
    model.to(device)
    
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'l2':
        criterion = nn.MSELoss()
    elif loss_type == 'huber':
        criterion = nn.HuberLoss(delta=delta)
    elif loss_type == 'smooth_l1':
        criterion = nn.SmoothL1Loss(beta=delta)
    # criterion = nn.HuberLoss(delta=delta)
    # criterion = nn.SmoothL1Loss(beta=delta)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    #writer = tensorboard.SummaryWriter('/scratch/yd1008/tensorboard_output/')

    time_map_indices = train_loader.dataset.time_map_indices

    epochs = best_config['epochs']
    tolerance = 10
    best_test_loss = float('inf')
    Early_Stopping = early_stopping(patience=tolerance)
    counter_old = 0
    x1, x2, x3 = patch_size


    if os.path.exists(root_dir+'/best_model.pth') and skip_training:
        model.load_state_dict(torch.load(root_dir+'/best_model.pth'))
    else:
        for epoch in range(1, epochs + 1):    
            model.train() 
            total_loss = 0.
            start_time = time.time()

            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(train_loader):
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), src_coord.to(device),\
                                                                            tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                optimizer.zero_grad()

                # dec_rollout = reduce(src.view(src.shape[0],window_size,patch_length,-1), 'b n p c -> b p c', 'mean')
                # dec_rollout = torch.zeros((src.shape[0],patch_length,src.shape[2])).float().to(device)
                if predict_res:
                    res = tgt - src
                    res_rollout = torch.cat([res[:,:-patch_length,:],res[:,-patch_length*2:-patch_length,:]],dim=1) + (torch.empty(tgt.shape).normal_(mean=0,std=noise_std/10)).to(device)###repeat the second to the last patch of residuals
                    output = model(src,res_rollout,src_coord,tgt_coord,src_ts,tgt_ts,time_map_indices)
                    output = output + src
                else:
                    dec_rollout = src[:,-patch_length:,:]
                    dec_in = torch.cat([src[:,patch_length:,:], dec_rollout], dim=1).float()
                    dec_in = dec_in + (torch.empty(tgt.shape).normal_(mean=0,std=noise_std)).to(device) 
                    output = model(src, dec_in, src_coord, tgt_coord, src_ts, tgt_ts, time_map_indices)
                loss = criterion(output[:,-x1*x2*x3:,:], tgt[:,-x1*x2*x3:,:]) + reg_var*torch.abs((torch.var(tgt[:,-x1*x2*x3:,:].detach(),1)-torch.var(output[:,-x1*x2*x3:,:],1))).mean()# - 0.1*torch.std(output,dim=1).mean()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss/len(train_loader.dataset)
            val_loss, r2, explained_variance = evaluate(model, val_loader, criterion, patch_size=patch_size, scaler = scaler, noise_std=noise_std, predict_res = predict_res)
            #val_loss = total_val_loss

            print(f'Epoch: {epoch}, train_loss: {avg_train_loss}, test_loss: {val_loss}, lr: {scheduler.get_last_lr()}, training time: {time.time()-start_time} s', flush=True)

            Early_Stopping(model, val_loss)
            counter_new = Early_Stopping.counter
            if counter_new != counter_old:
                scheduler.step()  # update lr if early stop
                counter_old = counter_new
            if Early_Stopping.early_stop:
                break
            if Early_Stopping.best_loss>=val_loss:
                tune.report(train_loss = avg_train_loss, val_loss = val_loss, r2 = r2, explained_variance = explained_variance, epoch=epoch)

            


if __name__ == "__main__":
    """
    Notes:
    num_samples: number of trails we plan to do
    config_1: tunes for the combination of patch_size, batch_size, window_size, and lr, which should be tuned with priority
    config_2: tunes for pe_type and the model parameters, which should be tuned after we get the best setting in config_1
    """

    num_samples = 500
    config_1 = {
        'unet_num_layer':tune.choice([2,3,4]),
        'unet_start_filts':tune.choice([32,64,128]),
        'tmsa_window_shift_size':tune.choice(['22221000', '24441000', '24441222']),
        'tmsa_depth':tune.choice([4,6,8]),
        'noise_std':tune.choice([0.01,0.05,0.1,0.15]),
        'window_size':tune.choice([5,6,7,8]),
        'scaler_type':tune.choice(['power_yeo','quantile']),
        'feature_size':tune.choice([288,288+144,288*2,288*2+144]),
        'num_enc_layers':tune.choice([1,2,3]),
        'num_dec_layers':tune.choice([3,4,5]),
        'num_heads':tune.choice([4,8]),
        'd_ff':tune.choice([512,1024]),
        'dropout':tune.choice([0.1,0.2]),
        'lr':tune.choice([1e-5,5e-6,1e-6]),
        'lr_decay':tune.choice([0.9,0.8]),
        'loss_type':tune.choice(['huber','smooth_l1']),
        'delta':tune.choice([1,1.5,2,2.5]),
        'reg_var':tune.choice([0,0.001,0.01,0.05]),
        'predict_res':tune.choice([True,False]),
}   

#     config_2 = {
#         'window_size':tune.grid_search([10]), 
#         'patch_x1':tune.grid_search([1]),
#         'patch_x2':tune.grid_search([1]),
#         'patch_x3':tune.grid_search([1]),
#         'batch_size':tune.grid_search([16]),
#         'pe_type':tune.grid_search(['1d','3d', '3d_temporal']),
#         'feature_size':tune.grid_search([120, 240]),
#         #'num_enc_layers':tune.grid_search([1]),
#         #'num_dec_layers':tune.grid_search([1]),
#         #'num_head':tune.grid_search([2]),
#         #'dropout':tune.grid_search([0.1,0.2]),
#         #'d_ff':tune.grid_search([512])
#         #'lr':tune.grid_search([0.0001]),
#         #'window_size':tune.grid_search([12,36,108,324]),
#         #'batch_size':tune.grid_search([16])
# }   


    ray.init(ignore_reinit_error=False, include_dashboard=True, dashboard_host='0.0.0.0')
    sched = ASHAScheduler(
            max_t=30,
            grace_period=10,
            reduction_factor=2)
    analysis = tune.run(tune.with_parameters(train), config=config_1, num_samples=num_samples, metric='val_loss', mode='min',\
          scheduler=sched, resources_per_trial={"cpu": 10,"gpu": 1}, max_concurrent_trials = 4, queue_trials = True, max_failures=0, local_dir="/scratch/yd1008/ray_results")

    best_trail = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_trail, flush=True)
    ray.shutdown()