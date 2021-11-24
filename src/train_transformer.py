#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from transformer import Tranformer
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
                print('Early stopping', flush=True)
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----', flush=True)


def evaluate(model,data_loader,criterion, patch_size=(1,1,16)):
    model.eval()    
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    total_loss = 0.
    x1, x2, x3 = patch_size
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():        
        for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(data_loader):
            if i == 0:
                enc_in = src
                dec_in = tgt
                test_rollout = tgt
            else:
                enc_in = test_rollout[:,-tgt.shape[1]:,:]
                dec_in = torch.zeros([enc_in.shape[0], x1*x2*x3, enc_in.shape[-1]]).float()
                dec_in = torch.cat([enc_in[:,:(tgt.shape[1]-x1*x2*x3),:], dec_in], dim=1).float()
                #dec_in = enc_in[:,:(window_size-1),:]
            enc_in, dec_in, tgt = enc_in.to(device), dec_in.to(device), tgt.to(device)
            src_coord, tgt_coord, src_ts, tgt_ts = src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
            
            output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts) 
            test_rollout = torch.cat([test_rollout,output[:,-x1*x2*x3:,:].detach().cpu()],dim = 1)
            truth = torch.cat((truth, tgt[:,-x1*x2*x3:,:].flatten().detach().cpu()), 0)
            test_result = torch.cat((test_result, output[:,-x1*x2*x3:,:].flatten().detach().cpu()), 0)

    prediction, truth = test_result.numpy(), truth.numpy()
    val_loss = mean_squared_error(truth, prediction)
    r2 = r2_score(truth, prediction)
    explained_variance = explained_variance_score(truth, prediction)
    return val_loss, r2, explained_variance


def train(config, checkpoint_dir):
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    grid_size = 16

    window_size = config['window_size']
    x1 = config['patch_x1']
    x2 = config['patch_x2']
    x3 = config['patch_x3']
    patch_size = (x1, x2, x3)
    pe_type = config['pe_type']
    feature_size = config['feature_size']
    batch_size = config['batch_size']

    scale = False
    num_enc_layers = 1
    num_dec_layers = 1
    num_head = 2
    dropout = 0.1 #config['dropout']
    d_ff = 2048
    lr = 1e-5
    lr_decay = 0.9
    
    #model = Tranformer(feature_size=feature_size,num_layers=num_layer,dropout=dropout,num_head=num_head)
    model = Tranformer(feature_size=feature_size,num_enc_layers=num_enc_layers,num_dec_layers = num_dec_layers,\
            d_ff = d_ff, dropout=dropout,num_head=num_head,pe_type=pe_type)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    Early_Stopping = early_stopping(patience=4)
    counter_old = 0
    epochs = 20        

    criterion = nn.MSELoss() ######MAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    writer = tensorboard.SummaryWriter('./test_logs')
        
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    #     model_state, optimizer_state = torch.load(checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
            
    train_loader,val_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
        pred_size = 1, batch_size = batch_size, num_workers = 2, pin_memory = False, use_coords = True, use_time = True,\
        test_mode = False, scale = scale, window_size = window_size, patch_size = patch_size)


    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.

        for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(train_loader):
            src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                            src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
            optimizer.zero_grad()
            dec_inp = torch.zeros([tgt.shape[0], x1*x2*x3, tgt.shape[-1]]).float().to(device)
            dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-x1*x2*x3),:], dec_inp], dim=1).float().to(device)
            output = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)
            loss = criterion(output[:,-x1*x2*x3:,:], tgt[:,-x1*x2*x3:,:])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = total_loss*batch_size/len(train_loader.dataset)
        val_loss, r2, explained_variance = evaluate(model, val_loader, criterion, patch_size=patch_size)


        #print(f'Epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}', flush=True)
            
        tune.report(train_loss = train_loss, val_loss = val_loss, r2 = r2, explained_variance = explained_variance, epoch=epoch)

        writer.add_scalar('train_loss',train_loss,epoch)
        writer.add_scalar('val_loss',val_loss,epoch)

        Early_Stopping(model, val_loss)

        counter_new = Early_Stopping.counter
        if counter_new != counter_old:
            scheduler.step()
            counter_old = counter_new
        if Early_Stopping.early_stop:
            break
        


if __name__ == "__main__":
    """
    Notes:
    num_samples: number of trails we plan to do
    config_1: tunes for the combination of patch_size, batch_size, window_size, and lr, which should be tuned with priority
    config_2: tunes for pe_type and the model parameters, which should be tuned after we get the best setting in config_1
    """
    num_samples = 100
    config_1 = {
        'window_size':tune.randint(5,40), 
        'patch_x1':tune.choice([1,2,4,16]),
        'patch_x2':tune.choice([1,2,4,16]),
        'patch_x3':tune.choice([1,2,4,16]),
        'batch_size':tune.randint(1,17),
        'pe_type':tune.grid_search(['3d']),
        'feature_size':tune.grid_search([120]),
        #'num_enc_layers':tune.grid_search([1]),
        #'num_dec_layers':tune.grid_search([1]),
        #'num_head':tune.grid_search([2]),
        #'dropout':tune.grid_search([0.1,0.2]),
        #'d_ff':tune.grid_search([512])
        #'lr':tune.grid_search([0.0001]),
        #'window_size':tune.grid_search([12,36,108,324]),
        #'batch_size':tune.grid_search([16])
}   

    config_2 = {
        'window_size':tune.grid_search([10]), 
        'patch_x1':tune.grid_search([1]),
        'patch_x2':tune.grid_search([1]),
        'patch_x3':tune.grid_search([1]),
        'batch_size':tune.grid_search([16]),
        'pe_type':tune.grid_search(['1d','3d', '3d_temporal']),
        'feature_size':tune.grid_search([120, 240]),
        #'num_enc_layers':tune.grid_search([1]),
        #'num_dec_layers':tune.grid_search([1]),
        #'num_head':tune.grid_search([2]),
        #'dropout':tune.grid_search([0.1,0.2]),
        #'d_ff':tune.grid_search([512])
        #'lr':tune.grid_search([0.0001]),
        #'window_size':tune.grid_search([12,36,108,324]),
        #'batch_size':tune.grid_search([16])
}   


    ray.init(ignore_reinit_error=False, include_dashboard=True, dashboard_host='0.0.0.0')
    sched = ASHAScheduler(
            max_t=200,
            grace_period=20,
            reduction_factor=2)
    analysis = tune.run(tune.with_parameters(train), config=config_1, num_samples=num_samples, metric='val_loss', mode='min',\
          scheduler=sched, resources_per_trial={"cpu": 32,"gpu": 4},local_dir="/scratch/zh2095/ray_results")

    best_trail = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_trail, flush=True)
    ray.shutdown()