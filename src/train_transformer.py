#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import time
import os
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
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


def evaluate_old(model,data_loader,criterion):
    model.eval()
    total_loss = 0.
    rmse = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    for (data,targets) in data_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data, targets)
        total_loss += criterion(output, targets).detach().cpu().numpy()
    return total_loss


def evaluate(model,data_loader,criterion, input_type='patch', patch_size=2):
    # model.eval()    
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    total_loss = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        if input_type == 'patch':
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(data_loader):
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                            src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                dec_inp = torch.zeros([tgt.shape[0], patch_size**3, tgt.shape[-1]]).float().to(device)
                dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-patch_size**3),:], dec_inp], dim=1).float().to(device)
                output = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)
                total_loss += criterion(output[:,-patch_size**3:,:], tgt[:,-patch_size**3:,:]).detach().cpu().numpy()
        
        else:
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(data_loader):
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                            src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                dec_inp = torch.zeros([tgt.shape[0], 1, tgt.shape[-1]]).float().to(device)
                dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-1),:], dec_inp], dim=1).float().to(device)
                output = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)
                total_loss += criterion(output[:,-1:,:], tgt[:,-1:,:]).detach().cpu().numpy()
        
    return total_loss



def train(config, checkpoint_dir):
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2

    input_type = config['input_type']
    pe_type = config['pe_type']
    feature_size = config['feature_size']
    num_enc_layers = config['num_enc_layers']
    num_dec_layers = config['num_dec_layers']
    num_head = config['num_head']
    
    patch_size = 2
    dropout = 0.1 #config['dropout']
    d_ff = 2048
    lr = 1e-5 #config['lr']
    window_size = 10 #config['window_size']
    batch_size = 256 #config['batch_size']
    
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    writer = tensorboard.SummaryWriter('./test_logs')
    
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    #     model_state, optimizer_state = torch.load(checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
        
    train_loader,val_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
        window_size = window_size, pred_size = 1, batch_size = batch_size, num_workers = 2, pin_memory = False,\
        use_coords = True, use_time = True, test_mode = False, input_type = input_type, patch_size=patch_size)

    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.
           
        if input_type == 'patch':
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(train_loader):
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                                src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                optimizer.zero_grad()
                dec_inp = torch.zeros([tgt.shape[0], patch_size**3, tgt.shape[-1]]).float().to(device)
                dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-patch_size**3),:], dec_inp], dim=1).float().to(device)
                output = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)
                loss = criterion(output[:,-patch_size**3:,:], tgt[:,-patch_size**3:,:])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
        else:
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(train_loader):
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                                src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                optimizer.zero_grad()
                dec_inp = torch.zeros([tgt.shape[0], 1, tgt.shape[-1]]).float().to(device)
                dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-1),:], dec_inp], dim=1).float().to(device)
                output = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)
                loss = criterion(output[:,-1,:], tgt[:,-1,:])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()



        val_loss = evaluate(model, val_loader, criterion, input_type=input_type, patch_size=patch_size)
        print(f'Epoch: {epoch}, train_loss: {total_loss}, val_loss: {val_loss}', flush=True)

        writer.add_scalar('train_loss',total_loss,epoch)
        writer.add_scalar('val_loss',val_loss,epoch)
        
        tune.report(val_loss=val_loss)

        Early_Stopping(model, val_loss/len(val_loader.dataset))

        counter_new = Early_Stopping.counter
        if counter_new != counter_old:
            scheduler.step()
            counter_old = counter_new

        if Early_Stopping.early_stop:
            break


if __name__ == "__main__":
    config = {
        'input_type':tune.grid_search(['space', 'time', 'patch']),
        'pe_type':tune.grid_search(['1d','3d','3d_temporal']),
        'feature_size':tune.grid_search([192,768]),
        'num_enc_layers':tune.grid_search([1]),
        'num_dec_layers':tune.grid_search([1]),
        'num_head':tune.grid_search([2]),
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
    analysis = tune.run(tune.with_parameters(train), config=config, metric='val_loss', mode='min',\
         scheduler=sched, resources_per_trial={"cpu": 32,"gpu": 4},local_dir="/scratch/zh2095/ray_results",)

    best_trail = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_trail, flush=True)
    ray.shutdown()