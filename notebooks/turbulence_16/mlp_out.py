#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import time
import os
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from _arfima import arfima
from mlp import MLP
from utils import *



def evaluate(model,data_loader,criterion):
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
        output = model(data)
        total_loss += criterion(output, targets).detach().cpu().numpy()
    return total_loss
    


if __name__ == "__main__":
    best_config = {'hidden_size': 128, 'num_hidden_layers': 2, 'dropout': 0.1, 'lr': 0.001}
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    hidden_size = config['hidden_size']
    num_hidden_layers = config['num_hidden_layers']
    dropout = config['dropout']
    lr = config['lr']
    window_size = 12#config['window_size']
    batch_size = 16#config['batch_size']
    
    model = MLP(1,1,hidden_size,num_hidden_layers, dropout)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    print('Using device: ',device)
    model.to(device)
            
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    writer = tensorboard.SummaryWriter('/scratch/yd1008/tensorboard_output/')
    
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    #     model_state, optimizer_state = torch.load(checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
        
    train_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
         window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 2, pin_memory = False, test_mode = True)

    epochs = 600
    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.

        for (data, targets) in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        val_loss = evaluate(model, test_loader, criterion)
        if epoch==1: ###DEBUG
            print(f'Total of {len(train_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set')
        print(f'Epoch: {epoch}, train_loss: {total_loss/len(train_loader.dataset)}, test_loss: {val_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}')
        writer.add_scalar('train_loss',total_loss,epoch)
        writer.add_scalar('val_loss',val_loss,epoch)
        scheduler.step() 
