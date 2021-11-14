#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
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
                print('Early stopping')
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----', flush=True)


def process_one_batch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts):
    dec_inp = torch.zeros([tgt.shape[0], 1, tgt.shape[-1]]).float().to(device)
    dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-1),:], dec_inp], dim=1).float().to(device)
    outputs = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)

    return outputs, tgt

def process_one_batch_patch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts):
    dec_inp = torch.zeros([tgt.shape[0], patch_size**3, tgt.shape[-1]]).float().to(device)
    dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-patch_size**3),:], dec_inp], dim=1).float().to(device)
    outputs = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)

    return outputs, tgt


def evaluate(model,data_loader,criterion, input_type='patch'):
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
                output, _ = process_one_batch_patch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts)
                total_loss += criterion(output[:,-patch_size**3:,:], tgt[:,-patch_size**3:,:]).detach().cpu().numpy()
                test_rollout = torch.cat([test_rollout,output[:,-patch_size**3:,:].detach().cpu()],dim = 1)
        
        else:
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(data_loader):
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                            src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                output, _ = process_one_batch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts)
                total_loss += criterion(output[:,-1:,:], tgt[:,-1:,:]).detach().cpu().numpy()
                test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
        
    return total_loss, test_rollout[:,-10:,:]



def predict_model(model, test_loader, window_size, epoch, input_type='patch', plot=True):
    # model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        if input_type == 'patch':
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(test_loader):
                if i == 0:
                    enc_in = src
                    dec_in = tgt
                    test_rollout = tgt
                else:
                    enc_in = test_rollout[:,-tgt.shape[1]:,:]
                    dec_in = torch.zeros([enc_in.shape[0], patch_size**3, enc_in.shape[-1]]).float()
                    dec_in = torch.cat([enc_in[:,:(tgt.shape[1]-patch_size**3),:], dec_in], dim=1).float()
                    #dec_in = enc_in[:,:(window_size-1),:]
                enc_in, dec_in, tgt = enc_in.to(device), dec_in.to(device), tgt.to(device)
                src_coord, tgt_coord, src_ts, tgt_ts = src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts)

                test_rollout = torch.cat([test_rollout,output[:,-patch_size**3:,:].detach().cpu()],dim = 1)
                test_result = torch.cat((test_result, output[:,-patch_size**3:,:].view(-1).detach().cpu()), 0)
                truth = torch.cat((truth, tgt[:,-patch_size**3:,:].view(-1).detach().cpu()), 0)
        
        else:
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(test_loader):
                if i == 0:
                    enc_in = src
                    dec_in = tgt
                    test_rollout = tgt
                else:
                    enc_in = test_rollout[:,-window_size:,:]
                    dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                    dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
                    #dec_in = enc_in[:,:(window_size-1),:]
                enc_in, dec_in, tgt = enc_in.to(device), dec_in.to(device), tgt.to(device)
                src_coord, tgt_coord, src_ts, tgt_ts = src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts)

                test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
                test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
                truth = torch.cat((truth, tgt[:,-1,:].view(-1).detach().cpu()), 0)
            
    
    if plot==True:
        # plot the last 1000 samples
        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        ax.plot(test_result[-1000:],label='forecast')
        ax.plot(truth[-1000:],label = 'truth')
        ax.plot(test_result[-1000:]-truth[-1000:],ls='--',label='residual')
        #ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        fig.savefig(root_dir + f'/figs/epoch{epoch}_{pe_type}_{input_type}.png')
        plt.close(fig)

    
  
    


if __name__ == "__main__":
    print(f'Pytorch version {torch.__version__}')
    root_dir = '/scratch/zh2095/tune_results'
    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
    best_config = {'input_type': 'patch', 'pe_type': '3d_temporal', 'patch_size': 2, 'feature_size': 192, 'num_enc_layers': 1, 'num_dec_layers': 1,\
                     'num_head': 2, 'd_ff': 2048, 'dropout': 0.1, 'window_size': 10, 'lr':1e-5, 'batch_size': 256}
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2

    patch_size = best_config['patch_size']
    input_type = best_config['input_type']
    pe_type = best_config['pe_type']
    feature_size = best_config['feature_size']
    num_enc_layers = best_config['num_enc_layers']
    num_dec_layers = best_config['num_dec_layers']
    d_ff = best_config['d_ff']
    num_head = best_config['num_head']
    dropout = best_config['dropout']
    lr = best_config['lr']
    window_size = best_config['window_size']
    batch_size = best_config['batch_size']

    model = Tranformer(feature_size=feature_size,num_enc_layers=num_enc_layers,num_dec_layers = num_dec_layers,\
            d_ff = d_ff, dropout=dropout,num_head=num_head,pe_type=pe_type)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    print('Using device: ',device)
    model.to(device)
                
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    writer = tensorboard.SummaryWriter('/scratch/zh2095/tensorboard_output/')
        
        # if checkpoint_dir:
        #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        #     model_state, optimizer_state = torch.load(checkpoint)
        #     model.load_state_dict(model_state)
        #     optimizer.load_state_dict(optimizer_state)
            
    train_loader, test_loader, scaler = get_data_loaders(train_proportion, test_proportion, val_proportion,\
            window_size = window_size, pred_size = 1, batch_size = batch_size, num_workers = 2, pin_memory = False,\
            use_coords = True, use_time = True, test_mode = True, input_type = input_type, patch_size=patch_size)

    epochs = 10
    train_losses = []
    test_losses = []
    tolerance = 10
    best_test_loss = float('inf')
    Early_Stopping = early_stopping(patience=5)
    counter_old = 0
    skip_training = False

    if os.path.exists(root_dir+'/best_model.pth') and skip_training:
        model.load_state_dict(torch.load(root_dir+'/best_model.pth'))
    else:
        for epoch in range(1, epochs + 1):    
            model.train() 
            total_loss = 0.
            start_time = time.time()
            if input_type == 'patch':
                for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(train_loader):
                    src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                                src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                    optimizer.zero_grad()
                    output, truth = process_one_batch_patch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts)
                    loss = criterion(output[:,-patch_size**3:,:], tgt[:,-patch_size**3:,:])
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
            else:
                for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(train_loader):
                    src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                                src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                    optimizer.zero_grad()
                    output, truth = process_one_batch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts)
                    loss = criterion(output[:,-1,:], tgt[:,-1,:])
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()


            if (epoch%2 == 0):
                print(f'Saving prediction for epoch {epoch}', flush=True)
                predict_model(model, test_loader, window_size, epoch, input_type=input_type, plot=True)    


            train_losses.append(total_loss*batch_size)
            test_loss, debug_output = evaluate(model, test_loader, criterion, input_type=input_type)
            test_losses.append(test_loss/len(test_loader.dataset))


            if epoch==1: ###DEBUG
                print(f'Total of {len(train_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set', flush=True)


            print(f'Epoch: {epoch}, train_loss: {total_loss*batch_size/len(train_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}', flush=True)
            print(f'Training time for 1 epoch with pe type {pe_type} and input type {input_type}: , {time.time()-start_time}', flush=True)

            Early_Stopping(model, test_loss/len(test_loader))

            counter_new = Early_Stopping.counter
            if counter_new != counter_old:
                scheduler.step()
                counter_old = counter_new

            if Early_Stopping.early_stop:
                break


            writer.add_scalar('train_loss',total_loss,epoch)
            writer.add_scalar('test_loss',test_loss,epoch)


            #if epoch%2 == 0:
                #scheduler.step() 

        #save model
        if os.path.exists(root_dir + '/best_model.pth'):  # checking if there is a file with this name
            os.remove(root_dir + '/best_model.pth')  # deleting the file
            torch.save(model.state_dict(), root_dir + '/best_model.pth')
        else:
            torch.save(model.state_dict(), root_dir + '/best_model.pth')

### Plot losses        
    xs = np.arange(len(train_losses))
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,train_losses)
    fig.savefig(root_dir + f'/figs/train_loss_{pe_type}_{input_type}.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,test_losses)
    fig.savefig(root_dir + f'/figs/test_loss_{pe_type}_{input_type}.png')
    plt.close(fig)
### Predict
    # model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        if input_type == 'patch':
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(test_loader):
                if i == 0:
                    enc_in = src
                    dec_in = tgt
                    test_rollout = tgt
                else:
                    enc_in = test_rollout[:,-tgt.shape[1]:,:]
                    dec_in = torch.zeros([enc_in.shape[0], patch_size**3, enc_in.shape[-1]]).float()
                    dec_in = torch.cat([enc_in[:,:(tgt.shape[1]-patch_size**3),:], dec_in], dim=1).float()
                    #dec_in = enc_in[:,:(window_size-1),:]
                enc_in, dec_in, tgt = enc_in.to(device), dec_in.to(device), tgt.to(device)
                src_coord, tgt_coord, src_ts, tgt_ts = src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts)

                test_rollout = torch.cat([test_rollout,output[:,-patch_size**3:,:].detach().cpu()],dim = 1)
                test_result = torch.cat((test_result, output[:,-patch_size**3:,:].view(-1).detach().cpu()), 0)
                truth = torch.cat((truth, tgt[:,-patch_size**3:,:].view(-1).detach().cpu()), 0)
        
        else:
            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(test_loader):
                if i == 0:
                    enc_in = src
                    dec_in = tgt
                    test_rollout = tgt
                else:
                    enc_in = test_rollout[:,-window_size:,:]
                    dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                    dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
                    #dec_in = enc_in[:,:(window_size-1),:]
                enc_in, dec_in, tgt = enc_in.to(device), dec_in.to(device), tgt.to(device)
                src_coord, tgt_coord, src_ts, tgt_ts = src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts)

                test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
                test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
                truth = torch.cat((truth, tgt[:,-1,:].view(-1).detach().cpu()), 0)

    ### Plot prediction
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(truth,label = 'truth')
    ax.plot(test_result,label='forecast')
    ax.plot(test_result-truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + '/figs/transformer_pred.png')
    plt.close(fig)


### Check MSE, MAE
    test_result = test_result.numpy()
    truth = truth.numpy()
    MSE = mean_squared_error(truth, test_result)
    MAE = mean_absolute_error(truth, test_result)
    print(f'MSE: {MSE}, MAE: {MAE}')
### Save model result
    test_result_df = pd.DataFrame(test_result)
    test_result_df.to_csv(root_dir + '/transformer_prediction.csv')



###Previous codes

# def evaluate(model,data_loader,criterion):
#     model.eval()
#     total_loss = 0.
#     rmse = 0.
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#     with torch.no_grad():
#         for (data,targets) in data_loader:
#             data, targets = data.to(device), targets.to(device)
#             output = model(data)
#             total_loss += criterion(output, targets).detach().cpu().numpy()
#     return total_loss
    


# if __name__ == "__main__":
#     best_config = {'feature_size': 128, 'num_layer': 2, 'num_head': 2, 'dropout': 0.1}
#     train_proportion = 0.6
#     test_proportion = 0.2
#     val_proportion = 0.2
#     feature_size = best_config['feature_size']
#     num_layer = best_config['num_layer']
#     num_head = best_config['num_head']
#     dropout = best_config['dropout']
#     lr = 0.01#config['lr']
#     window_size = 12#config['window_size']
#     batch_size = 16#config['batch_size']
    
#     model = Tranformer(feature_size=feature_size,num_layers=num_layer,dropout=dropout,num_head=num_head)
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#     print('Using device: ',device)
#     model.to(device)
            
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
#     writer = tensorboard.SummaryWriter('/scratch/yd1008/tensorboard_output/')
    
#     # if checkpoint_dir:
#     #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
#     #     model_state, optimizer_state = torch.load(checkpoint)
#     #     model.load_state_dict(model_state)
#     #     optimizer.load_state_dict(optimizer_state)
        
#     train_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
#          window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 2, pin_memory = False, test_mode = True)

#     epochs = 150
#     train_losses = []
#     test_losses = []
#     tolerance = 10
#     best_test_loss = float('inf')

#     for epoch in range(1, epochs + 1):
#         model.train() 
#         total_loss = 0.

#         for (data, targets) in train_loader:
#             data, targets = data.to(device), targets.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, targets)
#             total_loss += loss.item()
#             loss.backward()
#             optimizer.step()
            
#         train_losses.append(total_loss/len(train_loader.dataset))
#         test_loss = evaluate(model, test_loader, criterion)
#         test_losses.append(test_loss/len(test_loader.dataset))
#         if test_loss <= best_test_loss:
#             pass ##Implement early stopping
#         if epoch==1: ###DEBUG
#             print(f'Total of {len(train_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set')
#         print(f'Epoch: {epoch}, train_loss: {total_loss/len(train_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}')
#         writer.add_scalar('train_loss',total_loss,epoch)
#         writer.add_scalar('val_loss',test_loss,epoch)
#         scheduler.step() 

#     xs = np.arange(len(train_losses))
#     fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
#     ax.plot(xs,train_losses)
#     fig.savefig('/scratch/yd1008/nyu-capstone/tune_results/figs/transformer_train_loss.png')
#     plt.close(fig)
#     fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
#     ax.plot(xs,test_losses)
#     fig.savefig('/scratch/yd1008/nyu-capstone/tune_results/figs/transformer_test_loss.png')
#     plt.close(fig)
# ### Predict
#     model.eval()
#     test_result = torch.Tensor(0)    
#     truth = torch.Tensor(0)
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#     with torch.no_grad():
#         for (data,targets) in test_loader:
#             data, targets = data.to(device), targets.to(device)
#             output = model(data)
#             test_result = torch.cat((test_result, output[-1].view(-1).detach().cpu()), 0)
#             truth = torch.cat((truth, targets[-1].view(-1).detach().cpu()), 0)

#     fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
#     ax.plot(test_result,color="red")
#     ax.plot(truth,color="blue")
#     ax.plot(test_result-truth,color="green")
#     ax.grid(True, which='both')
#     ax.axhline(y=0, color='k')
#     fig.savefig('/scratch/yd1008/nyu-capstone/tune_results/figs/transformer_pred.png')
#     plt.close(fig)