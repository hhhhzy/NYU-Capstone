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
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----')
    
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
# def process_one_batch(batch_x, batch_y,data_coords,target_coords,data_timestamps,target_timestamps):
#         batch_x = batch_x.float().to(device)
#         batch_y = batch_y.float().to(device)

#         # decoder input
#         dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]]).float().to(device)
#         dec_inp = torch.cat([batch_y[:,:(window_size-1),:], dec_inp], dim=1).float().to(device)
#         # encoder - decoder
#         outputs = model(batch_x, dec_inp,data_coords,target_coords,data_timestamps,target_timestamps)

#         return outputs, batch_y

# def process_one_batch(batch_x, batch_y,data_coords,target_coords):
#         batch_x = batch_x.float().to(device)
#         batch_y = batch_y.float().to(device)

#         # decoder input
#         dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]]).float().to(device)
#         dec_inp = torch.cat([batch_y[:,:(window_size-1),:], dec_inp], dim=1).float().to(device)
#         # encoder - decoder
#         outputs = model(batch_x, dec_inp,data_coords,target_coords)

#         return outputs, batch_y

# def evaluate(model,data_loader,criterion):
#     model.eval()    
#     test_rollout = torch.Tensor(0)   
#     test_result = torch.Tensor(0)  
#     truth = torch.Tensor(0)
#     total_loss = 0.
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#     with torch.no_grad():
#         for i, ((data, targets),(data_coords, target_coords), (data_timestamps, target_timestamps)) in enumerate(data_loader):

#             enc_in, dec_in, targets,data_coords,target_coords,data_timestamps,target_timestamps = data.to(device), targets.to(device), targets.to(device),\
#                 data_coords.to(device),target_coords.to(device),data_timestamps.to(device),target_timestamps.to(device)
#             output, _ = process_one_batch(enc_in,dec_in,data_coords,target_coords,data_timestamps,target_timestamps)
#             total_loss += criterion(output[:,-1:,:], targets[:,-1:,:]).detach().cpu().numpy()
#             test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)

#     return total_loss

# def evaluate(model,data_loader,criterion):
#     model.eval()    
#     test_rollout = torch.Tensor(0)   
#     test_result = torch.Tensor(0)  
#     truth = torch.Tensor(0)
#     total_loss = 0.
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#     with torch.no_grad():
#         for i, ((data, targets),(data_coords, target_coords)) in enumerate(data_loader):

#             enc_in, dec_in, targets,data_coords,target_coords= data.to(device), targets.to(device), targets.to(device),\
#                 data_coords.to(device),target_coords.to(device)
#             output, _ = process_one_batch(enc_in,dec_in,data_coords,target_coords)
#             total_loss += criterion(output[:,-1:,:], targets[:,-1:,:]).detach().cpu().numpy()
#             test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)

#     return total_loss

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
            model = nn.parallel.DistributedDataParallel(model)
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
        
    return total_loss

# def predict_model(model, test_loader, window_size, epoch, plot=True):
#     model.eval()
#     test_rollout = torch.Tensor(0)   
#     test_result = torch.Tensor(0)  
#     truth = torch.Tensor(0)
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             model = nn.parallel.DistributedDataParallel(model)
#     with torch.no_grad():
#         for i, ((data, targets),(data_coords, target_coords), (data_timestamps, target_timestamps)) in enumerate(test_loader):
#             if i == 0:
#                 enc_in = data
#                 dec_in = targets
#                 test_rollout = targets
#             else:
#                 enc_in = test_rollout[:,-window_size:,:]
#                 dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
#                 dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
#             enc_in, dec_in, targets,data_coords,target_coords,data_timestamps,target_timestamps= enc_in.to(device), dec_in.to(device), targets.to(device),\
#                 data_coords.to(device),target_coords.to(device),data_timestamps.to(device),target_timestamps.to(device)
#             output = model(enc_in, dec_in,data_coords,target_coords,data_timestamps,target_timestamps)

#             test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
#             test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
#             truth = torch.cat((truth, targets[:,-1,:].view(-1).detach().cpu()), 0)
            
    # if plot==True:
    #     fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    #     ax.plot(test_result,label='forecast')
    #     ax.plot(truth,label = 'truth')
    #     ax.plot(test_result-truth,ls='--',label='residual')
    #     #ax.grid(True, which='both')
    #     ax.axhline(y=0)
    #     ax.legend(loc="upper right")
    #     fig.savefig(root_dir + f'/figs/transformer_epoch{epoch}_pred.png')
    #     plt.close(fig)

def predict_model(model, test_loader, window_size, epoch, plot=True):
    model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.parallel.DistributedDataParallel(model)
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
        # for i, ((data, targets),(data_coords, target_coords)) in enumerate(test_loader):
        #     if i == 0:
        #         enc_in = data
        #         dec_in = targets
        #         test_rollout = targets
        #     else:
        #         enc_in = test_rollout[:,-window_size:,:]
        #         dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
        #         dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
        #     enc_in, dec_in, targets,data_coords,target_coords= enc_in.to(device), dec_in.to(device), targets.to(device),\
        #         data_coords.to(device),target_coords.to(device)
        #     output = model(enc_in, dec_in,data_coords,target_coords)

        #     test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
        #     test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
        #     truth = torch.cat((truth, targets[:,-1,:].view(-1).detach().cpu()), 0)
            
    if plot==True:
        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        ax.plot(scaler.inverse_transform(test_result.numpy()[:1000]),label='forecast')
        ax.plot(scaler.inverse_transform(truth.numpy()[:1000]),label = 'truth')
        ax.plot(scaler.inverse_transform(test_result.numpy()[:1000])-scaler.inverse_transform(truth.numpy()[:1000]),ls='--',label='residual')
        #ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        fig.savefig(root_dir + f'/figs/transformer_epoch{epoch}_pred.png')
        plt.close(fig)
######### TRY PLOT ENTIRE DATA
        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        ax.plot(scaler.inverse_transform(test_result.numpy()),label='forecast')
        ax.plot(scaler.inverse_transform(truth.numpy()),label = 'truth')
        ax.plot(scaler.inverse_transform(test_result.numpy())-scaler.inverse_transform(truth.numpy()),ls='--',label='residual')
        #ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        fig.savefig(root_dir + f'/figs/transformer_epoch{epoch}_full_pred.png')
        plt.close(fig)


  
    


if __name__ == "__main__":
    print(f'Pytorch version {torch.__version__}')
    root_dir = '/scratch/yd1008/nyu-capstone/notebooks/turbulence_16_yd/tune_results/'
    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
  
    best_config = {'input_type': 'space', 'pe_type': '3d', 'patch_size': 1, 'feature_size': 120, 'num_enc_layers': 1, 'num_dec_layers': 1,\
                     'num_head': 2, 'd_ff': 512, 'dropout': 0.1, 'lr': 1e-5, 'lr_decay': 0.9, 'window_size': 144, 'batch_size': 16}
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    grid_size = 16

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
    lr_decay = best_config['lr_decay']
    window_size = best_config['window_size']
    batch_size = best_config['batch_size']
    

    model = Tranformer(feature_size=feature_size,num_enc_layers=num_enc_layers,num_dec_layers = num_dec_layers,\
        d_ff = d_ff, dropout=dropout,num_head=num_head,pe_type=pe_type,grid_size=grid_size)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.parallel.DistributedDataParallel(model)
    print('Using device: ',device)
    model.to(device)
      
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    writer = tensorboard.SummaryWriter('/scratch/yd1008/tensorboard_output/')
    
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    #     model_state, optimizer_state = torch.load(checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
    
    ### SPECIFYING use_coords=True WILL RETURN DATALOADERS FOR COORDS
    train_loader, test_loader, scaler = get_data_loaders(train_proportion, test_proportion, val_proportion,\
            window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 2, pin_memory = True,\
            use_coords = True, use_time = True, test_mode = True, input_type = input_type, patch_size=patch_size)

    print(f'Total of {len(train_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set')
    print(f'Training with pe type: {pe_type} and input type: {input_type}')
    epochs = 20
    train_losses = []
    test_losses = []
    tolerance = 10
    best_test_loss = float('inf')
    Early_Stopping = early_stopping(patience=20)
    counter_old = 0
    skip_training = False

    if os.path.exists(root_dir+'/best_model.pth') and skip_training:
        model.load_state_dict(torch.load(root_dir+'/best_model.pth'))
    else:
        tmp_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train() 
            total_loss = 0.
            start_time = time.time()
            #for i,((data, targets),(data_coords, target_coords), (data_timestamps, target_timestamps)) in enumerate(train_loader):
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
                #print(f'Saving prediction for epoch {epoch}')
                predict_model(model, test_loader, window_size, epoch, plot=True)  
                #print(f'Predict takes {time.time()-tmp_time}')  

            train_losses.append(total_loss*batch_size/len(train_loader.dataset))
            test_loss = evaluate(model, test_loader, criterion)
            test_losses.append(test_loss/len(test_loader.dataset))
            
            print(f'Epoch: {epoch}, train_loss: {total_loss*batch_size/len(train_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}, time_elapsed: {time.time()-start_time}',flush=True)

            Early_Stopping(model, test_loss/len(test_loader))

            counter_new = Early_Stopping.counter
            if counter_new != counter_old:
                scheduler.step()  # update lr if early stop
                counter_old = counter_new

            if Early_Stopping.early_stop:
                break

            writer.add_scalar('train_loss',total_loss,epoch)
            writer.add_scalar('val_loss',test_loss,epoch)

        if os.path.exists(root_dir + '/best_model.pth'):  # checking if there is a file with this name
            os.remove(root_dir + '/best_model.pth')  # deleting the file
            torch.save(model.state_dict(), root_dir + '/best_model.pth')
        else:
            torch.save(model.state_dict(), root_dir + '/best_model.pth')
### Plot losses        
    model = Early_Stopping.best_model
    xs = np.arange(len(train_losses))
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,train_losses)
    fig.savefig(root_dir + 'figs/transformer_train_loss.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,test_losses)
    fig.savefig(root_dir + 'figs/transformer_test_loss.png')
    plt.close(fig)
### Predict
    model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)

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
            

    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(test_result,label='forecast')
    ax.plot(truth,label = 'truth')
    ax.plot(test_result-truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/transformer_pred.png')
    plt.close(fig)

### Check MSE, MAE
    test_result = test_result.numpy()
    test_result = scaler.inverse_transform(test_result)
    truth = truth.numpy()
    truth = scaler.inverse_transform(truth)
    RMSE = mean_squared_error(truth, test_result)**0.5
    MAE = mean_absolute_error(truth, test_result)
    print(f'RMSE: {RMSE}, MAE: {MAE}')
    
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(test_result,label='forecast')
    ax.plot(truth,label = 'truth')
    ax.plot(test_result-truth,ls='--',label='residual')
    #ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + 'figs/transformer_inverse_prediction.png')
### Save model result
    test_result_df = pd.DataFrame(test_result,columns=['predictions'])
    test_result_df.to_csv(root_dir + 'transformer_prediction.csv')
