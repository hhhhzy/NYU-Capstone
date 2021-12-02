#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
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
            print(f'----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----')
    
def process_one_batch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts, patch_size):
    x1, x2, x3 = patch_size
    dec_inp = torch.zeros([tgt.shape[0], x1*x2*x3, tgt.shape[-1]]).float().to(device)
    dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-x1*x2*x3),:], dec_inp], dim=1).float().to(device)
    outputs = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)

    return outputs, tgt

def evaluate(model,data_loader,criterion, patch_size):
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
            src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                            src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
            dec_inp = torch.zeros([tgt.shape[0], x1*x2*x3, tgt.shape[-1]]).float().to(device)
            dec_inp = torch.cat([tgt[:,:(tgt.shape[1]-x1*x2*x3),:], dec_inp], dim=1).float().to(device)
            output = model(src, dec_inp, src_coord, tgt_coord, src_ts, tgt_ts)
            total_loss += criterion(output[:,-x1*x2*x3:,:], tgt[:,-x1*x2*x3:,:]).detach().cpu().numpy()

    return total_loss

def predict_model(model, test_loader, epoch, config={},\
                    plot=True, plot_range=[0,0.01], final_prediction=False):
    '''
    Note: 
        Don't forget to create a subfolder 'final_plot' under 'figs'
    parameters:
        plot_range: [a,b], 0<=a<b<=1, where a is the proportion that determines the start point to plot, b determines the end point. 
        final_prediction: True, if done with training and using the trained/saved model to predict the final result
        config: dictionary, the config of this plot, used for saving distinguishable plots for each trail
    '''
    model.eval()
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0) 
    truth = torch.Tensor(0) 
    test_ts = torch.Tensor(0)
    test_coord = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    with torch.no_grad():
        for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(test_loader):
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
            test_ts = torch.cat((test_ts, tgt_ts[:,-x1*x2*x3:,:].flatten().detach().cpu()), 0)
            test_coord = torch.cat((test_coord, tgt_coord[:,-x1*x2*x3:,:].reshape(-1,3).detach().cpu()), 0)
            truth = torch.cat((truth, tgt[:,-x1*x2*x3:,:].flatten().detach().cpu()), 0)
            test_result = torch.cat((test_result, output[:,-x1*x2*x3:,:].flatten().detach().cpu()), 0)
        a = torch.cat([test_ts.unsqueeze(-1), test_coord, test_result.unsqueeze(-1), truth.unsqueeze(-1)], dim=-1)
        a = a.numpy()
        a = a[np.argsort(a[:, 3])]
        a = a[np.argsort(a[:, 2], kind='stable')]
        a = a[np.argsort(a[:, 1], kind='stable')]
        a = a[np.argsort(a[:, 0], kind='stable')]
        if scale==True:
            final_result = {'time': a[:,0], 'x1': a[:,1], 'x2': a[:,2], 'x3': a[:,3], 'prediction': scaler.inverse_transform(a[:,4]), 'truth':scaler.inverse_transform(a[:,5])}
        elif scale==False:
            final_result = {'time': a[:,0], 'x1': a[:,1], 'x2': a[:,2], 'x3': a[:,3], 'prediction': a[:,4], 'truth':a[:,5]}
    
    if plot==True:
        # plot a part of the result
        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        plot_start = int(len(final_result['prediction'])*plot_range[0])
        plot_end = int(len(final_result['prediction'])*plot_range[1])
        ax.plot(final_result['truth'][plot_start:plot_end],label = 'truth')
        ax.plot(final_result['prediction'][plot_start:plot_end],label='forecast')
        ax.plot(final_result['prediction'][plot_start:plot_end] - final_result['truth'][plot_start:plot_end],ls='--',label='residual')            
        #ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        if final_prediction == True:
            fig.savefig(root_dir + '/figs/final_plot' + f"/range{plot_range}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}.png")
        else:
            fig.savefig(root_dir + '/figs/tmp_plot' + f"/pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}_epoch{epoch}.png")
            if epoch == config['epochs']:
                fig.savefig(root_dir + '/figs/final_plot'+ f"/range{plot_range}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}.png")
        plt.close(fig)
    if final_prediction == True:
        return final_result


  
    


if __name__ == "__main__":
    print(f'Pytorch version {torch.__version__}')
    root_dir = '/scratch/yd1008/nyu-capstone/notebooks/turbulence_16_yd/tune_results/'
    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
    plt.rcParams['animation.ffmpeg_path'] = '/ext3/conda/bootcamp/bin/ffmpeg'
  
    best_config = {'epochs':5, 'window_size': 5, 'patch_size': (6,6,6), 'pe_type': '3d_temporal', 'batch_size': 32, 'scale': True,'feature_size': 300\
                , 'num_enc_layers': 2, 'num_dec_layers': 2, 'num_head': 4, 'd_ff': 512, 'dropout': 0.1, 'lr': 1e-6, 'lr_decay': 0.8, 'option': 'patch_overlap'}
    
    window_size = best_config['window_size']
    patch_size = best_config['patch_size']
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
    scale = best_config['scale']
    option = best_config['option']

    # dataset parameters
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    grid_size = 16
    
    skip_training = False
    save_model = True

    print('-'*20 + ' Config ' + '-'*20, flush=True)
    print(best_config, flush=True)
    print('-'*50, flush=True)

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
        pred_size = 1, batch_size = batch_size, num_workers = 2, pin_memory = False, use_coords = True, use_time = True,\
        test_mode = True, scale = scale, window_size = window_size, patch_size = patch_size, option = option)
    
    epochs = best_config['epochs']
    train_losses = []
    test_losses = []
    tolerance = 10
    best_test_loss = float('inf')
    Early_Stopping = early_stopping(patience=5)
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
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), \
                                                                src_coord.to(device), tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                optimizer.zero_grad()
                output, truth = process_one_batch(src, tgt, src_coord, tgt_coord, src_ts, tgt_ts, patch_size)
                loss = criterion(output[:,-x1*x2*x3:,:], tgt[:,-x1*x2*x3:,:])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss*batch_size/len(train_loader.dataset)
            total_test_loss = evaluate(model, test_loader, criterion, patch_size=patch_size)
            avg_test_loss = total_test_loss/len(test_loader.dataset)

            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            

            if epoch==1: ###DEBUG
                print(f'Total of {len(train_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set', flush=True)

            print(f'Epoch: {epoch}, train_loss: {avg_train_loss}, test_loss: {avg_test_loss}, lr: {scheduler.get_last_lr()}, training time: {time.time()-start_time} s', flush=True)

            if (epoch%2 == 0):
                print(f'Saving prediction for epoch {epoch}', flush=True)
                predict_model(model, test_loader, epoch, config=best_config,\
                                    plot=True, plot_range=[0,0.01], final_prediction=False)   

            writer.add_scalar('train_loss',avg_train_loss,epoch)
            writer.add_scalar('test_loss',avg_test_loss,epoch)
            
            Early_Stopping(model, avg_test_loss)

            counter_new = Early_Stopping.counter
            if counter_new != counter_old:
                scheduler.step()  # update lr if early stop
                counter_old = counter_new
            if Early_Stopping.early_stop:
                break

        #save model
        if save_model:
            if os.path.exists(root_dir + '/best_model.pth'):  # checking if there is a file with this name
                os.remove(root_dir + '/best_model.pth')  # deleting the file
                torch.save(model.state_dict(), root_dir + '/best_model.pth')
            else:
                torch.save(model.state_dict(), root_dir + '/best_model.pth')
### Plot losses        
    xs = np.arange(len(train_losses))
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,train_losses)
    fig.savefig(root_dir + '/figs/loss' + f"/train_loss_pe{best_config['pe_type']}_batch{best_config['batch_size']}_window{best_config['window_size']}_patch{best_config['patch_size']}.png")
    plt.close(fig)
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,test_losses)
    fig.savefig(root_dir + '/figs/loss' + f"/test_loss_pe{best_config['pe_type']}_batch{best_config['batch_size']}_window{best_config['window_size']}_patch{best_config['patch_size']}.png")
    plt.close(fig)

### Predict
    start_time = time.time()
    final_result = predict_model(model, test_loader,  epoch, config=best_config,\
                                            plot=True, plot_range=[0,1], final_prediction=True)
    prediction = final_result['prediction']
    print('-'*20 + ' Measure for Simulation Speed ' + '-'*20)
    print(f'Time to forcast {len(prediction)} samples: {time.time()-start_time} s' )

### Check MSE, MAE
    truth = final_result['truth']
    MSE = mean_squared_error(truth, prediction)
    MAE = mean_absolute_error(truth, prediction)
    R2 = r2_score(truth, prediction)
    EXPLAINED_VARIANCE_SCORE = explained_variance_score(truth, prediction)
    print('-'*20 + ' Measure for Simulation Performance ' + '-'*20)
    print(f'MSE: {MSE}, MAE: {MAE}, R2: {R2}, EXPLAINED_VARIANCE_SCORE: {EXPLAINED_VARIANCE_SCORE}')

### Save model result
    test_result_df = pd.DataFrame.from_dict(final_result)
    print('-'*20 + ' Dataframe for Final Result ' + '-'*20)
    print(test_result_df)
    test_result_df.to_csv(root_dir + '/transformer_prediction.csv')

### slice plot
    img_dir = root_dir + '/slice_plot' 
    pred_df = pd.read_csv(root_dir + '/transformer_prediction.csv',index_col=0)
    grid_size = [16,16,16]
    axis_colnames = ['x1','x2','x3']
    slice_axis_index = 0
    pred_colname = 'prediction'
    truth_colname = 'truth'
    time_colname = 'time'
    plot_forecast(pred_df=pred_df, grid_size=grid_size, axis_colnames=axis_colnames, slice_axis_index=2, \
                        pred_colname=pred_colname,truth_colname=truth_colname, time_colname=time_colname,  \
                        plot_anime = True, img_dir = img_dir, config=best_config) 