#!/bin/bash python
import torch 
import torch.nn.functional as F
from torch.utils import tensorboard
import torch.optim as optim
from itertools import product
import pandas as pd
import numpy as np
from einops import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import os
from transformer_best import Transformer
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
    
def get_roll_strides(stride,dim=3):
    '''
    stride int: number of pixels to roll on each dimension
    '''
    assert type(stride) in [int,list], f"Expect stride to be type int or list, got {type(stride)}"
    if type(stride) == int:
        return list(set(product([stride,0],repeat=dim)))
    elif type(stride) == list:
        return list(set(product(stride+[0],repeat=dim)))

def evaluate(model,data_loader,criterion, patch_size, predict_res = False):
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
            src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), src_coord.to(device),\
                                                            tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
            if i==0:
                N,D,H,W,C = src.shape
                enc_in = src
                test_rollout = src
            else:
                enc_in = test_rollout[-N:]

             # dec_rollout = reduce(enc_in.view(B,window_size,patch_length,-1), 'b n p c -> b p c', 'mean')
            dec_rollout = torch.zeros_like(enc_in[:pred_size]) 
            dec_in = torch.cat([enc_in[pred_size:], dec_rollout], dim=0).float()
            # dec_in = dec_in + (torch.empty(dec_in.shape).normal_(mean=0,std=noise_std)).to(device)
            output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts,temporal_insert_layer = temporal_insert_layer)
            if predict_res:
                res = tgt - src
                total_loss += criterion(output[-pred_size:,:,:,:,0], res[-pred_size:,:,:,:,0]).detach().cpu().numpy()
                if i==0:
                    print(f'Output: \n {output[0,0,0,:20,0].view(-1)} \n Truth: \n {res[0,0,0,:20,0].view(-1)}')
                output = output + enc_in
            else: 
                total_loss += criterion(output[:,-x1*x2*x3:,0], tgt[:,-x1*x2*x3:,0]).detach().cpu().numpy()

            test_rollout = torch.cat([test_rollout,output[-pred_size:,:,:,:,:]], dim=0)

    return total_loss

def predict_model(model, test_loader, epoch, config={},\
                    plot=True, plot_range=[0,0.01], final_prediction=False, predict_res = False, file_prefix = '', shift_size = [0], pred_feature = 0):
    '''
    Note: 
        Dont forget to create a subfolder final_plot under figs
    parameters:
        plot_range: [a,b], 0<=a<b<=1, where a is the proportion that determines the start point to plot, b determines the end point. 
        final_prediction: True, if done with training and using the trained/saved model to predict the final result
        config: dictionary, the config of this plot, used for saving distinguishable plots for each trail
    '''
    model.eval()
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

            src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), src_coord.to(device),\
                                                                            tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
            
            temp_rollout = []
            for shift_size in get_roll_strides([0]):
                shift_size = (0,) + shift_size
                if i==0:
                    N,D,H,W,C = src.shape
                    enc_in = src
                    test_rollout = src
                    # src_block = src_block.to(device)
                else:
                    enc_in = test_rollout[-N:]
                # dec_rollout = reduce(enc_in.view(B,window_size,patch_length,-1), 'b n p c -> b p c', 'mean')
                dec_rollout = torch.zeros_like(enc_in[:pred_size]) 
                dec_in = torch.cat([enc_in[pred_size:], dec_rollout], dim=0).float()
                # dec_in = dec_in + (torch.empty(dec_in.shape).normal_(mean=0,std=noise_std)).to(device)
                output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts, shift_size, temporal_insert_layer)

                output = output + enc_in
                temp_rollout.append(output)

            output = torch.stack(temp_rollout, dim=0).mean(dim=0)
            test_rollout = torch.cat([test_rollout,output[-pred_size:,:,:,:,:]], dim=0)
            test_ts = torch.cat((test_ts, tgt_ts[-pred_size:,:,:,:,:].flatten().detach().cpu()), 0)
            test_coord = torch.cat((test_coord, tgt_coord[-pred_size:,:,:,:,:].reshape(-1,3).detach().cpu()), 0)
            truth = torch.cat((truth, tgt[-pred_size:,:,:,:,pred_feature].flatten().detach().cpu()), 0)
            test_result = torch.cat((test_result, output[-pred_size:,:,:,:,pred_feature].flatten().detach().cpu()), 0)
        

        # if predict_res:
        #     residuals = pd.DataFrame.from_dict(final_result)
        #     truth = pd.read_csv('tune_results/data_original.csv',index_col=0)
        #     ### Group by time since last timestamp is duplicated
        #     truth.time = np.round(truth.time.values,5)
        #     residuals.time = np.round(residuals.time.values,5)
        #     residuals = residuals.groupby(['time','x1','x2','x3']).mean().reset_index()
        #     truth = truth.groupby(['time','x1','x2','x3']).mean().reset_index()
        #     ### Reconstruct predictions, truth is also reconstructed to validate answer
        #     pred_timestamps = residuals.time.unique()
        #     pred_start_time = residuals.time[0]
        #     pred_start_truth = truth[truth['time']==truth.time.unique()[-len(pred_timestamps)]]
        #     for i,(t1,t2) in enumerate(zip(pred_timestamps,truth.time.unique()[-len(pred_timestamps):])):
        #         truth_values = truth.loc[truth['time']==t2].truth_original.values
        #         pred_values = residuals.loc[residuals['time']==t1].prediction.values
        #         truth_check_values = residuals.loc[residuals['time']==t1].truth.values
        #         if i==0:
        #             prev_truth_plus_residual = pred_values+truth_values
        #         else:
        #             prev_truth_plus_residual = pred_values+prev_truth_plus_residual
        #         residuals.loc[residuals['time']==t1,'prediction']= prev_truth_plus_residual
        #         residuals.loc[residuals['time']==t1,'truth'] = truth_check_values+truth_values
        #     final_result = residuals

    if plot==True:
        a = torch.cat([test_ts.unsqueeze(-1), test_coord, test_result.unsqueeze(-1), truth.unsqueeze(-1)], dim=-1)
        a = a.numpy()
        a = a[np.argsort(a[:, 3])]
        a = a[np.argsort(a[:, 2], kind='stable')]
        a = a[np.argsort(a[:, 1], kind='stable')]
        a = a[np.argsort(a[:, 0], kind='stable')]
        final_result_unscaled = {'time': a[:,0], 'x1': a[:,1], 'x2': a[:,2], 'x3': a[:,3], 'prediction': a[:,4], 'truth':a[:,5]}

        fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
        plot_start = int(len(final_result_unscaled['prediction'])*plot_range[0])
        plot_end = int(len(final_result_unscaled['prediction'])*plot_range[1])
        ax.plot(final_result_unscaled['truth'][plot_start:plot_end],label = 'truth')
        ax.plot(final_result_unscaled['prediction'][plot_start:plot_end],label='forecast')
        # ax.plot(final_result_unscaled['prediction'][plot_start:plot_end] - final_result_unscaled['truth'][plot_start:plot_end],ls='--',label='residual')            
        #ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        # if final_prediction == True:
        fig.savefig(root_dir + '/figs/tmp_plot' + f"/{file_prefix}_unscaled_range{plot_range}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}_epoch{epoch}.png")
        plt.close(fig)
        

        if config['scale']==True:
            test_result = torch.Tensor(scaler.inverse_transform(test_result.unsqueeze(-1)))
            truth = torch.Tensor(scaler.inverse_transform(truth.unsqueeze(-1)))
        else:
            test_result = test_result.unsqueeze(-1)
            truth = truth.unsqueeze(-1)

        a = torch.cat([test_ts.unsqueeze(-1), test_coord, test_result, truth], dim=-1)
        a = a.numpy()
        a = a[np.argsort(a[:, 3])]
        a = a[np.argsort(a[:, 2], kind='stable')]
        a = a[np.argsort(a[:, 1], kind='stable')]
        a = a[np.argsort(a[:, 0], kind='stable')]
        final_result = {'time': a[:,0], 'x1': a[:,1], 'x2': a[:,2], 'x3': a[:,3], 'prediction': a[:,4], 'truth':a[:,5]}
        

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
            fig.savefig(root_dir + '/figs/final_plot' + f"/{file_prefix}_range{plot_range}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}.png")
        else:
            fig.savefig(root_dir + '/figs/tmp_plot' + f"/{file_prefix}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}_epoch{epoch}.png")
            if epoch == config['epochs']:
                fig.savefig(root_dir + '/figs/final_plot'+ f"/{file_prefix}_range{plot_range}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}.png")
        plt.close(fig)
    if final_prediction == True:
        return final_result


  


  
    


if __name__ == "__main__":
    print(f'Pytorch version {torch.__version__}')
    root_dir = '/scratch/yd1008/nyu_capstone_2/notebooks/turbulence_16_yd/tune_results_2/'
    sns.set_style("whitegrid")
    sns.set_palette(['#57068c','#E31212','#01AD86'])
    plt.rcParams['animation.ffmpeg_path'] = '/ext3/conda/bootcamp/bin/ffmpeg'

    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)
  
    
    # Tuned result 1: 100 samples, 30 epochs
    # {'unet_num_layer': 4, 'unet_start_filts': 128, 'tmsa_window_shift_size': '24441222', 'tmsa_depth': 4, 'noise_std': 0.15, 'window_size': 8, \
    # 'scaler_type': 'standard', 'feature_size': 288, 'num_enc_layers': 3, 'num_dec_layers': 3, 'num_heads': 8, 'd_ff': 1024, 'dropout': 0.1, \
    # 'lr': 1e-05, 'lr_decay': 0.9, 'loss_type': 'huber', 'delta': 1.5, 'reg_var': 0.05}

    # Tuned result 2: random try 0.0021
    # {'unet_num_layer': 3, 'unet_start_filts': 128, 'tmsa_window_shift_size': '24441222', 'tmsa_depth': 4, \
    # 'noise_std': 0.01, 'window_size': 7, 'scaler_type': 'quantile', 'feature_size': 576, 'num_enc_layers': 2, \
    # 'num_dec_layers': 3, 'num_heads': 4, 'd_ff': 512, 'dropout': 0.2, 'lr': 5e-06, 'lr_decay': 0.9, \
    # 'loss_type': 'smooth_l1', 'delta': 2.5, 'reg_var': 0.05, 'predict_res': True}

    # Tuned result 3: random try 0.0027
    # {'unet_num_layer': 2, 'unet_start_filts': 128, 'tmsa_window_shift_size': '24441000', 'tmsa_depth': 6, \
    # 'noise_std': 0.1, 'window_size': 6, 'scaler_type': 'quantile', 'feature_size': 720, 'num_enc_layers': 2, \
    # 'num_dec_layers': 3, 'num_heads': 8, 'd_ff': 512, 'dropout': 0.2, 'lr': 1e-05, 'lr_decay': 0.9, \
    # 'loss_type': 'huber', 'delta': 2.5, 'reg_var': 0.05, 'predict_res': True}
    # GOOD BUT rollout tend to shrink down 

    {'unet_num_layer': 4, 'unet_start_filts': 32, 'tmsa_window_shift_size': '24441000', 'tmsa_depth': 4, \
    'noise_std': 0.15, 'window_size': 5, 'scaler_type': 'quantile', 'feature_size': 432, 'num_enc_layers': 1, \
    'num_dec_layers': 4, 'num_heads': 8, 'd_ff': 512, 'dropout': 0.2, 'lr': 1e-05, 'lr_decay': 0.8, \
    'loss_type': 'smooth_l1', 'delta': 1.5, 'reg_var': 0.05, 'predict_res': True}


    #512 |     2.5 |       0.1 |            720 | huber       | 1e-05 |        0.8 |        0.15 |                4 |                3 |           4 | True          |     0.05  | quantile      |            4 |                 24441000 |                4 |                 32 |             7 |      1 |         183.059  |  0.00418879  |  0.0291284 |  0.278215 
    #Regular CONV
    #conv_config = {'num_layer':5, 'kernel_size':3, 'padding':1, 'padding_mode':'replicate', 'conv_type': 'regular'}
    #UNet CONV

    {'unet_num_layer': 3, 'unet_start_filts': 128, 'tmsa_window_shift_size': '24441000', 'tmsa_depth': 6, 'noise_std': 0.15, 'window_size': 4, \
    'scaler_type': 'standard', 'feature_size': 288, 'num_enc_layers': 1, 'num_dec_layers': 3, 'num_heads': 4, 'd_ff': 1024, 'dropout': 0.1, \
    'lr': 0.001, 'lr_decay': 0.8, 'loss_type': 'smooth_l1', 'delta': 0.2, 'reg_var': 0.001, 'temporal_insert_layer': -1, 'decoder_only': False, 'tmsa_with_conv': False}

    {'unet_num_layer': 3, 'unet_start_filts': 128, 'tmsa_window_shift_size': '24441222', 'tmsa_depth': 8, 'noise_std': 0.01, 'window_size': 4, \
    'scaler_type': 'standard', 'feature_size': 576, 'num_enc_layers': 2, 'num_dec_layers': 5, 'num_heads': 4, 'd_ff': 1024, 'dropout': 0.2, \
    'lr': 0.0001, 'lr_decay': 0.9, 'loss_type': 'smooth_l1', 'delta': 1.0, 'reg_var': 0.05, 'temporal_insert_layer': -2, 'decoder_only': True, 'tmsa_with_conv': True}
    
    {'unet_num_layer': 4, 'unet_start_filts': 128, 'tmsa_window_shift_size': '24441000', 'tmsa_depth': 8, 'noise_std': 0.05, 'window_size': 5, \
    'scaler_type': 'standard', 'feature_size': 288, 'num_enc_layers': 2, 'num_dec_layers': 4, 'num_heads': 8, 'd_ff': 1024, 'dropout': 0.1, \
    'lr': 0.001, 'lr_decay': 0.8, 'loss_type': 'huber', 'delta': 0.1, 'reg_var': 0.01, 'temporal_insert_layer': -2, 'decoder_only': True, 'tmsa_with_conv': False}

    {'unet_num_layer': 2, 'unet_start_filts': 64, 'tmsa_window_shift_size': '22221000', 'tmsa_depth': 8, 'noise_std': 0.15, 'window_size': 3, \
    'scaler_type': 'standard', 'feature_size': 1152, 'num_enc_layers': 2, 'num_dec_layers': 4, 'num_heads': 4, 'd_ff': 1024, 'dropout': 0.2, \
    'lr': 0.0001, 'lr_decay': 0.8, 'loss_type': 'huber', 'delta': 1.5, 'reg_var': 0.05, 'temporal_insert_layer': -1, 'decoder_only': False, 'tmsa_with_conv': False} #0.49

    {'unet_num_layer': 4, 'unet_start_filts': 32, 'tmsa_window_shift_size': '22221000', 'tmsa_depth': 6, 'noise_std': 0.01, 'window_size': 4, \
    'scaler_type': 'standard', 'feature_size': 864, 'num_enc_layers': 2, 'num_dec_layers': 4, 'num_heads': 4, 'd_ff': 512, 'dropout': 0.2, \
    'lr': 0.0001, 'lr_decay': 0.8, 'loss_type': 'smooth_l1', 'delta': 0.1, 'reg_var': 0.001, 'temporal_insert_layer': -1, 'decoder_only': False, 'tmsa_with_conv': False} #0.44

    {'unet_num_layer': 2, 'unet_start_filts': 32, 'tmsa_window_shift_size': '24441000', 'tmsa_depth': 8, 'noise_std': 0.01, 'window_size': 4,\
     'scaler_type': 'standard', 'feature_size': 576, 'num_enc_layers': 2, 'num_dec_layers': 5, 'num_heads': 8, 'd_ff': 1024, 'dropout': 0.2, \
     'lr': 0.0001, 'lr_decay': 0.9, 'loss_type': 'smooth_l1', 'delta': 0.1, 'reg_var': 0.01, 'temporal_insert_layer': -1, 'decoder_only': False, 'tmsa_with_conv': False} #good on val but not on test

    {'unet_num_layer': 3, 'unet_start_filts': 64, 'tmsa_window_shift_size': '24441000', 'tmsa_depth': 8, 'noise_std': 0.01, 'window_size': 4,\
      'scaler_type': 'standard', 'feature_size': 576, 'num_enc_layers': 1, 'num_dec_layers': 5, 'num_heads': 8, 'd_ff': 512, 'dropout': 0.1,\
       'lr': 0.001, 'lr_decay': 0.9, 'loss_type': 'l1', 'delta': 0.1, 'reg_var': 0.01, 'temporal_insert_layer': -1, 'decoder_only': True, 'tmsa_with_conv': True} #no encoder

    {'unet_num_layer': 3, 'unet_start_filts': 32, 'tmsa_window_shift_size': '24441222', 'tmsa_depth': 2, 'noise_std': 0.0, 'window_size': 6,\
     'scaler_type': 'standard', 'feature_size': 1440, 'num_enc_layers': 1, 'num_dec_layers': 4, 'num_heads': 8, 'd_ff': 1024, 'dropout': 0.2,\
     'lr': 0.0001, 'lr_decay': 0.9, 'loss_type': 'huber', 'delta': 0.7, 'reg_var': 0.05, 'temporal_insert_layer': -2, 'decoder_only': True, 'tmsa_with_conv': False}

    {'unet_num_layer': 2, 'unet_start_filts': 16, 'tmsa_window_shift_size': '22221000', 'tmsa_depth': 8, 'noise_std': 0.1, 'window_size': 6, \
    'scaler_type': 'standard', 'feature_size': 576, 'num_enc_layers': 2, 'num_dec_layers': 3, 'num_heads': 8, 'd_ff': 1024, 'dropout': 0.2, \
    'lr': 0.0001, 'lr_decay': 0.9, 'loss_type': 'huber', 'delta': 0.3, 'reg_var': 0.01, 'temporal_insert_layer': -1, 'decoder_only': False, 'pos_insert': 'tmsa', 'encoder_decoder_type': 'conv'}

    #Equally good hyperparameters 0.019/0.5
    #| train_9dc19_00104 | RUNNING    | 10.32.35.128:2228471 |    512 | False          |    0.2  |       0.1 | conv                   |            864 | smooth_l1   | 0.0001 |        0.9 |        0    |                3 |                3 |           4 | transformer  |     0     | standard      |                      -1 |            6 |                 22221000 |                2 |                 16 |             3 |      7 |         3937.14  |     0.564175 |  0.0199103 |    0.506311   |
    {'unet_num_layer': 2, 'unet_start_filts': 16, 'tmsa_window_shift_size': '22221000', 'tmsa_depth': 4, 'noise_std': 0.0, 'window_size': 6,\
     'scaler_type': 'standard', 'feature_size': 576, 'num_enc_layers': 3, 'num_dec_layers': 3, 'num_heads': 8, 'd_ff': 512, 'dropout': 0.2, \
     'lr': 0.0001, 'lr_decay': 0.9, 'loss_type': 'l1', 'delta': 0.2, 'reg_var': 0.0, 'temporal_insert_layer': -2, 'decoder_only': False, 'pos_insert': 'tmsa', 'encoder_decoder_type': 'conv'}

    ###BEST NOW
    # conv_config = {'num_layer':4, 'start_filts':32, 'conv_type': 'UNet'}
    # tmsa_config = {'tmsa_with_conv': False, 'use_tmsa':True, 'use_tgt_tmsa':True, 'window_patch_size': (2,2,2,2), 'shift_size': (1,0,0,0), 'depth': 6, 'num_heads':4}
    # data_config = {'scale': False, 'noise_std':0.01, 'window_size': 4, 'option': 'patch', 'predict_res': False, 'scaler_type':'standard','patch_size': (4,4,4),}
    # best_config = {'epochs':30, 'pe_type': '3d_temporal', 'batch_size': 1, 'feature_size': 288*4, 'num_enc_layers': 2\
    #             , 'num_dec_layers': 4, 'temporal_insert_layer' : 3, 'num_head': 4, 'd_ff': 512, 'dropout': 0.2, 'lr': 1e-4, 'lr_decay': 0.8, 'loss_type':'smooth_l1', 'delta': 0.1\
    #             , 'mask_type':'patch','decoder_only':False, 'reg_var':0.001}

    conv_config = {'num_layer':2, 'start_filts':16, 'conv_type': 'UNet'}
    tmsa_config = {'pos_insert': 'tmsa', 'use_tmsa':True, 'use_tgt_tmsa':True, 'window_patch_size': (2,2,2,2), 'shift_size': (1,0,0,0), 'depth': 4, 'num_heads':8}
    data_config = {'scale': False, 'noise_std':0.0, 'window_size': 6, 'option': 'patch', 'predict_res': False, 'scaler_type':'standard','patch_size': (4,4,4),}
    best_config = {'epochs':30, 'pe_type': '3d_temporal', 'batch_size': 1, 'feature_size': 288*2, 'num_enc_layers': 3\
                , 'num_dec_layers': 3, 'temporal_insert_layer' : 2, 'num_head': 8, 'd_ff': 512, 'dropout': 0.2, 'lr': 1e-4, 'lr_decay': 0.9, 'loss_type':'l1', 'delta': 0.2\
                , 'mask_type':'patch','decoder_only':False, 'reg_var':0.0}
    predict_res = True
    simulation_seed = '_seed8'
    pe_type = best_config['pe_type']
    batch_size = best_config['batch_size']
    feature_size = best_config['feature_size']
    num_enc_layers = best_config['num_enc_layers']
    num_dec_layers = best_config['num_dec_layers']
    temporal_insert_layer = best_config['temporal_insert_layer']
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
    # predict_res = data_config['predict_res']
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
    train_loader, test_loader, scaler, data = get_data_loaders(train_proportion, test_proportion, val_proportion,\
        pred_size = pred_size, batch_size = batch_size, num_workers = 1, pin_memory = False, use_coords = True, use_time = True,\
        test_mode = True, scale = scale, window_size = window_size, patch_size = patch_size, option = option, predict_res = False,\
        noise_std = noise_std, scaler_type = scaler_type, seed = simulation_seed)
    
    model = Transformer(data, feature_size=feature_size,num_enc_layers=num_enc_layers,num_dec_layers = num_dec_layers,\
        d_ff = d_ff, dropout=dropout,num_head=num_head,pe_type=pe_type,grid_size=(grid_size,)*3,mask_type=mask_type,\
        patch_size=patch_size,window_size=window_size,pred_size=pred_size,decoder_only=decoder_only, tmsa_config = tmsa_config, conv_config = conv_config, load_prev_acrc = True)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
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
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    #writer = tensorboard.SummaryWriter('/scratch/yd1008/tensorboard_output/')

    epochs = best_config['epochs']
    train_losses = []
    test_losses = []
    tolerance = 10
    best_test_loss = float('inf')
    Early_Stopping = early_stopping(patience=tolerance)
    counter_old = 0
    x1, x2, x3 = patch_size
    g_cpu = torch.Generator()
    seed = 1008

    saved_model_name = 'best_model_0006_0807.pth'
    if os.path.exists(root_dir+f'{saved_model_name}') and skip_training:
        print('Loading model from: ', root_dir+f'{saved_model_name}')
        model.load_state_dict(torch.load(root_dir+f'{saved_model_name}'))
    else:
        for epoch in range(1, epochs + 1):    
            model.train() 
            total_loss = 0.
            start_time = time.time()

            for i, ((src, tgt), (src_coord, tgt_coord), (src_ts, tgt_ts)) in enumerate(train_loader):
                src, tgt, src_coord, tgt_coord, src_ts, tgt_ts = src.to(device), tgt.to(device), src_coord.to(device),\
                                                                            tgt_coord.to(device), src_ts.to(device), tgt_ts.to(device)
                for shift_size in get_roll_strides([0]):
                    shift_size = (0,) + shift_size
                    seed += 1
                    optimizer.zero_grad()

                    # for shift_size in get_roll_strides(2): ### Add shift size as model input if uncommenting this
                    #     shift_size = (0,) + shift_size
                        #dec_rollout = reduce(src.view(src.shape[0],window_size,patch_length,-1), 'b n p c -> b p c', 'mean')
                    dec_rollout = torch.zeros_like(src[:pred_size])
                    dec_in = torch.cat([src[pred_size:], dec_rollout], dim=0).float()

                    dec_in = dec_in + (torch.empty(tgt.shape).normal_(mean=0,std=noise_std,generator = g_cpu.manual_seed(seed))).to(device)
                    enc_in = src + (torch.empty(tgt.shape).normal_(mean=0,std=noise_std,generator = g_cpu.manual_seed(seed*100))).to(device)
                    output = model(enc_in, dec_in, src_coord, tgt_coord, src_ts, tgt_ts, shift_size, temporal_insert_layer = temporal_insert_layer)
                    if predict_res:
                        res = tgt-src
                        loss = criterion(output[-pred_size:,:,:,:,:], res[-pred_size:,:,:,:,:]) + reg_var*torch.abs((torch.var(res.detach(),(0,-1))-torch.var(output,(0,-1)))).mean()
                    else:
                        loss = criterion(output, tgt) + reg_var*torch.abs((torch.var(tgt.detach(),0)-torch.var(output,0))).mean()
                # loss = criterion(output, tgt)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

            avg_train_loss = total_loss*batch_size/len(train_loader)
            total_test_loss = evaluate(model, test_loader, criterion, patch_size=patch_size, predict_res = predict_res)
            avg_test_loss = total_test_loss/len(test_loader)

            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            

            if epoch==1: ###DEBUG
                print(f'Total of {len(train_loader)} samples in training set and {len(test_loader)} samples in test set', flush=True)

            if (epoch%4 == 0):
                print(f'Saving prediction for epoch {epoch}', flush=True)
                eval_start_time = time.time()
                predict_model(model, test_loader, epoch, config=best_config,\
                                    plot=True, plot_range=[0,0.5], final_prediction=False, predict_res = predict_res, file_prefix = 'test')  
                print(f'Epoch: {epoch}, train_loss: {avg_train_loss}, test_loss: {avg_test_loss}, lr: {scheduler.get_last_lr()}, training time: {time.time()-start_time} s, evaluation time: {time.time()-eval_start_time}', flush=True)
 
            else:
                print(f'Epoch: {epoch}, train_loss: {avg_train_loss}, test_loss: {avg_test_loss}, lr: {scheduler.get_last_lr()}, training time: {time.time()-start_time} s', flush=True)


            Early_Stopping(model, avg_test_loss)

            counter_new = Early_Stopping.counter
            if counter_new != counter_old:
                scheduler.step()  # update lr if early stop
                counter_old = counter_new
            if Early_Stopping.early_stop:
                break

        #save model
        if save_model:
            if os.path.exists(root_dir + f'/best_model{simulation_seed}.pth'):  # checking if there is a file with this name
                os.remove(root_dir + f'/best_model{simulation_seed}.pth')  # deleting the file
                torch.save(model.state_dict(), root_dir + f'/best_model{simulation_seed}.pth')
            else:
                torch.save(model.state_dict(), root_dir + f'/best_model{simulation_seed}.pth')
### Plot losses        
    xs = np.arange(len(train_losses))
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,train_losses)
    fig.savefig(root_dir + '/figs/loss' + f"/train_loss_pe{best_config['pe_type']}_batch{best_config['batch_size']}_window{data_config['window_size']}_patch{data_config['patch_size']}.png")
    plt.close(fig)
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,test_losses)
    fig.savefig(root_dir + '/figs/loss' + f"/test_loss_pe{best_config['pe_type']}_batch{best_config['batch_size']}_window{data_config['window_size']}_patch{data_config['patch_size']}.png")
    plt.close(fig)

### Predict
    def save_pred(dataloader, file_prefix, shift_size, pred_feature=0):
        print('-'*20 + f'Saving prediction results for {file_prefix}' + '-'*20, flush=True)
        start_time = time.time()
        final_result = predict_model(model, dataloader,  best_config['epochs'], config=best_config,\
                                                plot=True, plot_range=[0,1], final_prediction=True, predict_res = predict_res, file_prefix = file_prefix, shift_size = shift_size, pred_feature = pred_feature)
        prediction = final_result['prediction']
        print('-'*20 + ' Measure for Simulation Speed ' + '-'*20, flush=True)
        print(f'Time to forcast on {len(prediction)} samples: {time.time()-start_time} s', flush=True)
        ### Check MSE, MAE
        try:
            truth = final_result['truth'].values
            grid_prod = grid_size**3
            for i in range(len(truth)//grid_prod-1):
                MSE = mean_squared_error(truth[:i*grid_prod], prediction[:i*grid_prod])
                MAE = mean_absolute_error(truth[:i*grid_prod], prediction[:i*grid_prod])
                R2 = r2_score(truth[:i*grid_prod], prediction[:i*grid_prod])
                EXPLAINED_VARIANCE_SCORE = explained_variance_score(truth[:i*grid_prod], prediction[:i*grid_prod])
                print('-'*20 + ' Measure for Simulation Performance ' + '-'*20, flush=True)
                print(f' Rollout step: {1}: \n MSE: {MSE}, MAE: {MAE}, R2: {R2}, EXPLAINED_VARIANCE_SCORE: {EXPLAINED_VARIANCE_SCORE}', flush=True)
        except:
            truth = final_result['truth']
            MSE = mean_squared_error(truth, prediction)
            MAE = mean_absolute_error(truth, prediction)
            R2 = r2_score(truth, prediction)
            EXPLAINED_VARIANCE_SCORE = explained_variance_score(truth, prediction)
            print('-'*20 + ' Measure for Simulation Performance ' + '-'*20, flush=True)
            print(f'MSE: {MSE}, MAE: {MAE}, R2: {R2}, EXPLAINED_VARIANCE_SCORE: {EXPLAINED_VARIANCE_SCORE}', flush=True)
        ### Save model result
        test_result_df = pd.DataFrame.from_dict(final_result)
        print('-'*20 + ' Dataframe for Final Result ' + '-'*20, flush=True)
        print(test_result_df, flush=True)
        test_result_df.to_csv(root_dir + f'/{file_prefix}_transformer_prediction_2.csv')
        ### slice plot
        img_dir = root_dir + '/slice_plot' 
        pred_df = pd.read_csv(root_dir + f'/{file_prefix}_transformer_prediction_2.csv',index_col=0)
        grid_size = [16,16,16]
        axis_colnames = ['x1','x2','x3']
        slice_axis_index = 0
        pred_colname = 'prediction'
        truth_colname = 'truth'
        time_colname = 'time'
        plot_forecast(pred_df=pred_df, grid_size=grid_size, axis_colnames=axis_colnames, slice_axis_index=2, \
                            pred_colname=pred_colname,truth_colname=truth_colname, time_colname=time_colname,  \
                            plot_anime = True, img_dir = img_dir, config=best_config, file_prefix=file_prefix) 
    # save_pred(test_loader, 'test_shift2_', [2])
    # save_pred(train_loader, 'train_shift2_',[2])
    save_pred(test_loader, f'{simulation_seed[1:]}_test', [1,2,3],0)
    save_pred(test_loader, f'{simulation_seed[1:]}_test_vel1', [1,2,3],1)
    save_pred(test_loader, f'{simulation_seed[1:]}_test_vel2', [1,2,3],2)
    save_pred(test_loader, f'{simulation_seed[1:]}_test_vel3', [1,2,3],3)
    save_pred(test_loader, f'{simulation_seed[1:]}_test_press', [1,2,3],4)
    #save_pred(train_loader, 'train',[1,2,3])
    # save_pred(train_loader, 'train', [1,2,3],0)
    # save_pred(train_loader, 'train_vel1', [1,2,3],1)
    # save_pred(train_loader, 'train_vel2', [1,2,3],2)
    # save_pred(train_loader, 'train_vel3', [1,2,3],3)
    # save_pred(train_loader, 'train_press', [1,2,3],4)
    