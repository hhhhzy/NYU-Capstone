#!/bin/bash python
import os
import numpy as np
import torch 
import torch.nn as nn
from athena_read import *
import scipy.io


def get_output(local=True):

    if local:
        burger = scipy.io.loadmat(r'C:\Users\curti\Desktop\nyu-capstone\notebooks\burger\burgers_v100_t100_r1024_N2048.mat')
    else:
        burger = scipy.io.loadmat('/scratch/tl2204/burgers_v100_t100_r1024_N2048.mat')
    output = burger['output']
    
    output = output[1,:,::8]
    
    
    return output.flatten()
    

# def to_windowed(data,window_size,pred_size):
    
#     out = []
#     for i in range(len(data) - window_size):
#             feature = np.array(data[i:i+(window_size)])
#             target = np.array(data[i+pred_size:i+window_size+pred_size])
#             out.append((feature, target))

#     return np.array(out)
def to_windowed(data,window_size,pred_size):
    
    n = 128
    out = []
    for i in range(len(data)-n*window_size):
        feature  = np.array(data[[i+n*k for k in range(window_size)]])
        target = np.array(data[[i+n*(k+pred_size) for k in range(window_size)]])        
        out.append((feature,target))

    return np.array(out)

def train_test_val_split(x_vals, train_proportion = 0.6, val_proportion = 0.2, test_proportion = 0.2\
              , window_size = 12, pred_size = 1):

    total_len = len(x_vals)
    train_len = int(total_len*train_proportion)
    val_len = int(total_len*val_proportion)
    test_len = int(total_len*test_proportion)


    train_data = x_vals[0:train_len]
    val_data = x_vals[train_len:(train_len+val_len)]
    test_data = x_vals[(train_len+val_len):]

    train = to_windowed(train_data,window_size,pred_size)
    val = to_windowed(val_data,window_size,pred_size)
    test = to_windowed(test_data,window_size,pred_size)

    train = torch.from_numpy(train).float()
    val = torch.from_numpy(val).float()
    test = torch.from_numpy(test).float()

    return train,val,test,train_data,val_data,test_data

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,x):
        self.x=x
 
    def __len__(self):
        return len(self.x)
 
    def __getitem__(self,idx):
        return(self.x[idx][0].view(-1,1), self.x[idx][1].view(-1,1))
    
def get_data_loaders(train_proportion = 0.5, test_proportion = 0.25, val_proportion = 0.25,window_size = 10, \
    pred_size =1, batch_size = 16, num_workers = 1, pin_memory = True, test_mode = False): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(505)
    
    # data_path='C:/Users/52673/Desktop/NYU MSDS/3-DS-1006 CAPSTONE/data_turb_dedt1_16' # if locally
    data = get_output()
    
    train_data,val_data,test_data,train_original,val_original,test_original = train_test_val_split(\
        data ,train_proportion = train_proportion\
        , val_proportion = val_proportion, test_proportion = test_proportion\
        , window_size = window_size, pred_size = pred_size)
    if test_mode:
        train_val_data = torch.cat((train_data,val_data),0)
        dataset_train_val, dataset_test = CustomDataset(train_val_data), CustomDataset(test_data)
        train_val_loader = torch.utils.data.DataLoader(dataset_train_val, batch_size=batch_size, 
                                        drop_last=False, 
                                        num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, 
                                        drop_last=False, 
                                        num_workers=num_workers, pin_memory=pin_memory) 
        return train_val_loader, test_loader
    if not test_mode:                           
        dataset_train, dataset_test, dataset_val = CustomDataset(train_data), CustomDataset(test_data), CustomDataset(val_data)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                            drop_last=False, 
                                            num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                            drop_last=False, 
                                            num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, 
                                            drop_last=False, 
                                            num_workers=num_workers, pin_memory=pin_memory)
                        
        return train_loader,val_loader, test_loader
