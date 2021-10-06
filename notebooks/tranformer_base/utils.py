#!/bin/bash python
import numpy as np
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from _arfima import arfima
from mlp import MLP
from utils import *
def to_windowed(data,window_size,pred_size):
    out = []
    for i in range(len(data)-window_size):
        feature = np.array(data[i:i+(window_size)])
        target = np.array(data[i+pred_size:i+window_size+pred_size])
        out.append((feature,target))
        

    return np.array(out)#, np.array(targets)

def train_test_val_split(x_vals ,train_proportion = 0.6, val_proportion = 0.2, test_proportion = 0.2\
              , window_size = 12, pred_size = 1):

    total_len = len(x_vals)
    train_len = int(total_len*train_proportion)
    val_len = int(total_len*val_proportion)
    test_len = int(total_len*test_proportion)
    ### Add a scaler here on x_vals
    scaler = StandardScaler()
    x_vals = scaler.fit_transform(x_vals.reshape(-1, 1)).reshape(-1)

    train_data = x_vals[0:train_len]
    val_data = x_vals[train_len:(train_len+val_len)]
    test_data = x_vals[(train_len+val_len):]

    train = to_windowed(train_data,window_size,pred_size)
    val = to_windowed(val_data,window_size,pred_size)
    test = to_windowed(test_data,window_size,pred_size)

    train = torch.from_numpy(train).float()
    val = torch.from_numpy(val).float()
    test = torch.from_numpy(test).float()

    return train,val,test,train_data,val_data,test_data, scaler

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,x):
        self.x=x
 
    def __len__(self):
        return len(self.x)
 
    def __getitem__(self,idx):
        return(self.x[idx][0].view(-1,1),self.x[idx][1].view(-1,1))
    
def get_data_loaders(train_proportion = 0.6, test_proportion = 0.2, val_proportion = 0.2,window_size = 10, pred_size =1, batch_size = 10, num_workers = 1, pin_memory = True): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(505)
    long_range_stationary_x_vals = arfima([0.5,0.4],0.3,[0.2,0.1],10000,warmup=2^10)
    train_data,val_data,test_data,train_original,val_original,test_original, scaler = train_test_val_split(\
        long_range_stationary_x_vals ,train_proportion = train_proportion\
        , val_proportion = val_proportion, test_proportion = test_proportion\
        , window_size = window_size, pred_size = pred_size)
    
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
