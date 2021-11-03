#!/bin/bash python
import os
import numpy as np
import torch 
import torch.nn as nn
from _arfima import arfima
from utils import *
from athena_read import *
from sklearn.preprocessing import StandardScaler

def get_rho(data_path):
    lst = sorted(os.listdir(data_path))[4:]
    rho = []
    coords = []
    for name in lst:
        path = data_path+'/'+name
        d = athdf(path)
        x = np.arange(len(d['x1v']))
        y = np.arange(len(d['x2v']))
        z = np.arange(len(d['x3v']))
        coord = np.array(np.meshgrid(x,y,z)).T.reshape(-1,3)
        rho.append(d['rho'])
        #print('rho:',d['rho'].shape, 'coord.shape:',coord.shape)
        coords.extend(coord)

    rho = np.array(rho)

    ntime, nx1, nx2, nx3 = np.shape(rho)
    rho_reshaped = rho.flatten()
    coords = np.array(coords)
    print(f'rho shape: {rho_reshaped.shape}, coords shape: {coords.shape}')
    return rho_reshaped, coords

def to_windowed(data,window_size,pred_size):
    
    out = []
    for i in range(len(data) - window_size):
            feature = np.array(data[i:i+(window_size)])
            target = np.array(data[i+pred_size:i+window_size+pred_size])
            out.append((feature, target))

    return np.array(out)

def train_test_val_split(x_vals , train_proportion = 0.6, val_proportion = 0.2, test_proportion = 0.2\
              , window_size = 12, pred_size = 1, scale = True):

    if scale:
        scaler = StandardScaler()
        x_vals = scaler.fit_transform(x_vals.reshape(-1, 1)).reshape(-1)
    
    x_vals = to_windowed(x_vals, window_size, pred_size)

    total_len = len(x_vals)
    train_len = int(total_len*train_proportion)
    val_len = int(total_len*val_proportion)
    

    train_data = x_vals[0:train_len]
    val_data = x_vals[train_len:(train_len+val_len)]
    test_data = x_vals[(train_len+val_len):]

    print('train_data',train_data.shape)

    train = torch.from_numpy(train_data).float()
    val = torch.from_numpy(val_data).float()
    test = torch.from_numpy(test_data).float()
    if scale:
        return train,val,test,train_data,val_data,test_data,scaler
    return train,val,test,train_data,val_data,test_data

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,x):
        self.x=x
 
    def __len__(self):
        return len(self.x)
 
    def __getitem__(self,idx):
        return(self.x[idx][0].view(-1,1), self.x[idx][1].view(-1,1))
    
class CustomCoordsDataset(torch.utils.data.Dataset):
    def __init__(self,x):
        self.x=x
 
    def __len__(self):
        return self.x.shape[0]
 
    def __getitem__(self,idx):
        return(self.x[idx][0], self.x[idx][1])

def get_data_loaders(train_proportion = 0.5, test_proportion = 0.25, val_proportion = 0.25,window_size = 10, \
    pred_size =1, batch_size = 16, num_workers = 1, pin_memory = True, use_coords = False, test_mode = False): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(505)
    
    # data_path='/scratch/yd1008/nyu-capstone/data_turb_dedt1_16'
    # data, coords = get_rho(data_path)
    #long_range_stationary_x_vals = (np.sin(2*np.pi*time_vec/period))+(np.cos(3*np.pi*time_vec/period)) + 0.25*np.random.randn(time_vec.size)
    data = arfima([0.5,0.4],0.3,[0.2,0.1],10000,warmup=2^10)
    if use_coords:
        print('-'*20,'split for coords')
        train_coords,val_coords,test_coords,train_coords_original,val_coords_original,coords_test_original = train_test_val_split(\
            coords ,train_proportion = train_proportion\
            , val_proportion = val_proportion, test_proportion = test_proportion\
            , window_size = window_size, pred_size = pred_size, scale=False)
        print(f'train_coords: {train_coords.shape}')
    print('-'*20,'split for data')
    train_data,val_data,test_data,train_original,val_original,test_original,scaler = train_test_val_split(\
        data ,train_proportion = train_proportion\
        , val_proportion = val_proportion, test_proportion = test_proportion\
        , window_size = window_size, pred_size = pred_size)
    print(f'train_data: {train_data.shape}')
    if test_mode:
        train_val_data = torch.cat((train_data,val_data),0)
        dataset_train_val, dataset_test = CustomDataset(train_val_data), CustomDataset(test_data)
        train_val_loader = torch.utils.data.DataLoader(dataset_train_val, batch_size=batch_size, 
                                        drop_last=False, 
                                        num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, 
                                        drop_last=False, 
                                        num_workers=num_workers, pin_memory=pin_memory) 
        if use_coords:
            ### create data loaders for coordinates, returning the corresponding xyz positions of size [batch, 2, seq_len, 3] of a given sqeunce of data of size [batch, 2, seq_len], not functionable on multivariate time sereis
            train_val_coords = torch.cat((train_coords,val_coords),0)
            coords_train_val, coords_test = CustomCoordsDataset(train_val_coords), CustomCoordsDataset(test_coords)
            train_val_coords_loader = torch.utils.data.DataLoader(coords_train_val, batch_size=batch_size, 
                                            drop_last=False, 
                                            num_workers=num_workers, pin_memory=pin_memory)
            test_coords_loader = torch.utils.data.DataLoader(coords_test, batch_size=1, 
                                            drop_last=False, 
                                            num_workers=num_workers, pin_memory=pin_memory) 
            return train_val_loader, test_loader, train_val_coords_loader, test_coords_loader, scaler
        return train_val_loader, test_loader, scaler
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
        if use_coords:
            coords_train, coords_test, coords_val = CustomCoordsDataset(train_data), CustomCoordsDataset(test_data), CustomCoordsDataset(val_data)
            train_coords_loader = torch.utils.data.DataLoader(coords_train, batch_size=batch_size, 
                                                drop_last=False, 
                                                num_workers=num_workers, pin_memory=pin_memory)
            test_coords_loader = torch.utils.data.DataLoader(coords_test, batch_size=batch_size, 
                                                drop_last=False, 
                                                num_workers=num_workers, pin_memory=pin_memory)
            val_coords_loader = torch.utils.data.DataLoader(coords_val, batch_size=batch_size, 
                                                drop_last=False, 
                                                num_workers=num_workers, pin_memory=pin_memory)
            return train_loader,val_loader, test_loader, train_coords_loader, test_coords_loader, val_coords_loader

        return train_loader,val_loader, test_loader, train_coords_loader, test_coords_loader, val_coords_loader
