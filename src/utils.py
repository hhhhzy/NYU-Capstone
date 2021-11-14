#!/bin/bash python
import os
import numpy as np
import torch 
import torch.nn as nn
from athena_read import *
from sklearn.preprocessing import StandardScaler

def get_rho(data_path):
    lst = sorted(os.listdir(data_path))[4:]
    rho = []
    coords = []
    timestamps = []
    for name in lst:
        path = data_path+'/'+name
        d = athdf(path)
        nx1 = len(d['x1v'])
        nx2 = len(d['x2v'])
        nx3 = len(d['x3v'])
        coord = np.transpose(np.array(np.meshgrid(np.arange(nx1),np.arange(nx2),np.arange(nx3))), axes=[2,1,3,0]).reshape(-1,3)
        rho.append(d['rho'])
        timestamp_repeated = [d['Time']]*(np.prod(d['rho'].shape))
        timestamps.extend(timestamp_repeated) ### Add nx1*nx2*nx3 time values into the output list
        #print('rho:',d['rho'].shape, 'coord.shape:',coord.shape,'len timestamp:',len(timestamp_repeated))
        coords.extend(coord)

    rho = np.array(rho)
    meshed_blocks = (nx1, nx2, nx3)
    timestamps = np.array(timestamps)
    rho_reshaped = rho.flatten()
    coords = np.array(coords)
    #print(f'rho shape: {rho_reshaped.shape}, coords shape: {coords.shape}')
    return rho_reshaped, meshed_blocks, coords, timestamps

def to_windowed(data, meshed_blocks, window_size, pred_size, option='patch', patch_size=2):
    """
    note: window_size represents number of timestamps when option='time' or 'patch', while represents number of blocks when option='space' 
    """
    out = []
    if option == 'space':
        length = len(data) - window_size
        for i in range(length):
            feature = np.array(data[i:i+(window_size)])
            target = np.array(data[i+pred_size:i+window_size+pred_size])
            out.append((feature, target))
    
    elif option == 'time':
        nx1, nx2, nx3 = meshed_blocks
        length = len(data)-nx1*nx2*nx3*window_size
        for i in range(length):
            feature  = np.array(data[[i+nx1*nx2*nx3*k for k in range(window_size)]])
            target = np.array(data[[i+nx1*nx2*nx3*(k+pred_size) for k in range(window_size)]])        
            out.append((feature,target))

    elif option == 'patch':
        nx1, nx2, nx3 = meshed_blocks
        length = int((len(data)-nx1*nx2*nx3*window_size)/(patch_size**3))
        for i in range(length):
            feature  = np.array(data[[i + time*nx1*nx2*nx3 + j*(nx2*nx3) + k*(nx3) + l \
                                for time in range(window_size) for j in range(patch_size) for k in range(patch_size) for l in range(patch_size)]])
            target  = np.array(data[[i + (time+pred_size)*nx1*nx2*nx3 + j*(nx2*nx3) + k*(nx3) + l \
                                for time in range(window_size) for j in range(patch_size) for k in range(patch_size) for l in range(patch_size)]])      
            out.append((feature,target))
            
    return np.array(out)


def train_test_val_split(data, meshed_blocks, train_proportion = 0.6, val_proportion = 0.2, test_proportion = 0.2\
              , window_size = 12, pred_size = 1, scale = False, option='patch', patch_size=2):

    scaler = StandardScaler()
    if scale == True:
        data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
    
    x_vals = to_windowed(data, meshed_blocks, window_size, pred_size, option=option, patch_size=patch_size)

    total_len = len(x_vals)
    train_len = int(total_len*train_proportion)
    val_len = int(total_len*val_proportion)
    
    train = x_vals[0:train_len]
    val = x_vals[train_len:(train_len+val_len)]
    test = x_vals[(train_len+val_len):]

    train_data = torch.from_numpy(train).float()
    val_data = torch.from_numpy(val).float()
    test_data = torch.from_numpy(test).float()

    return train_data,val_data,test_data,scaler


### Adjust __init__ to fit the inputs
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,x,coords,timestamp):
        self.x=x
        self.coords=coords
        self.timestamp=timestamp
 
    def __len__(self):
        return len(self.x)
 
    def __getitem__(self,idx):
        return((self.x[idx][0].view(-1,1), self.x[idx][1].view(-1,1)),(self.coords[idx][0], self.coords[idx][1]),(self.timestamp[idx][0], self.timestamp[idx][1]))
    
# class CustomFeatureDataset(torch.utils.data.Dataset):
#     def __init__(self,x):
#         self.x=x
 
#     def __len__(self):
#         return self.x.shape[0]
 
#     def __getitem__(self,idx):
#         return(self.x[idx][0], self.x[idx][1])

def get_data_loaders(train_proportion = 0.5, test_proportion = 0.25, val_proportion = 0.25,window_size = 10, \
                        pred_size =1, batch_size = 16, num_workers = 1, pin_memory = True, \
                        use_coords = True, use_time = True, test_mode = False, input_type='patch', patch_size=2): 

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(505)
    
    data_path='/scratch/zh2095/data_turb_dedt1_16'
    data, meshed_blocks, coords, timestamps = get_rho(data_path)

    ###FOR SIMPLE TEST SINE AND COSINE 
    #long_range_stationary_x_vals = (np.sin(2*np.pi*time_vec/period))+(np.cos(3*np.pi*time_vec/period)) + 0.25*np.random.randn(time_vec.size)
    ###FOR ARFIMA TEST
    #data = arfima([0.5,0.4],0.3,[0.2,0.1],10000,warmup=2^10)

    if use_coords:
        print('-'*20,'split for coords')
        train_coords,val_coords,test_coords, scaler = train_test_val_split(\
            coords, meshed_blocks = meshed_blocks, train_proportion = train_proportion\
            , val_proportion = val_proportion, test_proportion = test_proportion\
            , window_size = window_size, pred_size = pred_size, scale = False, option = input_type, patch_size=patch_size)
        print(f'train_coords: {train_coords.shape}')

    if use_time:
        print('-'*20,'split for timestamp')
        train_timestamps,val_timestamps,test_timestamps, scaler = train_test_val_split(\
            timestamps, meshed_blocks = meshed_blocks, train_proportion = train_proportion\
            , val_proportion = val_proportion, test_proportion = test_proportion\
            , window_size = window_size, pred_size = pred_size, scale = False, option = input_type, patch_size=patch_size)
        print(f'train_timestamps: {train_timestamps.shape}')

    print('-'*20,'split for data')
    train_data,val_data,test_data, scaler = train_test_val_split(\
        data, meshed_blocks = meshed_blocks, train_proportion = train_proportion\
        , val_proportion = val_proportion, test_proportion = test_proportion\
        , window_size = window_size, pred_size = pred_size, scale = False, option = input_type, patch_size=patch_size)
    print(f'train_data: {train_data.shape}')
    
    if test_mode:
        train_val_data = torch.cat((train_data,val_data),0)
        train_val_coords = torch.cat((train_coords,val_coords),0)
        train_val_timestamps = torch.cat((train_timestamps,val_timestamps),0)

        dataset_train_val, dataset_test = CustomDataset(train_val_data,train_val_coords,train_val_timestamps), CustomDataset(test_data,test_coords,test_timestamps)
        train_val_loader = torch.utils.data.DataLoader(dataset_train_val, batch_size=batch_size, \
                                        drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                        persistent_workers=True, prefetch_factor = 16)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, \
                                        drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                        persistent_workers=True, prefetch_factor = 128) 

        # if use_coords and use_time:
        #     ### create data loaders for coordinates, returning the corresponding xyz positions of size [batch, 2, seq_len, 3] of a given sqeunce of data of size [batch, 2, seq_len], not functionable on multivariate time sereis
        #     train_val_coords = torch.cat((train_coords,val_coords),0)
        #     coords_train_val, coords_test = CustomFeatureDataset(train_val_coords), CustomFeatureDataset(test_coords)
        #     train_val_coords_loader = torch.utils.data.DataLoader(coords_train_val, batch_size=batch_size, 
        #                                     drop_last=False, 
        #                                     num_workers=num_workers, pin_memory=pin_memory)
        #     test_coords_loader = torch.utils.data.DataLoader(coords_test, batch_size=1, 
        #                                     drop_last=False, 
        #                                     num_workers=num_workers, pin_memory=pin_memory) 
        #     ### create data loaders for timestamps
        #     train_val_timestamps = torch.cat((train_timestamps,val_timestamps),0)
        #     timestamps_train_val, timestamps_test = CustomFeatureDataset(train_val_timestamps), CustomFeatureDataset(test_timestamps)
        #     train_val_timestamps_loader = torch.utils.data.DataLoader(timestamps_train_val, batch_size=batch_size, 
        #                                     drop_last=False, 
        #                                     num_workers=num_workers, pin_memory=pin_memory)
        #     test_timestamps_loader = torch.utils.data.DataLoader(timestamps_test, batch_size=1, 
        #                                     drop_last=False, 
        #                                     num_workers=num_workers, pin_memory=pin_memory) 
        #     return train_val_loader, test_loader, train_val_coords_loader, test_coords_loader, train_val_timestamps_loader,test_timestamps_loader scaler 
        return train_val_loader, test_loader, scaler
    
    if not test_mode:                           
        dataset_train, dataset_test, dataset_val = CustomDataset(train_data,train_coords,train_timestamps), CustomDataset(test_data,test_coords,test_timestamps), CustomDataset(val_data,val_coords,val_timestamps)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, \
                                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                            persistent_workers=True, prefetch_factor = 16)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, \
                                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                            persistent_workers=True, prefetch_factor = 16)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, \
                                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                            persistent_workers=True, prefetch_factor = 16)
        # if use_coords:
        #     coords_train, coords_test, coords_val = CustomFeatureDataset(train_data), CustomFeatureDataset(test_data), CustomFeatureDataset(val_data)
        #     train_coords_loader = torch.utils.data.DataLoader(coords_train, batch_size=batch_size, 
        #                                         drop_last=False, 
        #                                         num_workers=num_workers, pin_memory=pin_memory)
        #     test_coords_loader = torch.utils.data.DataLoader(coords_test, batch_size=batch_size, 
        #                                         drop_last=False, 
        #                                         num_workers=num_workers, pin_memory=pin_memory)
        #     val_coords_loader = torch.utils.data.DataLoader(coords_val, batch_size=batch_size, 
        #                                         drop_last=False, 
        #                                         num_workers=num_workers, pin_memory=pin_memory)
        #     return train_loader,val_loader, test_loader, train_coords_loader, test_coords_loader, val_coords_loader

        return train_loader,val_loader, test_loader
