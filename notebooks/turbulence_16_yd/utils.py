#!/bin/bash python
import os
import numpy as np
import pandas as pd
import math
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from _arfima import arfima
from athena_read import *
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

### VARIABLES(KEYS) IN DATASET
# dict_keys(['Coordinates', 'DatasetNames', 'MaxLevel', 'MeshBlockSize', 'NumCycles', \
# 'NumMeshBlocks', 'NumVariables', 'RootGridSize', 'RootGridX1', 'RootGridX2', 'RootGridX3', \
# 'Time', 'VariableNames', 'x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', 'rho', 'press', 'vel1', 'vel2', 'vel3'])
def get_rho(data_path, predict_res = False, noise_std = 0.01, var_name = 'rho'):
    np.random.seed(1008)
    lst = sorted(os.listdir(data_path))[4:-1]
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
        rho.append(d[var_name])
        timestamp_repeated = [d['Time']]*(np.prod(d[var_name].shape))
        timestamps.extend(timestamp_repeated) 
        coords.extend(coord)
        #print(f"Name: {name}, keys: {d.keys()}", flush=True)#x1v: {d['x1v']}, x2v: {d['x2v']}, x3v:{d['x3v']}

    rho = np.array(rho)
    meshed_blocks = (nx1, nx2, nx3)
    timestamps = np.array(timestamps)
    coords = np.array(coords)
    rho_original = rho.flatten()
    #rho_original = rho_original + np.random.normal(0,noise_std,len(rho_original))

    ### TEST: predict all residuals instead of rhos
    if predict_res:
        rho_residual = (np.roll(rho_original,-nx1*nx2*nx3)-rho_original)[:-nx1*nx2*nx3] ### Residual
        timestamps = timestamps[nx1*nx2*nx3:]
        coords = coords[nx1*nx2*nx3:]
        #rho_original = rho_original[:-nx1*nx2*nx3] ### to reconstruct truth from residuals, take the first meshblock of rho_original and add the residuals.
        return rho_original[nx1*nx2*nx3:], rho_residual, meshed_blocks, coords, timestamps
    else: 
        return rho_original, meshed_blocks, coords, timestamps
    # print(f'rho shape: {rho_original.shape}, coords shape: {coords.shape}')
    # return rho_original, rho_residual, meshed_blocks, coords, timestamps


# def to_windowed(data, meshed_blocks, pred_size, window_size , patch_size=(1,1,16), option='patch', patch_stride = (1,1,1)):
#     """
#     window_size: Here represents the number of time steps in each window
#     patch_size: (x1,x2,x3), determines the shape of the cuboid patch
#     patch_stride: (s1,s2,s3), how many number of values patches move around in each meshblock in x,y,z direction
#     """
#     out = []
#     if option == 'space':
#         length = len(data) - window_size
#         for i in range(length):
#             feature = np.array(data[i:i+(window_size)])
#             target = np.array(data[i+pred_size:i+window_size+pred_size])
#             out.append((feature, target))
#         patches_per_block = None 

#     elif option == 'time':
#         nx1, nx2, nx3 = meshed_blocks
#         length = len(data)-nx1*nx2*nx3*window_size
#         for i in range(length):
#             feature  = np.array(data[[i+nx1*nx2*nx3*k for k in range(window_size)]])
#             target = np.array(data[[i+nx1*nx2*nx3*(k+pred_size) for k in range(window_size)]])        
#             out.append((feature,target))
#         patches_per_block = None 

#     elif option == 'patch':
#         x1, x2, x3 = patch_size
#         nx1, nx2, nx3 = meshed_blocks

#         vertices = []
#         for t in range(int((len(data)-nx1*nx2*nx3*window_size)/(nx1*nx2*nx3))):
#             for i in range(nx1//x1):
#                 for j in range(nx2//x2):
#                     for k in range(nx3//x3):
#                         vertices.append(t*nx1*nx2*nx3 + i*x1*nx2*nx3 + j*x2*nx3 + k*x3)

#         for i in vertices:
#             feature  = np.array(data[[i + time*nx1*nx2*nx3 + j*(nx2*nx3) + k*(nx3) + l \
#                                 for time in range(window_size-pred_size+1) for j in range(x1) for k in range(x2) for l in range(x3)]])
#             target  = np.array(data[[i + (time+pred_size)*nx1*nx2*nx3 + j*(nx2*nx3) + k*(nx3) + l \
#                                 for time in range(window_size-pred_size+1) for j in range(x1) for k in range(x2) for l in range(x3)]])  
#             out.append((feature,target))
#         patches_per_block = np.prod([nx1//x1,nx2//x2,nx3//x3])
        
#     elif option == 'patch_overlap':
#         x1, x2, x3 = patch_size
#         nx1, nx2, nx3 = meshed_blocks
#         stride_x, stride_y, stride_z = patch_stride
#         vertices = []
#         for t in range(int((len(data)-nx1*nx2*nx3*window_size)/(nx1*nx2*nx3))):
#             for k in range(math.floor((nx3-x3)/stride_z+1)):
#                 for j in range(math.floor((nx2-x2)/stride_y+1)):
#                     for i in range(math.floor((nx1-x1)/stride_x+1)):
#                         vertices.append(t*nx1*nx2*nx3 + k*nx1*nx2 + j*nx1 + i)
#         for i in vertices:
#             feature  = np.array(data[[i + time*nx1*nx2*nx3 + l*(nx2*nx3) + k*(nx3) + j \
#                                 for time in range(window_size) for j in range(x1) for k in range(x2) for l in range(x3)]])
#             target  = np.array(data[[i + (time+pred_size)*nx1*nx2*nx3 + l*(nx2*nx3) + k*(nx3) + j \
#                                 for time in range(window_size) for j in range(x1) for k in range(x2) for l in range(x3)]])      
#             out.append((feature,target))
#         patches_per_block = np.prod([nx1-x1+1,nx2-x2+1,nx3-x3+1])
            
#     return np.array(out), patches_per_block

def to_windowed(x, block_size, window_size, feature_size):
    nx1, nx2, nx3 = block_size
    x = torch.Tensor(x) #make sure its a float tensor
    x = x.view(-1,nx1,nx2,nx3,feature_size)
    x_windowed = x.unfold(0,window_size,1).permute(0,5,1,2,3,4)
    return x_windowed

# def train_test_val_split(data, meshed_blocks, train_proportion = 0.6, val_proportion = 0.2, test_proportion = 0.2\
#               , pred_size = 1, scale = False, window_size = 10, patch_size=(1,1,16), option='patch', scaler_type='standard'):
    
#     if scale == True:
#         if scaler_type not in ['standard','power_box','power_yeo','robust']:
#             scaler_type = 'standard'
#         if scaler_type == 'standard':
#             scaler = StandardScaler()
#         elif scaler_type == 'power_box':
#             scaler = PowerTransformer(method='box-cox',standardize=True)
#         elif scaler_type == 'power_yeo':
#             scaler = PowerTransformer(method='yeo-johnson',standardize=True)
#         elif scaler_type == 'robust':
#             scaler = RobustScaler(with_centering=False,with_scaling=True,quantile_range=(25.0, 75.0),copy=True,unit_variance=False)
#         elif scaler_type == 'quantile':
#             scaler = QuantileTransformer(output_distribution='normal')
#         print(f'Using scaler: {scaler_type}')
#         data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
#     if option in ['patch','patch_overlap']: ### Force each set start on a new block
#         windows, patches_per_block = to_windowed(data, meshed_blocks, pred_size, window_size, patch_size, option)
#         total_num_blocks = int(len(data)/np.prod(meshed_blocks)) #-(window_size-1)-(pred_size)
#         window_adjust_length = (window_size-1)+pred_size ###Move the sliding window to cover the data lost on the edges

#         train_num_blocks = int(total_num_blocks*train_proportion)-window_adjust_length
#         val_num_blocks = int(total_num_blocks*val_proportion)

#         val_start_block = train_num_blocks

#         train_data_size = train_num_blocks*patches_per_block
#         val_data_size = val_num_blocks*patches_per_block

#         val_start_index = val_start_block*patches_per_block

#         train = windows[:train_data_size]
#         val = windows[val_start_index:val_start_index+val_data_size]
#         test = windows[val_start_index+val_data_size:]
  
#     else:
#         windows = to_windowed(data, meshed_blocks, pred_size, window_size, patch_size, option)

#         total_len = len(windows)
#         train_len = int(total_len*train_proportion)
#         val_len = int(total_len*val_proportion)
        
#         train = windows[0:train_len]
#         val = windows[train_len:(train_len+val_len)]
#         test = windows[(train_len+val_len):]
#     print(train.shape,val.shape,test.shape)
#     train_data = torch.from_numpy(train).float()
#     val_data = torch.from_numpy(val).float()
#     test_data = torch.from_numpy(test).float()
    
#     return train_data,val_data,test_data,scaler

def train_test_val_split(data, meshed_blocks, train_proportion = 0.6, val_proportion = 0.2, test_proportion = 0.2\
              , pred_size = 1, feature_size = 5, scale = False, window_size = 10, patch_size=(1,1,16), option='patch', scaler_type='standard'):
    
    if scale == True:
        if scaler_type not in ['standard','power_box','power_yeo','robust']:
            scaler_type = 'standard'
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'power_box':
            scaler = PowerTransformer(method='box-cox',standardize=True)
        elif scaler_type == 'power_yeo':
            scaler = PowerTransformer(method='yeo-johnson',standardize=True)
        elif scaler_type == 'robust':
            scaler = RobustScaler(with_centering=False,with_scaling=True,quantile_range=(25.0, 75.0),copy=True,unit_variance=False)
        elif scaler_type == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        print(f'Using scaler: {scaler_type}')
        data = scaler.fit_transform(data)
    else:
        scaler = None
    
    data_windowed = to_windowed(data, meshed_blocks, window_size, feature_size) #already float tensors

    total_len = data_windowed.shape[0]
    train_len = int(total_len*train_proportion)
    val_len = int(total_len*val_proportion)
    train_data = data_windowed[0:train_len]
    val_data = data_windowed[train_len:(train_len+val_len)]
    test_data = data_windowed[(train_len+val_len):]
    
    return train_data,val_data,test_data,scaler


## Adjust __init__ to fit the inputs
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,x,coords,timestamp,data,grid_size,window_size):
        self.x=x
        self.coords=coords
        self.timestamp=timestamp
        self.data = data
        self.window_size = window_size
        self.grid_dim = np.prod([d for d in grid_size])
 
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,idx):
        src_timestamp, tgt_timestamp = self.timestamp[idx], self.timestamp[idx+1]
        src_coords, tgt_coords = self.coords[idx], self.coords[idx+1]
        src_x, tgt_x = self.x[idx], self.x[idx+1]
        return( (src_x, tgt_x),\
                (src_coords, tgt_coords),\
                (src_timestamp, tgt_timestamp) )


def get_data_loaders(train_proportion = 0.5, test_proportion = 0.25, val_proportion = 0.25, \
                        pred_size =1, batch_size = 16, num_workers = 1, pin_memory = True, \
                        use_coords = True, use_time = True, test_mode = False, scale = False, \
                        window_size = 10, patch_size=(1,1,16), grid_size=(16,16,16), option='patch', predict_res=False, noise_std=0.01, scaler_type='standard'): 
    np.random.seed(1008)
    
    data_path='/scratch/yd1008/nyu-capstone/data_turb_dedt1_16'
    if predict_res:
        data_original, data, meshed_blocks, coords, timestamps = get_rho(data_path, predict_res=predict_res, noise_std =noise_std, var_name='rho')
    else:
        density, meshed_blocks, coords, timestamps = get_rho(data_path, predict_res=predict_res, noise_std=noise_std, var_name='rho') 
        vel1, _, _, _ = get_rho(data_path, predict_res=predict_res, noise_std=noise_std, var_name='vel1') 
        vel2, _, _, _ = get_rho(data_path, predict_res=predict_res, noise_std=noise_std, var_name='vel2') 
        vel3, _, _, _ = get_rho(data_path, predict_res=predict_res, noise_std=noise_std, var_name='vel3') 
        press, _, _, _ = get_rho(data_path, predict_res=predict_res, noise_std=noise_std, var_name='press') 
        data = np.stack((density, vel1, vel2, vel3, press), axis=-1)
        print(f'data shape: {data.shape}')


    ###FOR SIMPLE TEST SINE AND COSINE 
    #long_range_stationary_x_vals = (np.sin(2*np.pi*time_vec/period))+(np.cos(3*np.pi*time_vec/period)) + 0.25*np.random.randn(time_vec.size)
    ###FOR ARFIMA TEST
    #data = arfima([0.5,0.4],0.3,[0.2,0.1],10000,warmup=2^10)

    if use_coords:
        print('-'*20,'split for coords')
        train_coords,val_coords,test_coords, _ = train_test_val_split(\
            coords, meshed_blocks = meshed_blocks, train_proportion = train_proportion\
            , val_proportion = val_proportion, test_proportion = test_proportion\
            , pred_size = pred_size, feature_size = 3,scale = False, window_size = window_size, patch_size = patch_size, option = option)
        print(f'train_coords: {train_coords.shape}, val_coords: {val_coords.shape}, test_coords: {test_coords.shape}')


    if use_time:
        print('-'*20,'split for timestamp')
        train_timestamps,val_timestamps,test_timestamps, _ = train_test_val_split(\
            timestamps, meshed_blocks = meshed_blocks, train_proportion = train_proportion\
            , val_proportion = val_proportion, test_proportion = test_proportion\
            , pred_size = pred_size, feature_size = 1 ,scale = False, window_size = window_size, patch_size = patch_size, option = option)
        print(f'train_timestamps: {train_timestamps.shape}, val_timestamps: {val_timestamps.shape}, test_timestamps: {test_timestamps.shape}')

    print('-'*20,'split for data')
    train_data,val_data,test_data, scaler = train_test_val_split(\
        data, meshed_blocks = meshed_blocks, train_proportion = train_proportion\
        , val_proportion = val_proportion, test_proportion = test_proportion\
        , pred_size = pred_size, feature_size = 5,scale = scale, window_size = window_size, patch_size = patch_size, option = option, scaler_type = scaler_type)
    print(f'train_data: {train_data.shape}, val_data: {val_data.shape}, test_data: {test_data.shape}')

#----------------------------------------------------------------
### Save the original data table for reconstructing from residual predictions. May need to be optimized?
    if predict_res: 
        a = np.hstack([timestamps.reshape(-1,1), coords, data_original.reshape(-1,1)])
        a = a[np.argsort(a[:, 3])]
        a = a[np.argsort(a[:, 2], kind='stable')]
        a = a[np.argsort(a[:, 1], kind='stable')]
        a = a[np.argsort(a[:, 0], kind='stable')]
        if scale==True:
            data_origin_dict = {'time': a[:,0], 'x1': a[:,1], 'x2': a[:,2], 'x3': a[:,3], 'truth_original':scaler.inverse_transform(a[:,4])}
        elif scale==False:
            data_origin_dict = {'time': a[:,0], 'x1': a[:,1], 'x2': a[:,2], 'x3': a[:,3], 'truth_original':a[:,4]}
        data_origin_df = pd.DataFrame.from_dict(data_origin_dict)
        data_origin_df.to_csv('/scratch/yd1008/nyu-capstone/notebooks/turbulence_16_yd/tune_results/' + '/data_original.csv')
        #print(data_origin_df,flush=True)
 #----------------------------------------------------------------   

    if test_mode:
        train_val_data = torch.cat((train_data,val_data),0)
        train_val_coords = torch.cat((train_coords,val_coords),0)
        train_val_timestamps = torch.cat((train_timestamps,val_timestamps),0)

        ### Get the first block in test_original to perform rollout that gives back the original data based on predicted residuals

        dataset_train_val, dataset_test = CustomDataset(train_val_data,train_val_coords,train_val_timestamps,data,grid_size,window_size)\
                                    , CustomDataset(test_data,test_coords,test_timestamps,data,grid_size,window_size)
        train_val_loader = torch.utils.data.DataLoader(dataset_train_val, batch_size=batch_size, \
                                        drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                        persistent_workers=True, prefetch_factor = 16)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, \
                                        drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                        persistent_workers=True, prefetch_factor = 16) 
        return dataset_train_val, dataset_test, scaler, torch.from_numpy(data).float()
    if not test_mode:                           
        dataset_train, dataset_test, dataset_val = CustomDataset(train_data,train_coords,train_timestamps,data,grid_size,window_size)\
                                                ,CustomDataset(test_data,test_coords,test_timestamps,data,grid_size,window_size)\
                                                , CustomDataset(val_data,val_coords,val_timestamps,data,grid_size,window_size)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, \
                                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                            persistent_workers=True, prefetch_factor = 16)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, \
                                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                            persistent_workers=True, prefetch_factor = 16)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, \
                                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory,\
                                            persistent_workers=True, prefetch_factor = 16)
       
        return dataset_train,dataset_val, dataset_test, scaler, torch.from_numpy(data).float()

# img_dir = 'figs' ###dir to save images to
# pred_df = pd.read_csv('transformer_prediction_coords.csv',index_col=0) ###dir of csv file, or pandas dataframe


# grid_size = [16,16,16]
# axis_colnames = ['x1','x2','x3']
# slice_axis_index = 0
# pred_colname = 'prediction'
# truth_colname = 'truth'
# time_colname = 'time'


def plot_forecast(pred_df=None, grid_size=16, axis_colnames=['x1','x2','x3'], slice_axis_index=0, \
                  pred_colname='prediction',truth_colname='truth', time_colname='time',  \
                  plot_anime = True, img_dir = 'figs', config={}, file_prefix='test'):
    '''
    Notes: 
        please create two subfolders 'gif' and 'mp4' to save the animated slice plot for each trail 
    Parameters:
        pred_df: string of dir of csv file or pandas.Dataframe, predictions with coordinates and time
        grid_size: int or list-like, dimensions of one meshblock
        axis_colnames: list of strings, column names of coordinates in pred_df
        slice_axis_index: int, index of axis in axis_colnames to slice on 
        pred_colname: string, column name of predictions in pred_df
        truth_colname: string, column name of truths in pred_df
        time_colname: string, column name of timestamps in pred_df
        plot_anime: bool, animated plots will be saved if True 
        img_dir: str, path to folder where plots are saved 
        config: dictionary, the config of this plot, used for saving distinguishable animated plots for each trail
    Sample use:
        img_dir = 'figs' 
        pred_df = pd.read_csv('transformer_prediction_coords.csv',index_col=0) or 'transformer_prediction_coords.csv'
        grid_size = [16,16,16]
        axis_colnames = ['x1','x2','x3']
        slice_axis_index = 0
        pred_colname = 'prediction'
        truth_colname = 'truth'
        time_colname = 'time'
        plot_forecast(pred_df=pred_df, grid_size=grid_size, axis_colnames=axis_colnames, slice_axis_index=2, \
                        pred_colname=pred_colname,truth_colname=truth_colname, time_colname=time_colname,  \
                        plot_anime = True, img_dir = 'figs/', config=best_config)   
    '''
    if len(grid_size)!=3:
        grid_size = [grid_size]*3
    if type(pred_df)== str:
        preds_all = pd.read_csv(pred_df,index_col=None)
    else:
        preds_all = pred_df
    print(grid_size)
    v_max, v_min = max(preds_all[truth_colname].values),min(preds_all[truth_colname].values)
    v_max_res, v_min_res = max(preds_all[truth_colname].values-preds_all[pred_colname].values),min(preds_all[truth_colname].values-preds_all[pred_colname].values)
    print(v_max,v_min)
    predictions_per_simulation = np.prod(grid_size)
    slice_axis_colname = axis_colnames[slice_axis_index]
    #nonslice_axis_colname = axis_colnames.remove(slice_axis_colname)
    slice_axis_shape = grid_size.pop(slice_axis_index)
    nonslice_axis_shape = grid_size

    timestamps = sorted(preds_all[time_colname].unique())
    axis_vals = sorted(preds_all[slice_axis_colname].unique())

    ### create a dict to save all values
    result_dict = {}
    print('Processing dataframe...')
    for timestamp in timestamps:
        single_simulation_df = preds_all.loc[(preds_all[time_colname]==timestamp)]
        if single_simulation_df.shape[0]>=predictions_per_simulation:
            if single_simulation_df.shape[0]>predictions_per_simulation:
                single_simulation_df = single_simulation_df.groupby([time_colname]+axis_colnames).mean().reset_index()
            result_dict[timestamp] = {}
            result_dict[timestamp]['slice_axis_val'] = []
            result_dict[timestamp]['preds'] = []
            result_dict[timestamp]['truth'] = []
            for axis_val in axis_vals:
                slice_df = single_simulation_df.loc[single_simulation_df[slice_axis_colname]==axis_val]
                slice_preds = slice_df[pred_colname].values.reshape(nonslice_axis_shape)
                slice_truth = slice_df[truth_colname].values.reshape(nonslice_axis_shape)
                result_dict[timestamp]['slice_axis_val'].append(axis_val)
                result_dict[timestamp]['preds'].append(slice_preds)
                result_dict[timestamp]['truth'].append(slice_truth)  
            for item in result_dict[timestamp]:
                result_dict[timestamp][item] = np.array(result_dict[timestamp][item])
        else:
            print(f'Found {single_simulation_df.shape[0]} predictions in simulation at timestamp {timestamp}, but expect {predictions_per_simulation}')

    print('Generating plots...')
    ### plot for each timestamp   
    r2s = []
    mses = []
    maes = []
    evs = []
    for ts_idx,ts in enumerate(list(result_dict.keys())):
        fig,axes = plt.subplots(nrows = slice_axis_shape, ncols = 3, figsize=(40, 40),subplot_kw={'xticks': [], 'yticks': []})
        #plt.setp(axes, ylim=(0, 14),xlim=(0,2))
        plt.subplots_adjust(left=0.1,bottom=0, right=0.3, top=0.98, wspace=0, hspace=0.3)
        axis_val = result_dict[ts]['slice_axis_val']
        preds = result_dict[ts]['preds']
        truth = result_dict[ts]['truth']
        
        r2s.append(r2_score(truth.flatten(),preds.flatten()))
        mses.append(mean_squared_error(truth.flatten(),preds.flatten()))
        maes.append(mean_absolute_error(truth.flatten(),preds.flatten()))
        evs.append(explained_variance_score(truth.flatten(),preds.flatten()))
        
        v_min = np.min(truth)
        v_max = np.max(truth)
        for i in range(slice_axis_shape):
            if ts_idx==0:
                im_pred = axes[slice_axis_shape-i-1][0].imshow(preds[i],vmin=v_min, vmax=v_max, aspect='equal',animated=False)
                axes[slice_axis_shape-i-1][0].set_ylabel(f'Slice {i}',size=15)
                im_truth = axes[slice_axis_shape-i-1][1].imshow(truth[i],vmin=v_min, vmax=v_max, aspect='equal',animated=False)
                im_residual = axes[slice_axis_shape-i-1][2].imshow(truth[i]-preds[i],vmin=v_min_res, vmax=v_max_res,aspect='equal',animated=False)
                
                fig.colorbar(im_truth,ax=axes[slice_axis_shape-i-1][1])
                fig.colorbar(im_pred,ax=axes[slice_axis_shape-i-1][0])
                fig.colorbar(im_residual,ax=axes[slice_axis_shape-i-1][2])
            else:
                im_pred = axes[slice_axis_shape-i-1][0].imshow(preds[i],vmin=v_min, vmax=v_max, aspect='equal',animated=True)
                axes[slice_axis_shape-i-1][0].set_ylabel(f'Slice {i}',size=15)
                im_truth = axes[slice_axis_shape-i-1][1].imshow(truth[i],vmin=v_min, vmax=v_max, aspect='equal',animated=True)
                im_residual = axes[slice_axis_shape-i-1][2].imshow(truth[i]-preds[i],vmin=v_min_res, vmax=v_max_res, aspect='equal',animated=True)
                
                fig.colorbar(im_truth,ax=axes[slice_axis_shape-i-1][1])
                fig.colorbar(im_pred,ax=axes[slice_axis_shape-i-1][0])
                fig.colorbar(im_residual,ax=axes[slice_axis_shape-i-1][2])

        axes[-1,0].annotate('Forecast',(0.5, 0), xytext=(0, -30),textcoords='offset points', xycoords='axes fraction', ha='center', va='top', size=20)
        axes[-1,1].annotate('Truth',(0.5, 0), xytext=(0, -30),textcoords='offset points', xycoords='axes fraction', ha='center', va='top', size=20)
        axes[-1,2].annotate('Residual',(0.5, 0), xytext=(0, -30),textcoords='offset points', xycoords='axes fraction', ha='center', va='top', size=20)
        #axes[-1,0].annotate('Slice number', (0, 0.5), xytext=(-50, 0), textcoords='offset points', xycoords='axes fraction', ha='left', va='center', size=15, rotation=90)
        fig.suptitle(f'Forecasts on slices across {slice_axis_colname} at timestamp {ts}',x=0.2,y=1,size=20)
        plt.savefig(img_dir+f'/{file_prefix}_ts_{ts}_result_across_{slice_axis_colname}.png', bbox_inches="tight", facecolor='white')
        plt.close()
        ###plot r2:
    plt.plot(np.arange(len(list(result_dict.keys()))),mses,label='MSE')
    plt.plot(np.arange(len(list(result_dict.keys()))),maes,label='MAE')
    plt.plot(np.arange(len(list(result_dict.keys()))),r2s,label='R2')
    plt.plot(np.arange(len(list(result_dict.keys()))),evs,label='Explained Variance')
    plt.legend()
    plt.savefig(img_dir+f'{file_prefix}_metrics_rollout.png')
    plt.close()
        
    if plot_anime:
        imgs = []
        fig,axes = plt.subplots(nrows = slice_axis_shape, ncols = 3, figsize=(40, 40),subplot_kw={'xticks': [], 'yticks': []})
        plt.subplots_adjust(left=0.1,bottom=0, right=0.3, top=0.98, wspace=0, hspace=0.3)
        print('Generating animation...')
        for ts_idx,ts in enumerate(list(result_dict.keys())):
            axis_val = result_dict[ts]['slice_axis_val']
            preds = result_dict[ts]['preds']
            truth = result_dict[ts]['truth']
            tmp_imgs = []
            try:
                v_max = max(truth)
                v_min = min(truth)
            except:
                pass
            for i in range(slice_axis_shape):
                if ts_idx==0:
                    im_pred = axes[slice_axis_shape-i-1][0].imshow(preds[i],vmin=v_min, vmax=v_max, aspect='equal',animated=False)
                    axes[slice_axis_shape-i-1][0].set_ylabel(f'Slice {i}',size=15)
                    im_truth = axes[slice_axis_shape-i-1][1].imshow(truth[i],vmin=v_min, vmax=v_max, aspect='equal',animated=False)
                    im_residual = axes[slice_axis_shape-i-1][2].imshow(truth[i]-preds[i],vmin=v_min_res, vmax=v_max_res, aspect='equal',animated=False)
                else:
                    im_pred = axes[slice_axis_shape-i-1][0].imshow(preds[i],vmin=v_min, vmax=v_max, aspect='equal',animated=True)
                    axes[slice_axis_shape-i-1][0].set_ylabel(f'Slice {i}',size=15)
                    im_truth = axes[slice_axis_shape-i-1][1].imshow(truth[i],vmin=v_min, vmax=v_max, aspect='equal',animated=True)
                    im_residual = axes[slice_axis_shape-i-1][2].imshow(truth[i]-preds[i],vmin=v_min_res, vmax=v_max_res, aspect='equal',animated=True)
                    tmp_imgs.extend([im_pred,im_truth,im_residual])
            imgs.append(tmp_imgs)
        axes[-1,0].annotate('Forecast',(0.5, 0), xytext=(0, -30),textcoords='offset points', xycoords='axes fraction', ha='center', va='top', size=20)
        axes[-1,1].annotate('Truth',(0.5, 0), xytext=(0, -30),textcoords='offset points', xycoords='axes fraction', ha='center', va='top', size=20)
        axes[-1,2].annotate('Residual',(0.5, 0), xytext=(0, -30),textcoords='offset points', xycoords='axes fraction', ha='center', va='top', size=20)
        fig.suptitle(f'Forecasts on slices across {slice_axis_colname} animated',x=0.2,y=1,size=20)
        ani = animation.ArtistAnimation(fig, imgs, interval=500, repeat_delay = 1000, blit=True)
        try:
            writer = animation.FFMpegWriter(fps=30, bitrate=1800)
            ani.save(img_dir+"/mp4"+f"/{file_prefix}_{slice_axis_colname}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}.mp4", writer=writer) 
        except:
            print('Saving animation in .mp4 format, try installing ffmpeg package. \n Saving to .gif instead')
            ani.save(img_dir+"/gif"+f"/{file_prefix}_{slice_axis_colname}_pe{config['pe_type']}_batch{config['batch_size']}_window{config['window_size']}_patch{config['patch_size']}.gif")
        plt.close()
        
def plot_demo(num_timestamps, grid_size, patch_size, img_dir):
    '''
    Parameters:
        num_timestamps: int, number of simulations to draw
        grid_size: int or list-like, dimensions of one meshblock
        patch_size: list-like, dimensions of patches
        img_dir: str, path to folder where plots are saved
    '''
    if type(grid_size)==int:
        grid_size = [grid_size]*3
    def explode(data):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    def create_indices(grid_size,patch_size):
        n_voxels = np.zeros((grid_size[0], grid_size[1], grid_size[2]), dtype=bool)
        n_voxels[-patch_size[0]:, :patch_size[1], -patch_size[2]:] = True
        # n_voxels[-1, 0, :] = True
        # n_voxels[1, 0, 2] = True
        # n_voxels[2, 0, 1] = True
        facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')
        edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
        filled = np.ones(n_voxels.shape)

        filled_2 = explode(filled)
        fcolors_2 = explode(facecolors)
        ecolors_2 = explode(edgecolors)

        x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.05
        y[:, 0::2, :] += 0.05
        z[:, :, 0::2] += 0.05
        x[1::2, :, :] += 0.95
        y[:, 1::2, :] += 0.95
        z[:, :, 1::2] += 0.95
        return x,y,z,filled_2,fcolors_2,ecolors_2

    blocks_per_patch = np.prod(patch_size)
    x_fullgrid, y_fullgrid, z_fullgrid,filled_fullgrid,fcolors_fullgrid,ecolors_fullgrid = create_indices(grid_size,patch_size)
    x_flattened, y_flattened, z_flattened,filled_flattened,fcolors_flattened,ecolors_flattened = create_indices((blocks_per_patch,1,1),(blocks_per_patch,1,1))
    fig = plt.figure(figsize=(3*num_timestamps,6))
    transFigure = fig.transFigure.inverted()
    for ts in range(num_timestamps):
        ax1 = fig.add_subplot(2, num_timestamps, ts+1, projection='3d')
        ax1.voxels(x_fullgrid, y_fullgrid, z_fullgrid, filled_fullgrid, facecolors=fcolors_fullgrid, edgecolors=ecolors_fullgrid)
        ax1.set_axis_off()
        ax1.set_title(f'Simulation @ t={ts+1}')
        
        ax2 = fig.add_subplot(2, num_timestamps, num_timestamps+ts+1, projection='3d')
        ax2.voxels(x_flattened, y_flattened, z_flattened, filled_flattened, facecolors=fcolors_flattened, edgecolors=ecolors_flattened)
        ax2.set_axis_off()
        ax2.set_ylim(1-blocks_per_patch,1)
        ax2.set_zlim(1-blocks_per_patch,1)
        ax2.annotate('Flatten', xy=(0, 0), xytext=(-0.05, 0.12),fontsize=11)
        if ts!=num_timestamps-1:
            ax2.annotate('+', xy=(0, 0), xytext=(0.11, 0.06),fontsize=15)
        
        xyA = [0.01,-0.02]
        xyB = [0.02,0.065]
        coord1 = transFigure.transform(ax1.transData.transform(xyA))
        coord2 = transFigure.transform(ax2.transData.transform(xyB))
        arrow = patches.FancyArrowPatch(coord1,coord2,shrinkA=0,shrinkB=0,transform=fig.transFigure,color='Black',\
                                        arrowstyle="-|>",mutation_scale=30,linewidth=1.5,)
        fig.patches.append(arrow)

    #plt.show()
    plt.savefig(img_dir+'patch_demo.png', bbox_inches="tight", facecolor='white')