#!/bin/bash python
import os
import numpy as np
import pandas as pd
import math
import time
from _arfima import arfima
from athena_read import *

### VARIABLES(KEYS) IN DATASET
# dict_keys(['Coordinates', 'DatasetNames', 'MaxLevel', 'MeshBlockSize', 'NumCycles', \
# 'NumMeshBlocks', 'NumVariables', 'RootGridSize', 'RootGridX1', 'RootGridX2', 'RootGridX3', \
# 'Time', 'VariableNames', 'x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', 'rho', 'press', 'vel1', 'vel2', 'vel3'])
def get_csv(data_path, csv_dir, target_var = 'rho', downsample = True, grid_size = 16):
    """
    grid_size: the size after downsampling if downsample=True, else the original size
    """
    lst = sorted(os.listdir(data_path))[4:-1]
    data = []
    coords = []
    timestamps = []
    for name in lst:
        path = data_path+'/'+name
        d = athdf(path)
        coord = np.transpose(np.array(np.meshgrid(np.arange(grid_size),np.arange(grid_size),np.arange(grid_size))), axes=[2,1,3,0]).reshape(-1,3)
        data.append(d[target_var])
        timestamp_repeated = [d['Time']]*grid_size**3
        timestamps.extend(timestamp_repeated) 
        coords.extend(coord)
        #print(f"Name: {name}, keys: {d.keys()}", flush=True)#x1v: {d['x1v']}, x2v: {d['x2v']}, x3v:{d['x3v']}

    data = np.array(data)
    if downsample:
        downsampled = 'downsampled'
        data_downsampled = np.zeros([data.shape[0], grid_size, grid_size, grid_size])
        coef = data.shape[1] // grid_size
        for time in range(data.shape[0]):
            for x1 in range(grid_size):
                for x2 in range(grid_size):
                    for x3 in range(grid_size):
                        data_downsampled[time, x1, x2, x3] = data[time, x1:x1+coef, x2:x2+coef, x3:x3+coef].mean()
        data_original = data_downsampled.flatten()
    else:
        downsampled = 'original'
        data_original = data.flatten()
    timestamps = np.array(timestamps)
    coords = np.array(coords)


    a = np.hstack([timestamps.reshape(-1,1), coords, data_original.reshape(-1,1)])
    a = a[np.argsort(a[:, 3])]
    a = a[np.argsort(a[:, 2], kind='stable')]
    a = a[np.argsort(a[:, 1], kind='stable')]
    a = a[np.argsort(a[:, 0], kind='stable')]
    
    original_dict = {'time': a[:,0], 'x1': a[:,1], 'x2': a[:,2], 'x3': a[:,3], 'target': a[:,4]}
    df = pd.DataFrame.from_dict(original_dict)
    df.to_csv(csv_dir + f'/{target_var}_{downsampled}_{grid_size}.csv')
    

if __name__ == "__main__":
    data_path = '/scratch/zh2095/data_turb_dedt1_128'
    csv_dir = '/scratch/zh2095/nyu-capstone/data'
    start_time = time.time()
    get_csv(data_path=data_path, csv_dir=csv_dir, target_var='rho', downsample=True, grid_size=16)
    print(f'time for generatig csv: {time.time()-start_time} s', flush=True)