#!/bin/bash python
import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange, reduce as Reduce
import numpy as np
import math
import time

from unet_3d import UNet


class conv_3d(nn.Module):
    def __init__(self, dim, dim_out, num_layer = 3, kernel_size = 5, stride = 1, padding = 2, dilation = 1, padding_mode = 'reflect'):
        super().__init__()
        self.dim = dim
        layers = []
        for _ in range(num_layer-1):
            layers.append(nn.Conv3d(in_channels=dim, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode))
            layers.append(nn.BatchNorm3d(dim_out))
            #layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(nn.Conv3d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode))
        layers.append(nn.BatchNorm3d(dim_out))
        self.conv = nn.Sequential(*layers)
        # self.conv = nn.Sequential(
        #     nn.Conv3d(in_channels=dim, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv3d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv3d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode),
        #     # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     # nn.Conv3d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode)
        # )
        self.init_weights()

    def forward(self, x, grid_size, pad_block=0, return_flattened = False):
        '''
        x: flattened input 
        '''
        x = x.contiguous().view((-1,)+grid_size+(self.dim,)) 
        # if pad_block>0:
        #     x = torch.cat([x,Reduce(x,'n g1 g2 g3 c -> g1 g2 g3 c', 'mean')],dim=0) ### mean padding the predicted block
        #     #x = F.pad(x,(0,0,0,0,0,0,0,0,0,pad_block)) #pad 0s for the block predicted
        x = rearrange(x, "n d h w c -> n c d h w")
        x_conv = self.conv(x)
        if return_flattened:
            x_conv = rearrange(x_conv, "n c d h w -> (n d h w) c")
        else:
            x_conv = rearrange(x_conv, "n c d h w -> n d h w c")
        return x_conv
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, activation='sin'):
        super(TemporalEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w0 = nn.parameter.Parameter(torch.randn(input_dim, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(input_dim, output_dim-1))
        self.b = nn.parameter.Parameter(torch.randn(output_dim-1))
        if activation == 'sin':
            self.f = torch.sin
        elif activation == 'cos':
            self.f = torch.cos
        self.fc1 = nn.Linear(output_dim, output_dim)
    
    def l1(self, tau):
        f = self.f
        v1 = f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], 2)

    def forward(self, x):
        #x = x.unsqueeze(2)
        x = self.l1(x)
        x = self.fc1(x)
        return x

class Unet(nn.Module):
    def __init__(self,feature_size=250, conv_config={}):

        super(Unet, self).__init__()

        self.feature_size = feature_size

        self.conv_encoder = nn.Conv3d(in_channels=5, out_channels=feature_size, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='circular')#conv_3d(5,feature_size, num_layer = 2, kernel_size = 3, padding = 1, padding_mode = 'circular')

        self.conv_embedding = UNet(in_channels = feature_size,
                                    out_channels = feature_size,
                                    n_blocks = conv_config['num_layer'],
                                    start_filts = conv_config['start_filts'],
                                    up_mode = 'transpose',
                                    merge_mode = 'concat',
                                    planar_blocks = (),
                                    batch_norm = 'unset',
                                    attention = False,
                                    activation = 'rrelu',
                                    normalization = 'batch',
                                    full_norm = True,
                                    dim = 3,
                                    conv_mode = 'same')

        self.conv_decoder = nn.Conv3d(in_channels=feature_size, out_channels=5, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='circular')#conv_3d(feature_size,5, num_layer = 2, kernel_size = 3, padding = 1, padding_mode = 'circular')

        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, tgt, src_coord, tgt_coord, src_ts, tgt_ts):

        device = src.device
    
        src_block = self.conv_encoder(src.permute(0,4,1,2,3)) 
        ### CONV EMBEDDING
        src_conv_embedded = self.conv_embedding(src_block)

        src_conv_embedded = self.conv_decoder(src_conv_embedded).permute(0,2,3,4,1)
 
        return src_conv_embedded
        

   