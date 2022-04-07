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

from tmsa import TMSAG
from unet_3d import UNet

def reshape_3d(x, grid_size,):
    return x.reshape((-1,)+grid_size)

def roll_block(x, shift_size, reverse=False):
    '''
    meshblock: torch.Tensor size (N, D, H, W, C) #meshblocks x (nx1 x nx2 x nx3) x #channels
    shift_size: tuple (St, Sd, S)
    '''
    if reverse:
        shift_size = [-s for s in list(shift_size)]
    return torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3]), dims=(0,1,2,3))

def block_to_patch(x, patch_size, pad_size=1, stride_size = None):
    '''
    x: (N,H,D,W,C) input data
    return:
        patched_src: (num_patch_x1 x num_patch_x2 x num_patch_x3, window_size x patch_size_1 x patch_size_2 x patch_size_3, num_channel)
        patched_tgt: patched_src shifted by pred_size, with mean padding on the predicted patches
    '''
    N, _, _, _, _ = x.shape
    stride_size = stride_size or patch_size
    p1, p2, p3 = patch_size
    s1, s2, s3 = stride_size
    if pad_size>0:
        #x_padded = torch.cat([x,Reduce(x,'n g1 g2 g3 c -> g1 g2 g3 c', 'mean').unsqueeze(0)],dim=0) # mean padding only pad size 1 is implemented
        x_padded = torch.cat([x,x[-pad_size:]],dim=0) # same padding for the target block
        # x_padded = F.pad(x,(0,0,0,0,0,0,0,0,0,pad_size)) # zero padding for target block
    x_padded = x_padded.unfold(0,N,pad_size).unfold(1,p1,s1).unfold(2,p2,s2).unfold(3,p3,s3)
    src, tgt = rearrange(x_padded, 'nb n1 n2 n3 c b p1 p2 p3 -> nb (n1 n2 n3) (b p1 p2 p3) c')
    return src, tgt

def patch_to_block(x, window_size, patch_size, grid_size):
    '''
    Inverse of block_to_patch, refer to block_to_patch documentation
    '''
    _, _, c = x.shape
    p1, p2, p3 = patch_size
    g1, g2, g3 = grid_size
    n1, n2, n3 = g1//p1, g2//p2, g3//p3
    rearranged = rearrange(x, '(n1 n2 n3) (b p1 p2 p3) c -> b n1 n2 n3 c p1 p2 p3', \
                       p1=p1, p2=p2, p3=p3, b=window_size, n1=n1, n2=n2, n3=n3)
    return rearranged.permute(0,1,5,2,6,3,7,4).reshape(window_size,g1,g2,g3,c)  

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout= 0.1, max_len= 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEmbedding3D(nn.Module):
    def __init__(self, d_model_, grid_size, max_len=5000):
        super(PositionalEmbedding3D, self).__init__()
        # Compute the positional encodings once in log space.
        if d_model_%6!=0:
            raise Exception('d_model_ should be divisible by 6')
        if type(grid_size) == int:
            grid_x, grid_y, grid_z = grid_size,grid_size,grid_size
        elif len(grid_size) == 3:
            grid_x, grid_y, grid_z = grid_size
        else:
            raise Exception(f'Expect grid_size to be 3-dimensional, or scaler if all dimensions are the same, got {grid_size.shape} instead')      
        self.d_model_ = d_model_
        d_model = int(np.floor(d_model_/3))

        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe_x = torch.zeros(max_len, d_model).float()
        #pe_x.require_grad = False
        position_x = torch.arange(0, max_len).float().unsqueeze(1)#.unsqueeze(1)
        pe_x[:, 0::2] = torch.sin(position_x * div_term)
        pe_x[:, 1::2] = torch.cos(position_x * div_term)
        pe_x = pe_x[:grid_x,:].unsqueeze(1).unsqueeze(1)

        pe_y = torch.zeros(max_len, d_model).float()
        #pe_y.require_grad = False
        position_y = torch.arange(0, max_len).float().unsqueeze(1)
        pe_y[:, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 1::2] = torch.cos(position_y * div_term)
        pe_y = pe_y[:grid_y,:].unsqueeze(1)

        pe_z = torch.zeros(max_len, d_model).float()
        #pe_z.require_grad = False
        position_z = torch.arange(0, max_len).float().unsqueeze(1)
        pe_z[:, 0::2] = torch.sin(position_z * div_term)
        pe_z[:, 1::2] = torch.cos(position_z * div_term)
        pe_z = pe_z[:grid_z,:]

        pe = torch.zeros((grid_x,grid_y,grid_z,d_model_))
        pe.require_grad = False
        pe[:,:,:,:d_model] = pe_x
        pe[:,:,:,d_model:2*d_model] = pe_y
        pe[:,:,:,2*d_model:] = pe_z
        self.register_buffer('pe', pe)


    def forward(self,batch):
        '''
        :param x: 3d tensor of size (batch, seq_len, num_axis)
        '''
        # batch_size, x, y, z = batch.shape
        # pe3d = torch.zeros(batch_size, 1, 1, 1)
        coords = batch.view(-1,batch.shape[-1]).int()
        coords = coords.transpose(0,1).int()
        xs,ys,zs = coords[0].long(),coords[1].long(),coords[2].long()
        return self.pe[xs,ys,zs].view(batch.shape[0],-1,self.d_model_)  #torch.stack([self.pe[i,j,k] for i,j,k in zip(xs,ys,zs)]).view(batch.shape[0],-1,self.d_model_)

class conv_3d(nn.Module):
    def __init__(self, dim, dim_out, num_layer = 3, kernel_size = 5, stride = 1, padding = 2, dilation = 1, padding_mode = 'reflect'):
        super().__init__()
        self.dim = dim
        layers = []
        for _ in range(num_layer-1):
            layers.append(nn.Conv3d(in_channels=dim, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode))
            layers.append(nn.BatchNorm3d(dim_out))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
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

class Transformer(nn.Module):
    def __init__(self,all_data,feature_size=250,num_enc_layers=1,num_dec_layers=1,d_ff = 256, dropout=0.1,num_head=2,pe_type='3d',\
                grid_size=(16,16,16),mask_type=None,patch_size=(2,2,2),window_size=5,pred_size=1,decoder_only=False,tmsa_config={},conv_config={}):
        '''
        mask_type: 'patch' if using cuboic patches, which masks by patch instead of elements. Default to None (square_subsequent mask)
        '''
        super(Transformer, self).__init__()

        self.all_data = all_data
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.patch_length = np.prod(patch_size)
        self.window_size = window_size
        self.pred_size = pred_size
        self.grid_size = grid_size
        self.grid_dim = np.prod(grid_size)
        self.decoder_only = decoder_only

        self.pe_type = pe_type
        self.src_mask = mask_type
        self.pos_encoder = PositionalEncoding(feature_size)
        self.pos3d_encoder = PositionalEmbedding3D(feature_size,grid_size)
        self.temporal_encoder = TemporalEmbedding(input_dim=1, output_dim=feature_size, activation='sin')

        self.conv_type = conv_config['conv_type']
        if conv_config['conv_type'] == 'UNet':
            print('Using UNET')
            self.conv_embedding = UNet(in_channels = feature_size,
                                        out_channels = feature_size//2,
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
        else:
            print('Using regular CONV')
            self.conv_embedding = conv_3d(feature_size,feature_size, num_layer = conv_config['num_layer'], kernel_size = conv_config['kernel_size'], padding = conv_config['padding'], padding_mode = conv_config['padding_mode']) ### 5 a good number?


        self.use_tmsa = tmsa_config['use_tmsa']
        self.use_tgt_tmsa = tmsa_config['use_tgt_tmsa']
        if self.use_tmsa:
            self.tmsa_block = TMSAG(dim = feature_size,
                            dim_out = feature_size//2,
                            depth = tmsa_config['depth'],
                            num_heads = tmsa_config['num_heads'],
                            window_patch_size=tmsa_config['window_patch_size'],
                            shift_size=tmsa_config['shift_size'],
                            mut_attn=True,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_path=0., #drop probability in drop_path
                            norm_layer=nn.LayerNorm,
                            use_checkpoint_attn=False,
                            use_checkpoint_ffn=False
                            )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, \
            nhead=num_head, dropout=dropout, dim_feedforward = d_ff)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)      
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, \
            nhead=num_head, dropout=dropout, dim_feedforward = d_ff)  
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_dec_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_coord, tgt_coord, src_ts, tgt_ts, time_map_indices):

        B,N,_ = src.shape
        device = src.device
         

        # try:
        #     self.test_plot(tmsa_embedded[0][:,:,6].squeeze(-1).detach().cpu().numpy(),\
        #                     block[0][:,:,6].squeeze(-1).detach().cpu().numpy(),\
        #                     'tmsa_n0_a_a_6.png')
        #     self.test_plot(tmsa_embedded[4][:,:,8].squeeze(-1).detach().cpu().numpy(),\
        #                     block[4][:,:,8].squeeze(-1).detach().cpu().numpy(),\
        #                     'tmsa_n4_a_a_6.png')
        #     self.test_plot(tmsa_embedded[8][:,:,14].squeeze(-1).detach().cpu().numpy(),\
        #                     block[8][:,:,14].squeeze(-1).detach().cpu().numpy(),\
        #                     'tmsa_n8_a_a_14.png')
        # except:
        #     pass
            
        if self.pe_type == '1d':
            src = src.permute(1,0,2)
            tgt = tgt.permute(1,0,2)
            src = self.pos_encoder(src)
            tgt = self.pos_encoder(tgt)
            src = src.permute(1,0,2)
            tgt = tgt.permute(1,0,2)

        elif self.pe_type == '3d':
            src = src + self.pos3d_encoder(src_coord)
            tgt = tgt + self.pos3d_encoder(tgt_coord)
            
        elif self.pe_type == '3d_temporal':

            src = src + self.pos3d_encoder(src_coord) + self.temporal_encoder(src_ts)
            tgt = tgt + self.pos3d_encoder(tgt_coord) + self.temporal_encoder(tgt_ts)
            
        #print(f'src shape {src.shape}, tgt shape: {tgt.shape}')
        ### generate patch mask
        block = patch_to_block(src,self.window_size,self.patch_size,self.grid_size)
        ### CONV EMBEDDING
        if self.conv_type == 'UNet':
            # print(f'{self.conv_embedding}', flush=True)
            conv_embedded = self.conv_embedding(block.permute(0,4,1,2,3)).permute(0,2,3,4,1)
        else:
            conv_embedded = self.conv_embedding(block,self.grid_size)
        # src_conv, tgt_conv = block_to_patch(conv_embedded, self.patch_size, pad_size=1)

        ### TMSA
        # src = src + src_conv
        # tgt = tgt + tgt_conv
        if self.use_tmsa:
            # conv_embedded = block #+ conv_embedded
            tmsa_embedded = self.tmsa_block(block)
            # src_tmsa, tgt_tmsa = block_to_patch(tmsa_embedded, self.patch_size, pad_size=1)
            # src = src + src_tmsa
            # if self.use_tgt_tmsa:
            #     tgt = tgt  + tgt_tmsa
            # else:
            #     tgt = tgt
        # print(f'conv: {conv_embedded.shape}, tmsa: {tmsa_embedded.shape}', flush=True)
        src_embedded, tgt_embedded = block_to_patch(torch.cat([conv_embedded,tmsa_embedded], dim=4), self.patch_size, pad_size=1)
        src = src + src_embedded
        if self.use_tgt_tmsa:
            tgt = tgt  + tgt_embedded
        else:
            tgt = tgt


        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        if self.src_mask == 'patch':
            self.mask = self._generate_patch_mask(self.patch_size,self.window_size,0).to(device)
            self.dec_src_mask = self._generate_patch_mask(self.patch_size,self.window_size,-1).to(device)
            #print(f'Using patch mask: mask: {self.mask}, shape: {self.mask.shape}', flush=True)
            #self.src_mask = mask
        elif self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            self.mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
            #print(f'Using original mask: mask: {self.mask}, shape: {self.mask.shape}', flush=True)
            #self.src_mask = mask

        # print('PE out shape: ',self.pos_encoder(src).shape)
        # print('PE out shape: ',self.pos_encoder(src).shape)
        # src = self.pos_encoder(src)
        # tgt = self.pos_encoder(tgt)
        #print(f'after pe src shape {src.shape}, tgt shape: {tgt.shape}')
        if self.decoder_only:
            #print('USING DECODER ONLY')
            #print(f'after pe src shape {src.shape}, tgt shape: {tgt.shape}', flush=True) #(sequence, batch_size, feature_size)
            output_dec = self.transformer_decoder(tgt,src,self.mask,self.dec_src_mask)
        else:
            output_enc = self.transformer_encoder(src,self.mask) 
            #print(f'encoder output shape: {output_enc.shape}', flush=True)
            output_dec = self.transformer_decoder(tgt,output_enc,self.mask, self.dec_src_mask)
            #print(f'decoder output shape: {output_dec.shape}', flush=True)
        output = self.decoder(output_dec)

        #print('output shape: ',output.shape)
        return output.permute(1,0,2)

    # def generate_cov3d_embedding(self, data):
    #     return self.conv_embedding(data, self.grid_size)

    def test_plot(data1, data2, name):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(data1, interpolation='None')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(data2, interpolation='None')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        fig.savefig('/scratch/yd1008/nyu_capstone_2/notebooks/turbulence_16_yd/tune_results/'+name)
        return None

    def block_time_coord_indexing(self, timestamp, coords, time_map_indices, conv_embedded_blocks, device, shift_size=0):
        #time_indices = torch.repeat_interleave(torch.arange(shift_size,self.window_size+shift_size),self.patch_length).unsqueeze(-1)
        time_indices = torch.repeat_interleave(torch.zeros(1),self.patch_length).unsqueeze(-1)
        time_coord_indices = torch.cat([time_indices.to(device),coords],dim=1)
        time_coord_indices = time_coord_indices.long()
        return conv_embedded_blocks[time_coord_indices.transpose(0,1).chunk(chunks=4, dim=0)].squeeze(0) #shape NxC get corresponding conv embeddings on timestamp and coords, 4 for 4 dimensional indices

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_patch_mask(self,patch_size,window_size,shift=0):
        patch_prod = torch.prod(torch.tensor(patch_size))
        mask = (torch.triu(torch.ones(window_size, window_size),shift) == 1).transpose(0, 1)
        mask = torch.repeat_interleave(mask, patch_prod, dim=0)
        mask = torch.repeat_interleave(mask, patch_prod, dim=1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask