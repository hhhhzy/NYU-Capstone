#!/bin/bash python
import torch 
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange, reduce as Reduce
import numpy as np
import math
import time

from tmsa import TMSAG
from unet_3d import UNet

# def reshape_3d(x, grid_size,):
#     return x.reshape((-1,)+grid_size)

def roll_block(x, shift_size, reverse=False):
    '''
    meshblock: torch.Tensor size (N, D, H, W, C) #meshblocks x (nx1 x nx2 x nx3) x #channels
    shift_size: tuple (Sn, Sd, Sh, Sw)
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
    if pad_size==1:
        #x_padded = torch.cat([x,Reduce(x,'n g1 g2 g3 c -> g1 g2 g3 c', 'mean').unsqueeze(0)],dim=0) # mean padding only pad size 1 is implemented
        # x_padded = torch.cat([x,x[-pad_size:]],dim=0) # same padding for the target block
        x_padded = F.pad(x,(0,0,0,0,0,0,0,0,0,pad_size)) # zero padding for target block
        x_padded = x_padded.unfold(0,N,pad_size).unfold(1,p1,s1).unfold(2,p2,s2).unfold(3,p3,s3)
        out_x, out_y = rearrange(x_padded, 'nb n1 n2 n3 c b p1 p2 p3 -> nb (n1 n2 n3) (b p1 p2 p3) c')
        return out_x, out_y
    if pad_size==0:
        x_padded = x.unfold(1,p1,s1).unfold(2,p2,s2).unfold(3,p3,s3)
        out_x = rearrange(x_padded, 'b n1 n2 n3 c p1 p2 p3 -> (n1 n2 n3) (b p1 p2 p3) c')
        return out_x

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
            #layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        if num_layer ==1:
            layers.append(nn.Conv3d(in_channels=dim, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode))
        else:
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

class Transformer_Decoder(nn.Module):
    def __init__(self,feature_size,num_layers, decoder_layer):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.norm = nn.LayerNorm(feature_size, eps=1e-5)
    
    def forward(self,tgt,memory,tgt_mask,memory_mask,embedding,embedding_insert_layer):
        assert embedding_insert_layer <= self.num_layers
        assert embedding.shape == tgt.shape
        output = tgt
        for i,layer in enumerate(self.layers):
            if embedding_insert_layer == (i+1):
                output = output + embedding
            output = layer(output,memory,tgt_mask,memory_mask)

        output = self.norm(output)
        
        return output

class Transformer(nn.Module):
    def __init__(self,all_data,feature_size=250,num_enc_layers=1,num_dec_layers=1,d_ff = 256, dropout=0.1,num_head=2,pe_type='3d',encoder_decoder_type='conv',\
                grid_size=(16,16,16),mask_type=None,patch_size=(2,2,2),window_size=5,pred_size=1,decoder_only=False,tmsa_config={},conv_config={},load_prev_acrc=False,\
                ablation={'tmsa':True, 'temp_embed':True, 'encoder':True}):
        '''
        mask_type: 'patch' if using cuboic patches, which masks by patch instead of elements. Default to None (square_subsequent mask)
        '''
        super(Transformer, self).__init__()
        self.encoder_decoder_type = encoder_decoder_type
        self.all_data = all_data
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.patch_length = np.prod(patch_size)
        self.window_size = window_size
        self.pred_size = pred_size
        self.grid_size = grid_size
        self.grid_dim = np.prod(grid_size)
        self.decoder_only = decoder_only
        self.num_dec_layers = num_dec_layers
        self.pe_type = pe_type
        self.src_mask = mask_type
        self.pos_encoder = PositionalEncoding(feature_size)
        self.pos3d_encoder = PositionalEmbedding3D(feature_size,grid_size)
        self.pos_insert = tmsa_config['pos_insert']
        self.temporal_encoder = TemporalEmbedding(input_dim=1, output_dim=feature_size, activation='sin')

        # self.norm1_src = nn.InstanceNorm3d(feature_size, eps=1e-5)
        # self.norm1_tgt = nn.InstanceNorm3d(feature_size, eps=1e-5)
        # self.norm2_src = nn.InstanceNorm3d(feature_size, eps=1e-5)
        # self.norm2_tgt = nn.InstanceNorm3d(feature_size, eps=1e-5)
        self.norm1_src = nn.LayerNorm(feature_size, eps=1e-5)

        self.conv_encoder = conv_3d(5,feature_size, num_layer = 2, kernel_size = 3, padding = 1, padding_mode = 'circular')

       
        self.use_tmsa = tmsa_config['use_tmsa']
        self.use_tgt_tmsa = tmsa_config['use_tgt_tmsa']
        if self.use_tmsa and ablation['tmsa']:
            self.tmsa_block = TMSAG(dim = feature_size,
                            dim_out = feature_size,
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
                            
        if not decoder_only:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, \
                nhead=num_head, dropout=dropout, dim_feedforward = d_ff)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)      

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, \
            nhead=num_head, dropout=dropout, dim_feedforward = d_ff)  
        self.transformer_decoder = Transformer_Decoder(feature_size,num_dec_layers,self.decoder_layer)

        self.linear_decoder = nn.Sequential(nn.Linear(feature_size,feature_size//2),
                                                nn.Linear(feature_size//2,5))
        self.ablation = ablation

        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        try:
            self.transformer_decoder.bias.data.zero_()
            self.transformer_decoder.weight.data.uniform_(-initrange, initrange)
        except:
            print('Error in initializing decoder weights')

    def forward(self, src, tgt, src_coord, tgt_coord, src_ts, tgt_ts, shift_size=(0,0,0,0), temporal_insert_layer=2):
        
        ###ROLL ALL INPUTS
        src = roll_block(src, shift_size, reverse=False)
        tgt = roll_block(tgt, shift_size, reverse=False)
        src_coord = roll_block(src_coord, shift_size, reverse=False)
        tgt_coord = roll_block(tgt_coord, shift_size, reverse=False)
        src_ts = roll_block(src_ts, shift_size, reverse=False)
        tgt_ts = roll_block(tgt_ts, shift_size, reverse=False)

        device = src.device
        src_coord_patch = block_to_patch(src_coord, self.patch_size, pad_size=0)   
        tgt_coord_patch = block_to_patch(tgt_coord, self.patch_size, pad_size=0) 
        src_ts_patch = block_to_patch(src_ts, self.patch_size, pad_size=0)   
        tgt_ts_patch = block_to_patch(tgt_ts, self.patch_size, pad_size=0)

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
            src_pos_embed = self.pos3d_encoder(src_coord_patch) 
            src_ts_embed = self.temporal_encoder(src_ts_patch)
            tgt_pos_embed = self.pos3d_encoder(tgt_coord_patch) 
            tgt_ts_embed = self.temporal_encoder(tgt_ts_patch)
        
        if not self.ablation['temp_embed']:
            src_ts_embed = torch.zeros_like(src_ts_embed)
            tgt_ts_embed = torch.zeros_like(tgt_ts_embed)
            
            
        ### conv encoder to change demension from 5 to feature_size

        src = self.norm1_src(self.conv_encoder(src,self.grid_size)) #+ src_pos_temp_block
        tgt = self.norm1_src(self.conv_encoder(tgt,self.grid_size)) #+ tgt_pos_temp_block



        src_pos_embed_b = patch_to_block(src_pos_embed, self.window_size, self.patch_size, self.grid_size)
        tgt_pos_embed_b = patch_to_block(tgt_pos_embed, self.window_size, self.patch_size, self.grid_size)

        #if not self.tmsa_with_conv:
        if self.ablation['tmsa']:
            if self.pos_insert in ['tmsa','both']:
                src_tmsa_embedded = self.tmsa_block(src+src_pos_embed_b)#+src_conv_embedded
                tgt_tmsa_embedded = self.tmsa_block(tgt+tgt_pos_embed_b)#+tgt_conv_embedded
            else:
                src_tmsa_embedded = self.tmsa_block(src)
                tgt_tmsa_embedded = self.tmsa_block(tgt)
        else:
            src_tmsa_embedded = src
            tgt_tmsa_embedded = tgt

        if self.pos_insert in ['transformer','both']:
            src = block_to_patch(src + src_tmsa_embedded, self.patch_size, pad_size=0) + src_pos_embed #+ src_ts_embed
            tgt = block_to_patch(tgt + tgt_tmsa_embedded, self.patch_size, pad_size=0) + tgt_pos_embed #+ tgt_ts_embed
        else:
            src = block_to_patch(src + src_tmsa_embedded, self.patch_size, pad_size=0)  
            tgt = block_to_patch(tgt + tgt_tmsa_embedded, self.patch_size, pad_size=0) 
        
      
        ### Transformer
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        if self.src_mask == 'patch':
            self.mask = self._generate_patch_mask(self.patch_size,self.window_size,0).to(device)
            self.dec_src_mask = self._generate_patch_mask(self.patch_size,self.window_size,0).to(device)

        elif self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            self.mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)

        if self.decoder_only:
            output_dec = self.transformer_decoder(tgt,src,self.mask,None, tgt_ts_embed.permute(1,0,2), temporal_insert_layer)
            # output_dec = self.decoder_layer(tgt,src,self.mask,self.dec_src_mask)
        else:
            output_enc = self.transformer_encoder(src,self.mask) 
            output_dec = self.transformer_decoder(tgt,output_enc,self.mask, self.dec_src_mask, tgt_ts_embed.permute(1,0,2), temporal_insert_layer)
        # print(f'output patch transformer shape: {output_dec.shape}')
        # print(f'output patch transformer: {output_dec[:27,-2,0]}')
        # print(f'output patch shape: {output.shape}')
        # print(f'output patch: {output[:27,-2,0]}')
        output = output_dec.permute(1,0,2)

        output = patch_to_block(output, self.window_size, self.patch_size, self.grid_size)

        output = self.linear_decoder(output)
        # output = self.linear_decoder(output)

        # print(f'output block: {output[-2,:3,:3,:3,0]}')
        ###ROLL BACK TO ORIGINAL ORDER
        output = roll_block(output, shift_size, reverse=True)

        return output

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