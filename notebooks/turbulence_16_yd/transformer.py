#!/bin/bash python
import torch 
import torch.nn as nn
import numpy as np
import math

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
        pe_x.require_grad = False
        position_x = torch.arange(0, max_len).float().unsqueeze(1)#.unsqueeze(1)
        pe_x[:, 0::2] = torch.sin(position_x * div_term)
        pe_x[:, 1::2] = torch.cos(position_x * div_term)
        pe_x = pe_x[:grid_x,:].unsqueeze(1).unsqueeze(1)

        pe_y = torch.zeros(max_len, d_model).float()
        pe_y.require_grad = False
        position_y = torch.arange(0, max_len).float().unsqueeze(1)
        pe_y[:, 0::2] = torch.sin(position_y * div_term)
        pe_y[:, 1::2] = torch.cos(position_y * div_term)
        pe_y = pe_y[:grid_y,:].unsqueeze(1)

        pe_z = torch.zeros(max_len, d_model).float()
        pe_z.require_grad = False
        position_z = torch.arange(0, max_len).float().unsqueeze(1)
        pe_z[:, 0::2] = torch.sin(position_z * div_term)
        pe_z[:, 1::2] = torch.cos(position_z * div_term)
        pe_z = pe_z[:grid_z,:]

        pe = torch.zeros((grid_x,grid_y,grid_z,d_model_))
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
        xs,ys,zs = coords[0],coords[1],coords[2]
        return torch.stack([self.pe[i,j,k] for i,j,k in zip(xs,ys,zs)]).view(batch.shape[0],-1,self.d_model_)

class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1, padding_mode='replicate')
        self.init_weights()
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
    def init_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

class Tranformer(nn.Module):
    def __init__(self,feature_size=250,num_enc_layers=1,num_dec_layers=1,d_ff = 256, dropout=0.1,num_head=2,grid_size=16):
        super(Tranformer, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.pos3d_encoder = PositionalEmbedding3D(feature_size,grid_size)
        self.token_embedding = TokenEmbedding(feature_size)

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
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src, tgt, src_coord, tgt_coord, src_ts, tgt_ts):
        # print('pos3d out shape: ',self.pos3d_encoder(src_coord).shape)
        # print('token embed out shape: ',self.token_embedding(src).shape)
        src = self.token_embedding(src)
        src = src + self.pos3d_encoder(src_coord)
        tgt = self.token_embedding(tgt)
        tgt = src + self.pos3d_encoder(tgt_coord)
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        #print(f'src shape {src.shape}, tgt shape: {tgt.shape}')
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
            self.src_mask = mask
        # print('PE out shape: ',self.pos_encoder(src).shape)
        # print('PE out shape: ',self.pos_encoder(src).shape)
        # src = self.pos_encoder(src)
        # tgt = self.pos_encoder(tgt)
        #print(f'after pe src shape {src.shape}, tgt shape: {tgt.shape}')

        output_enc = self.transformer_encoder(src,self.src_mask) 
        #print(f'after enc src shape {src.shape}, tgt shape: {tgt.shape}')

        output_dec = self.transformer_decoder(tgt,output_enc,self.src_mask)
        #print(f'after dec src shape {src.shape}, tgt shape: {tgt.shape}')

        output = self.decoder(output_dec)

        #print('output shape: ',output.shape)
        return output.permute(1,0,2)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask