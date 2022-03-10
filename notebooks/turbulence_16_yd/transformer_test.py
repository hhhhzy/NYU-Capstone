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

class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, activation='sin'):
        super(TemporalEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, self.input_dim-1))
        self.b = nn.parameter.Parameter(torch.randn(1, self.input_dim-1))
        if activation == 'sin':
            self.f = torch.sin
        elif activation == 'cos':
            self.f = torch.cos
        self.fc1 = nn.Linear(self.input_dim, self.output_dim)
    
    def l1(self, tau):
        f = self.f
        v1 = f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        #print(v1.shape)
        return torch.cat([v1, v2], 2)

    def forward(self, x):
        #x = x.unsqueeze(2)
        x = self.l1(x)
        x = self.fc1(x)
        return x

class multihead_attn(nn.Module):
    def __init__(self, embed_dim, num_head):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        assert self.embed_dim == self.head_dim*self.num_head, f'embedding dimension {self.embed_dim} is not divisible by num_head {self.num_head}'
        
        self.softmax = nn.Softmax(dim=-1)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        '''
        query, key, value: shape(batch_size, sequence_length, embedding_dim) #(B,S,E)
        '''
        B, S, E = query.shape
        query = self.query_proj(query).contiguous().view(B,S,self.num_head,self.head_dim).permute(0,2,1,3) #BxHxSxEh
        key = self.key_proj(key).contiguous().view(B,S,self.num_head,self.head_dim).permute(0,2,1,3) #BxHxSxEh
        value = self.value_proj(value).contiguous().view(B,S,self.num_head,self.head_dim).permute(0,2,1,3) #BxHxSxEh
        attn_out = self.attention(query, key, value, attn_mask) #BxHxSxEh
        attn_out = attn_out.transpose(1,2).contiguous().view(B, S, E) #BxHxSxE -> BxSxHxE
        
        return attn_out
    
    def attention(self, q, k, v, attn_mask, **kwargs):
        ### Not necessarily same as dims of input dimension if multihead_attn
        B, H, L, E = q.shape 
        _, H, K, _ = k.shape 
        
        attn = q/np.sqrt(E)@k.transpose(-1,-2) #BxHxLxE @ BxHxExK -> BxHxLxK

        if attn_mask is not None:
            if attn_mask.dim() == 2:  #broadcast 2D mask
                attn_mask = attn_mask.unsqueeze(0)
            attn = attn+attn_mask
            
        attn = self.softmax(attn) #BxHxLxK
        
        out = attn@v #BxHxLxK @ BxHxKxE -> BxHxLxE
        return out

class decoder_layer(nn.Module):
    def __init__(self, embed_dim, num_head, ff_dim, dropout):
        super().__init__()
        self.self_attention = multihead_attn(embed_dim, num_head)
        self.cross_attention = multihead_attn(embed_dim, num_head)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.activation = nn.ReLU()
        self.dropout_ff = nn.Dropout(dropout)
        self.ff2 = nn.Linear(ff_dim, embed_dim)

    def forward(self,tgt, mem, src_mask=None, tgt_mask=None):
        out = tgt #BxSxE
        out = self.norm1(out + self.dropout1(self.self_attention(out, out, out, tgt_mask))) #BxSxE
        out = self.norm2(out + self.dropout2(self.cross_attention(out, mem, mem, src_mask))) #BxSxE
        out = self.norm3(out + self.dropout3(self.ff_block(out))) #BxSxE REPLACE FF with conv3d?
        return out
    
    def ff_block(self,x):
        x = self.activation(self.ff1(x))
        x = self.ff2(self.dropout_ff(x))
        return x

class decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm
    def forward(self, tgt, mem, src_mask=None, tgt_mask=None):
        out = tgt
        for layer in self.decoder_layers:
            out = layer(out, mem, src_mask, tgt_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out

class Transformer(nn.Module):
    def __init__(self,feature_size=250,num_enc_layers=1,num_dec_layers=1,d_ff = 256, dropout=0.1,num_head=2,pe_type='3d',grid_size=16,mask_type=None,patch_size=(2,2,2),window_size=5,decoder_only=False):
        '''
        mask_type: 'patch' if using cuboic patches, which masks by patch instead of elements. Default to None (square_subsequent mask)
        '''
        super().__init__()
        self.patch_size = patch_size
        self.window_size = window_size
        self.decoder_only = decoder_only
        self.tgt_mask = mask_type
        self.pe_type = pe_type
        self.pos_encoder = PositionalEncoding(feature_size)
        self.pos3d_encoder = PositionalEmbedding3D(feature_size,grid_size)
        #self.token_embedding = TokenEmbedding(feature_size)
        self.temporal_encoder = TemporalEmbedding(input_dim=feature_size, output_dim=feature_size, activation='sin')

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, \
            nhead=num_head, dropout=dropout, dim_feedforward = d_ff)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_enc_layers)      
        self.decoder_layer = decoder_layer(embed_dim = feature_size, num_head = num_head, ff_dim = d_ff, dropout = dropout)  
        self.transformer_decoder = decoder(self.decoder_layer, num_layers=num_dec_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_coord, tgt_coord, src_ts, tgt_ts):
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
            # src = src.permute(1,0,2)
            # tgt = tgt.permute(1,0,2)
        
        elif self.pe_type == '3d_temporal':
            src = src + self.pos3d_encoder(src_coord) + self.temporal_encoder(src_ts)
            tgt = tgt + self.pos3d_encoder(tgt_coord) + self.temporal_encoder(tgt_ts)
            # src = src.permute(1,0,2)
            # tgt = tgt.permute(1,0,2)

        ### generate patch mask
        if self.tgt_mask == 'patch':
            device = src.device
            tgt_mask = self._generate_patch_mask(self.patch_size,self.window_size).to(device)
        elif self.tgt_mask is None or self.tgt_mask.size(0) != len(src):
            device = src.device
            tgt_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
            print(f'USING TRIANGULAR MASK!')
        # print('PE out shape: ',self.pos_encoder(src).shape)
        # print('PE out shape: ',self.pos_encoder(src).shape)
        # src = self.pos_encoder(src)
        # tgt = self.pos_encoder(tgt)
        #print(f'after pe src shape {src.shape}, tgt shape: {tgt.shape}')
        if self.decoder_only:
            #print('USING DECODER ONLY')
            #print(f'after pe src shape {src.shape}, tgt shape: {tgt.shape}', flush=True) #(sequence, batch_size, feature_size)
            output_dec = self.transformer_decoder(tgt,src,tgt_mask = tgt_mask)
        else:
            output_enc = self.transformer_encoder(src) 
            #print(f'encoder output shape: {output_enc.shape}', flush=True)
            output_dec = self.transformer_decoder(tgt,output_enc,tgt_mask = tgt_mask)
            #print(f'decoder output shape: {output_dec.shape}', flush=True)

        output = self.decoder(output_dec)

        #print('output shape: ',output.shape)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_patch_mask(self,patch_size,window_size):
        patch_prod = torch.prod(torch.tensor(patch_size))
        mask = (torch.triu(torch.ones(window_size, window_size)) == 1).transpose(0, 1)
        mask = torch.repeat_interleave(mask, patch_prod, dim=0)
        mask = torch.repeat_interleave(mask, patch_prod, dim=1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask