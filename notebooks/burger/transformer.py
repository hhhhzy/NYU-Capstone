#!/bin/bash python
import torch 
import torch.nn as nn
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

class Tranformer(nn.Module):
    def __init__(self,feature_size=250,num_enc_layers=1,num_dec_layers=1,d_ff = 256, dropout=0.1,num_head=2):
        super(Tranformer, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
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

    def forward(self,src, tgt):
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        #print(f'src shape {src.shape}, tgt shape: {tgt.shape}')
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
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