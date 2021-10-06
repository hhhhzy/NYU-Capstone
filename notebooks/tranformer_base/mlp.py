#!/bin/bash python

import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size, num_hidden_layers, dropout=None,
        ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout= dropout
        
        module_list = nn.ModuleList()
        module_list.append(nn.Linear(input_size,hidden_size))
        module_list.append(nn.ReLU(inplace=True))
        for _ in range(num_hidden_layers):
            module_list.extend([nn.Linear(hidden_size,hidden_size),nn.ReLU(inplace=True)])
            module_list.append(nn.Dropout(dropout))
            module_list.append(nn.LayerNorm(hidden_size))
        module_list.append(nn.Linear(hidden_size,output_size))
        
        self.sequential = nn.Sequential(*module_list)
        
    def forward(self, x):
        out = self.sequential(x)
        return out