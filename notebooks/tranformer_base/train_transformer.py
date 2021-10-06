#!/bin/bash python
import numpy as np
import torch 
import torch.nn as nn
from torch.utils import tensorboard
import torch.optim as optim
import math 
import time
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from _arfima import arfima


def to_windowed(data,window_size,pred_size):
    out = []
    for i in range(len(data)-window_size):
        feature = np.array(data[i:i+(window_size)])
        target = np.array(data[i+pred_size:i+window_size+pred_size])
        out.append((feature,target))
        

    return np.array(out)#, np.array(targets)

def train_test_val_split(x_vals ,train_proportion = 0.6, val_proportion = 0.2, test_proportion = 0.2\
              , window_size = 12, pred_size = 1):

    total_len = len(x_vals)
    train_len = int(total_len*train_proportion)
    val_len = int(total_len*val_proportion)
    test_len = int(total_len*test_proportion)
    ### Add a scaler here on x_vals
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_vals = scaler.fit_transform(x_vals.reshape(-1, 1)).reshape(-1)

    train_data = x_vals[0:train_len]
    val_data = x_vals[train_len:(train_len+val_len)]
    test_data = x_vals[(train_len+val_len):]

    train = to_windowed(train_data,window_size,pred_size)
    val = to_windowed(val_data,window_size,pred_size)
    test = to_windowed(test_data,window_size,pred_size)

    train = torch.from_numpy(train).float()
    val = torch.from_numpy(val).float()
    test = torch.from_numpy(test).float()

    return train,val,test,train_data,val_data,test_data, scaler

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,x):
        self.x=x
 
    def __len__(self):
        return len(self.x)
 
    def __getitem__(self,idx):
        return(self.x[idx][0].view(-1,1),self.x[idx][1].view(-1,1))
    
def get_data_loaders(train_proportion = 0.6, test_proportion = 0.2, val_proportion = 0.2,window_size = 10, pred_size =1, batch_size = 10, num_workers = 1, pin_memory = True): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(505)
    long_range_stationary_x_vals = arfima([0.5,0.4],0.3,[0.2,0.1],10000,warmup=2^10)
    train_data,val_data,test_data,train_original,val_original,test_original, scaler = train_test_val_split(\
        long_range_stationary_x_vals ,train_proportion = train_proportion\
        , val_proportion = val_proportion, test_proportion = test_proportion\
        , window_size = window_size, pred_size = pred_size)
    
    dataset_train, dataset_test, dataset_val = CustomDataset(train_data), CustomDataset(test_data), CustomDataset(val_data)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                        drop_last=False, 
                                        num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                        drop_last=False, 
                                        num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, 
                                        drop_last=False, 
                                        num_workers=num_workers, pin_memory=pin_memory)
    return train_loader,val_loader, test_loader

#----------------#
#  Define Model  #
#----------------#

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
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1,num_head=2):
        super(Tranformer, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
def evaluate(model,data_loader,criterion):
    model.eval()
    total_loss = 0.
    rmse = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    for (data,targets) in data_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        total_loss += criterion(output, targets).detach().cpu().numpy()
    return total_loss

def train_ts(config):
    lr = config['lr'] 
    window_size = config['window_size']
    batch_size = config['batch_size']
    train_loader,val_loader, test_loader = get_data_loaders(window_size=window_size, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Initialize model
    model = Tranformer(feature_size=250,num_layers=1,dropout=0.1,num_head=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95   )

    epochs = 10 


    writer = tensorboard.SummaryWriter('./test_logs')
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train() # Turn on the train mode \o/
        total_loss = 0.

        for (data, targets) in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            total_loss += loss.item()
            log_interval = int(len(train_loader) / batch_size / 5)
            
        writer.add_scalar('train_loss',total_loss,epoch)
        print(f'Config: {config}, Epoch: {epoch}')

        val_loss, rmse = evaluate(model, val_loader, criterion)
        writer.add_scalar('val_loss',val_loss,epoch)
        writer.add_scalar('rmse',rmse,epoch)
        tune.report(val_loss=val_loss, rmse=rmse)

        scheduler.step() 
        
if __name__ == "__main__":
    search_space = {
        "lr": tune.grid_search([0.1, 0.01]),
        "window_size": tune.grid_search([16, 64]),
        "batch_size": tune.grid_search([16, 64]),
    }

    ray.init(ignore_reinit_error=True)
    sched = AsyncHyperBandScheduler(mode='min')
    analysis = tune.run(train_ts, config=search_space, metric='rmse', scheduler=sched, resources_per_trial={"gpu": 1},local_dir="/scratch/yd1008/ray_results",)
    print('The best configs are: ',analysis.get_best_config(mode='min'))
    ray.shutdown()