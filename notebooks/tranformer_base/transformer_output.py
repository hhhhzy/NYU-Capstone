#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from _arfima import arfima
from transformer import Tranformer
from utils import *


def evaluate(model,data_loader,criterion,batch_size):
    model.eval()
    total_loss = 0.
    rmse = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for (data,targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            total_loss += criterion(output, targets).detach().cpu().numpy()
    return total_loss
    


if __name__ == "__main__":
    best_config = {'feature_size': 128, 'num_layer': 2, 'num_head': 2, 'dropout': 0.1}
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2
    feature_size = best_config['feature_size']
    num_layer = best_config['num_layer']
    num_head = best_config['num_head']
    dropout = best_config['dropout']
    lr = 0.01#config['lr']
    window_size = 12#config['window_size']
    batch_size = 16#config['batch_size']
    
    model = Tranformer(feature_size=feature_size,num_layers=num_layer,dropout=dropout,num_head=num_head)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    print('Using device: ',device)
    model.to(device)
            
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    writer = tensorboard.SummaryWriter('/scratch/yd1008/tensorboard_output/')
    
    # if checkpoint_dir:
    #     checkpoint = os.path.join(checkpoint_dir, "checkpoint")
    #     model_state, optimizer_state = torch.load(checkpoint)
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)
        
    train_loader, test_loader = get_data_loaders(train_proportion, test_proportion, val_proportion,\
         window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 2, pin_memory = False, test_mode = True)

    epochs = 150
    train_losses = []
    test_losses = []
    tolerance = 10
    best_test_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.

        for (data, targets) in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets).detach().cpu().numpy()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_losses.append(total_loss/len(train_loader.dataset))
        test_losses.append(total_loss/len(test_loader.dataset))
        test_loss = evaluate(model, test_loader, criterion)
        if test_loss <= best_test_loss:
            pass ##Implement early stopping
        if epoch==1: ###DEBUG
            print(f'Total of {len(train_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set')
        print(f'Epoch: {epoch}, train_loss: {total_loss/len(train_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}')
        writer.add_scalar('train_loss',total_loss,epoch)
        writer.add_scalar('val_loss',test_loss,epoch)
        scheduler.step() 

    xs = np.arange(len(train_losses))
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,train_losses)
    fig.savefig('/scratch/yd1008/nyu-capstone/tune_results/figs/transformer_train_loss.png')
    plt.close(fig)
    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(xs,test_losses)
    fig.savefig('/scratch/yd1008/nyu-capstone/tune_results/figs/transformer_test_loss.png')
    plt.close(fig)
### Predict
    model.eval()
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for (data,targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            test_result = torch.cat((test_result, output[-1].view(-1).detach().cpu()), 0)
            truth = torch.cat((truth, targets[-1].view(-1).detach().cpu()), 0)

    fig, ax = plt.subplots(nrows =1, ncols=1, figsize=(20,10))
    ax.plot(test_result,color="red")
    ax.plot(truth,color="blue")
    ax.plot(test_result-truth,color="green")
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    fig.savefig('/scratch/yd1008/nyu-capstone/tune_results/figs/transformer_pred.png')
    plt.close(fig)