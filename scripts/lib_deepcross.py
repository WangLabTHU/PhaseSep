import os, sys
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from scipy import stats
from scipy.stats import pearsonr
from collections import OrderedDict

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

sys.path.append("../")
from _utils import EarlyStopping, seq2onehot, open_fa, open_exp, write_fa, write_exp


''' --DeepCROSS utils-- '''


class DcrossBottleneckLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(DcrossBottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding="same")
    
    def forward(self,x):
        x = self.conv1( self.relu1( self.bn1(x) ) )
        x = self.conv2( self.relu2( self.bn2(x) ) )
        return x

class DcrossDenseBlock(nn.Module):
    def __init__(self, nb_layers=6, hidden_dim=128):
        super(DcrossDenseBlock, self).__init__()
        
        self.nb_layers = nb_layers
        self.hidden_dim = hidden_dim
        
        self.bn = nn.ModuleList([DcrossBottleneckLayer(hidden_dim + 32 * i) for i in range(nb_layers)])

    def forward(self, x):
        layers_concat = []
        layers_concat.append(x)
        for i in range(self.nb_layers):
            x = torch.cat(layers_concat, dim=1)
            x = self.bn[i](x)
            layers_concat.append(x)
        x = torch.cat(layers_concat, dim=1)
        return x

class DcrossTransitionLayer(nn.Module):
    def __init__(self, input_dim):
        super(DcrossTransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=int(input_dim*0.5), kernel_size=1)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.relu1( self.bn1(x) )
        x = self.pool1( self.conv1(x) )
        return x
    
''' -- DeepCROSS Model-- '''



class SequenceData(Dataset):
  def __init__(self,data, label):
    self.data = data
    self.target = label
  
  def __getitem__(self, index):
    return self.data[index], self.target[index]
    
  def __len__(self):
    return self.data.size(0)
  
  def __getdata__(self):
    return self.data, self.target

class TestData(Dataset):
    def __init__(self,data):
        self.data = data
  
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return self.data.size(0)
    
    def __getdata__(self):
        return self.data




class DeepCROSS(nn.Module):
    def __init__( self, seq_len=165, conv_hidden=64, nb_layers=6, n_tokens = 4):
        super(DeepCROSS, self).__init__()
        
        self.n_tokens = n_tokens
        self.seq_len = seq_len
        
        self.conv1 = nn.Conv1d(in_channels=n_tokens, out_channels=conv_hidden, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(conv_hidden)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.lstm1 = nn.LSTM(input_size=conv_hidden, hidden_size=conv_hidden, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=conv_hidden*2, hidden_size=conv_hidden, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=conv_hidden*2, hidden_size=conv_hidden, bidirectional=True)
        
        self.latent_dim = conv_hidden * 2
        self.seq_length = int(seq_len / 2)
        
        self.denseblock1 = DcrossDenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.transblock1 = DcrossTransitionLayer(conv_hidden*2 + 32*nb_layers)
        self.latent_dim = int(self.latent_dim / 2)
        self.seq_length = int(self.seq_length / 2)
        
        
        self.denseblock2 = DcrossDenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.transblock2 = DcrossTransitionLayer(self.latent_dim)
        self.latent_dim = int(self.latent_dim / 2)
        self.seq_length = int(self.seq_length / 2)
        
        self.denseblock3 = DcrossDenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.transblock3 = DcrossTransitionLayer(self.latent_dim)
        self.latent_dim = int(self.latent_dim / 2)
        self.seq_length = int(self.seq_length / 2)
        
        self.denseblock4 = DcrossDenseBlock(nb_layers, self.latent_dim)
        self.latent_dim = self.latent_dim + 32*nb_layers
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.seq_length = int(self.seq_length / 2)
        
        self.linear = nn.Linear( in_features=self.latent_dim * self.seq_length, out_features=1 )
        
    def forward(self,x):
        x = self.bn1( self.conv1(x) )
        x = self.pool1( self.relu1(x) ) # [batch_size, conv_hidden, 83]
        
        x = x.permute(2,0,1)
        x,_ = self.lstm1(x) # [83, batch_size, conv_hidden*2]
        x,_ = self.lstm2(x) # [83, batch_size, conv_hidden*2]
        x,_ = self.lstm3(x) # [83, batch_size, conv_hidden*2]
        x = x.permute(1,2,0)# [batch_size, conv_hidden*2, 83], length = np.floor((165 + 2 * 1 - 1 - 2)/2 + 1)
        
        x = self.denseblock1(x) # [batch_size, conv_hidden*2 + 32*6, 83]
        x = self.transblock1(x) # [batch_size, (conv_hidden*2 + 32*6)/2, (83/2)]       
        x = self.denseblock2(x)
        x = self.transblock2(x)
        x = self.denseblock3(x)
        x = self.transblock3(x)
        x = self.denseblock4(x) # [batch_size, latent_dim, seq_length] = [batch_size, 376, 10]
        
        x = self.avgpool1(x)
        x = self.flatten(x) # [batch_size, 376 x 5=1880]
        x = self.linear(x) # [batch_size, 1]
        return x
    


class DeepCROSS_language:
    def __init__(self, 
                 length,
                 batch_size = 64,
                 model_name = "deepcross",
                 epoch = 200,
                 patience = 50,
                 log_steps = 10,
                 save_steps = 20,
                 exp_mode = "direct"
                 ):
      
        self.model = DeepCROSS(seq_len=length)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch = epoch
        self.patience = patience
        self.seq_len = length
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"), ]
        self.exp_mode = exp_mode
    
    def train(self, dataset, labels, savepath):
      
        self.dataset = dataset
        self.labels = labels
        self.checkpoint_root = savepath
      
        filename_sim = self.checkpoint_root + self.model_name
        
        if not os.path.exists(filename_sim):
            os.makedirs(filename_sim)
        
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, 
                                       path=os.path.join(filename_sim, 'checkpoint.pth'), stop_order='max')
        
        total_feature = open_fa(self.dataset)
        total_feature = seq2onehot(total_feature, self.seq_len)
        total_label = open_exp(self.labels, operator=self.exp_mode)
        total_feature = torch.tensor(total_feature, dtype=float) # (sample num,length,4)
        total_label = torch.tensor(total_label, dtype=float) # (sample num)
        
        # Shuffle data with random seed 0
        rng = np.random.default_rng(seed=0)
        indices = np.arange(len(total_label))
        rng.shuffle(indices)
        total_feature = total_feature[indices]
        total_label = total_label[indices]
        
        total_length = int(total_feature.shape[0])
        r = int(total_length*0.7)
        train_feature = total_feature[0:r,:,:]
        train_label = total_label[0:r]
        valid_feature = total_feature[r:total_length,:,:]
        valid_label = total_label[r:total_length]
        
        train_dataset = SequenceData(train_feature, train_label)
        train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size, shuffle=True)
        valid_dataset = SequenceData(valid_feature, valid_label)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=self.batch_size, shuffle=True)
        
        
        train_log_filename = os.path.join(filename_sim, "train_log.txt")
        train_model_filename = os.path.join(filename_sim, "checkpoint.pth")
        print("results saved in: ", filename_sim)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),] 
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.HuberLoss(reduction='mean')
        
        for epoch in tqdm(range(0,self.epoch)):
            model.train()
            train_epoch_loss = []
            for idx,(feature,label) in enumerate(train_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                outputs = model(feature)
                optimizer.zero_grad()
                loss = criterion(label.float(),outputs.flatten())
                loss.backward()
                optimizer.step()
                train_epoch_loss.append(loss.item())

            model.eval()
            valid_exp_real = []
            valid_exp_pred = []
            for idx,(feature,label) in enumerate(valid_dataloader,0):
                feature = feature.to(torch.float32).to(device).permute(0,2,1)
                label = label.to(torch.float32).to(device)
                outputs = model(feature)
                valid_exp_real += label.float().tolist()
                valid_exp_pred += outputs.flatten().tolist()
            coefs = np.corrcoef(valid_exp_real,valid_exp_pred)
            coefs = coefs[0, 1]
            test_coefs = coefs
            
            print("real expression samples: ", valid_exp_real[0:5])
            print("pred expression samples: ", valid_exp_pred[0:5])
            print("current coeffs: ", test_coefs)
            cor_pearsonr = pearsonr(valid_exp_real, valid_exp_pred)
            print("current pearsons: ",cor_pearsonr)
            
            ## Early Stopping Step
            early_stopping(val_loss=test_coefs, model=self.model)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break
            
            if (epoch%self.log_steps == 0):
                to_write = "epoch={}, loss={}\n".format(epoch, np.average(train_epoch_loss))
                with open(train_log_filename, "a") as f:
                    f.write(to_write)
            if (epoch%self.save_steps == 0):
                torch.save(model.state_dict(), train_model_filename)
    
    def predict(self, model_path, data_path):
        
        model_path = os.path.dirname(model_path)
        path_check = '{}/checkpoint.pth'.format(model_path)
        path_seq_save =  '{}/seqs.txt'.format(model_path)
        path_pred_save = '{}/preds.txt'.format(model_path)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
        model = self.model.to(device)
        model.load_state_dict(torch.load(path_check))
        model.eval()
        seq_len = self.seq_len
        
        test_feature = open_fa(data_path)
        test_seqs = test_feature
        
        test_feature = seq2onehot(test_feature, seq_len)
        test_feature = torch.tensor(test_feature, dtype=float)
        test_dataset = TestData(test_feature)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)
        
        test_exp_pred = []
        for idx,feature in enumerate(test_dataloader,0):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)
            outputs = model(feature)
            pred = outputs.flatten().tolist()
            test_exp_pred += pred
        
        ## Saving Seqs
        f = open(path_seq_save,'w')
        i = 0
        while i < len(test_seqs):
            f.write('>' + str(i) + '\n')
            f.write(test_seqs[i] + '\n')
            i = i + 1
        f.close()
        
        ## Saving pred exps
        f = open(path_pred_save,'w')
        i = 0
        while i < len(test_exp_pred):
            f.write(str(np.round(test_exp_pred[i],2)) + '\n')
            i = i + 1
        f.close()

    def predict_input(self, model_path, inputs, mode="path"):
        
        model_path = os.path.dirname(model_path)
        path_check = '{}/checkpoint.pth'.format(model_path)
        
        device, = [torch.device("cuda" if torch.cuda.is_available() else "cpu"),]
        model = self.model.to(device)
        model.load_state_dict(torch.load(path_check))
        model.eval()
        seq_len = self.seq_len
        
        if mode=="path":
            test_feature = open_fa(inputs)
            test_feature = seq2onehot(test_feature, seq_len)
        elif mode=="data":
            test_feature = seq2onehot(inputs, seq_len)
        elif mode=="onehot":
            test_feature = inputs
        test_feature = torch.tensor(test_feature, dtype=float)
        test_dataset = TestData(test_feature)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size = 128, shuffle=False)
        
        exp = []
        for idx, feature in tqdm(enumerate(test_dataloader,0)):
            feature = feature.to(torch.float32).to(device).permute(0,2,1)
            outputs = model(feature)
            pred = outputs.flatten().tolist()
            exp += pred
        return exp



