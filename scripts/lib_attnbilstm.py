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

sys.path.append("../")
from _utils import EarlyStopping, seq2onehot, open_fa, open_exp, write_fa, write_exp

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

def plot_reg(label_real, label_pred, plot_path, c1="indianred"):
    
    def r(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return r_value 
    
    r_test = r(label_real, label_pred)
    testdata = pd.DataFrame({'pred': label_pred, 'label': label_real})

    sns.set_style("darkgrid")
    font = {'size' : 18}
    matplotlib.rc('font', **font)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)

    # c1 = "indianred"
    # c2 = 'mediumpurple'
    g = sns.JointGrid(x='label', y="pred", data=testdata, space=0, ratio=6, height=7)
    g.plot_joint(plt.scatter,s=20, color=c1, alpha=1, linewidth=0.4, edgecolor='black')
    f = g.fig
    ax = f.gca()
    
    ax.text(x=.71, y=0.03,s='r: ' + str(round(r_test, 2)), transform=ax.transAxes) # size
    g.plot_marginals(sns.kdeplot,fill=r_test, **{'linewidth':2, 'color':c1})
    g.set_axis_labels('Real Fold Change', 'Predicted', **{'size':18}) # **{'size':22}
    f = g.fig
    plt.savefig(plot_path)

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

class TorchStandardScaler:
    def __init__(self,mean=0,std=1):
        self.mean = mean
        self.std = std        
    def fit(self, x):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x
    def inv_transform(self, x):
        x = x * (self.std + 1e-7)
        x += self.mean
        return x
    def getvalue(self):
        return (self.mean, self.std)
    

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        input = input.permute(0,2,1)
        
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        
        output = output.permute(0,2,1)
        return output


class AttnBilstm(nn.Module):
    def __init__(self, motif_conv_hidden = 256, conv_hidden = 128, n_heads = 8, conv_width_motif = 30, dropout_rate = 0.2,
                 seq_len = 50):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=motif_conv_hidden, kernel_size=conv_width_motif, padding='same')
        self.norm1 = nn.BatchNorm1d(motif_conv_hidden)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=motif_conv_hidden, out_channels=conv_hidden, kernel_size=conv_width_motif, padding='same')
        self.norm2 = nn.BatchNorm1d(conv_hidden)
        self.relu2 = nn.ReLU()

        # (batch_size, hidden_dim, length) --> (length, batch_size, hidden_dim)
        self.attention1 = nn.TransformerEncoderLayer(d_model=conv_hidden, nhead=n_heads, batch_first=True)
        self.attention2 = nn.TransformerEncoderLayer(d_model=conv_hidden, nhead=n_heads, batch_first=True)
        
        self.bilstm = BidirectionalLSTM(input_size=conv_hidden, hidden_size=conv_hidden, output_size=int(conv_hidden//4)) # torch.Size([10, 32, 110])
        self.flatten1 = nn.Flatten() # torch.Size([10, 3520])
        self.dense1 = nn.Linear( int(conv_hidden//4) * seq_len, conv_hidden)
        self.relu3 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear( conv_hidden, conv_hidden)
        self.relu4 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.dense3 = nn.Linear(conv_hidden, 1)
        
        
    def forward(self,x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        
        x = x.permute(2,0,1)
        x = self.attention1(x)
        x = self.attention2(x)
        x = x.permute(1,2,0)
        
        x = self.bilstm(x)
        x = self.relu3(self.dense1(self.flatten1(x)))
        x = self.relu4(self.dense2(self.drop1(x)))
        x = self.dense3(self.drop2(x))
        return x


class AttnBilstm_language:
    def __init__(self, 
                 length,
                 batch_size = 64,
                 model_name = "attnbilstm",
                 epoch = 200,
                 patience = 50,
                 log_steps = 10,
                 save_steps = 20,
                 exp_mode = "direct"
                 ):
      
        self.model = AttnBilstm(seq_len=length)
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

# if __name__ == "__main__":
    
    ## part1: seqs, expr
    # df_induced = pd.read_csv("/home/qxdu/gpro/gpro/phase_seperation/mpra_0530/7_exp_mean_D_induced.tsv", sep="\t", header=None)
    # df_uninduced = pd.read_csv("/home/qxdu/gpro/gpro/phase_seperation/mpra_0530/7_exp_mean_D_uninduced.tsv", sep="\t", header=None)
    # df_induced.columns = ["seqs", "induced_mpra_means", "names"]
    # df_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]
    
    # df_merged = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
    # df_merged = df_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
    # df_merged["lfc"] = df_merged["induced_mpra_means"] / df_merged["uninduced_mpra_means"]
    
    # seqs = df_merged["seqs"]
    # fold_change = df_merged["lfc"]
    # tags = df_merged["names"]
    # write_fa("./merged_seqs.txt", seqs, tags)
    # write_exp("./merged_expr.txt", fold_change)
    
    # mutant_seqs = [seq for seq, tag in zip(seqs, tags) if tag.startswith("mutant")]
    # mutant_fold_change = [fc for fc, tag in zip(fold_change, tags) if tag.startswith("mutant")]
    # mutant_tags = [tag for tag in tags if tag.startswith("mutant")]
    # write_fa("./mutant_seqs.txt", mutant_seqs, mutant_tags)
    # write_exp("./mutant_expr.txt", mutant_fold_change)
    
    ## part2: train models
    # model = AttnBilstm_language(length=252, epoch=400, patience=50, model_name="basic")
    # dataset = './merged_seqs.txt'
    # labels  = './merged_expr.txt'
    # save_path = './checkpoints/'
    # model.train(dataset=dataset,labels=labels,savepath=save_path)
    
    ## part3: evaluation
    
    # model = AttnBilstm_language(model_name = "basic", length=252)
    # model_path = "/home/qxdu/gpro/gpro/phase_seperation/mpra_0530/checkpoints/basic/checkpoint.pth"
    # seqs = open_fa("/home/qxdu/gpro/gpro/phase_seperation/mpra_0530/mutant_seqs.txt")
    # expr = open_exp("/home/qxdu/gpro/gpro/phase_seperation/mpra_0530/mutant_expr.txt", operator="direct")
    # pred = model.predict_input(model_path=model_path, inputs=seqs, mode="data")
    # plot_reg(expr, pred, "./results/basic_regplot.png",c1="mediumpurple")