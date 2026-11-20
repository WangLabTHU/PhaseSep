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


''' --Malinois utils-- '''

import math
from collections import OrderedDict

class Conv1dNorm(nn.Module):
    """
    Convolutional layer with optional normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.
        dilation (int): Dilation rate of the convolution.
        groups (int): Number of groups for grouped convolution.
        bias (bool): Whether to include bias terms.
        batch_norm (bool): Whether to use batch normalization.
        weight_norm (bool): Whether to use weight normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, batch_norm=True, weight_norm=True):
        """
        Initialize Conv1dNorm layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding for the convolution.
            dilation (int): Dilation rate of the convolution.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether to include bias terms.
            batch_norm (bool): Whether to use batch normalization.
            weight_norm (bool): Whether to use weight normalization.
        """
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride, padding, dilation, groups, bias)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, 
                                           affine=True, track_running_stats=True)
    def forward(self, input):
        """
        Forward pass through the convolutional layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        try:
            return self.bn_layer( self.conv( input ) )
        except AttributeError:
            return self.conv( input )
        
class LinearNorm(nn.Module):
    """
    Linear layer with optional normalization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include bias terms.
        batch_norm (bool): Whether to use batch normalization.
        weight_norm (bool): Whether to use weight normalization.
    """
    def __init__(self, in_features, out_features, bias=True, 
                 batch_norm=True, weight_norm=True):
        """
        Initialize LinearNorm layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include bias terms.
            batch_norm (bool): Whether to use batch normalization.
            weight_norm (bool): Whether to use weight normalization.
        """
        super(LinearNorm, self).__init__()
        self.linear  = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, 
                                           affine=True, track_running_stats=True)
    def forward(self, input):
        """
        Forward pass through the linear layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        try:
            return self.bn_layer( self.linear( input ) )
        except AttributeError:
            return self.linear( input )

class GroupedLinear(nn.Module):
    """
    A custom linear transformation module that groups input and output features.

    Args:
        in_group_size (int): Number of input features in each group.
        out_group_size (int): Number of output features in each group.
        groups (int): Number of groups.

    Attributes:
        in_group_size (int): Number of input features in each group.
        out_group_size (int): Number of output features in each group.
        groups (int): Number of groups.
        weight (Parameter): Learnable weight parameter for the linear transformation.
        bias (Parameter): Learnable bias parameter for the linear transformation.

    Methods:
        reset_parameters(weights, bias):
            Initialize the weight and bias parameters with kaiming uniform initialization.
        forward(x):
            Apply the grouped linear transformation to the input tensor.

    Example:
        linear_layer = GroupedLinear(in_group_size=10, out_group_size=5, groups=2)
        output = linear_layer(input_tensor)
    """
    
    def __init__(self, in_group_size, out_group_size, groups):
        """
        Initialize the GroupedLinear module.

        Args:
            in_group_size (int): Number of input features in each group.
            out_group_size (int): Number of output features in each group.
            groups (int): Number of groups.

        Returns:
            None
        """
        super().__init__()
        
        self.in_group_size = in_group_size
        self.out_group_size= out_group_size
        self.groups        = groups
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias   = torch.nn.Parameter(torch.zeros(groups, 1, out_group_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        """
        Initialize the weight and bias parameters with kaiming uniform initialization.

        Args:
            weights (Tensor): The weight parameter tensor.
            bias (Tensor): The bias parameter tensor.

        Returns:
            None
        """
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        """
        Apply the grouped linear transformation to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed output tensor.
        """
        reorg = x.permute(1,0).reshape(self.groups, self.in_group_size, -1).permute(0,2,1)
        hook  = torch.bmm(reorg, self.weight) + self.bias
        reorg = hook.permute(0,2,1).reshape(self.out_group_size*self.groups,-1).permute(1,0)
        
        return reorg

class RepeatLayer(nn.Module):
    """
    A custom module to repeat the input tensor along specified dimensions.

    Args:
        *args (int): Size of repetitions along each specified dimension.

    Attributes:
        args (tuple): Sizes of repetitions along each specified dimension.

    Methods:
        forward(x):
            Repeat the input tensor along the specified dimensions.

    Example:
        repeat_layer = RepeatLayer(2, 3)
        output = repeat_layer(input_tensor)
    """
    
    def __init__(self, *args):
        """
        Initialize the RepeatLayer module.

        Args:
            *args (int): Size of repetitions along each specified dimension.

        Returns:
            None
        """
        super().__init__()
        self.args = args
        
    def forward(self, x):
        """
        Repeat the input tensor along the specified dimensions.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The repeated output tensor.
        """
        return x.repeat(*self.args)
    
class BranchedLinear(nn.Module):
    """
    A custom module that implements a branched linear architecture.

    Args:
        in_features (int): Number of input features.
        hidden_group_size (int): Number of hidden features in each group.
        out_group_size (int): Number of output features in each group.
        n_branches (int): Number of branches.
        n_layers (int): Number of layers in each branch.
        activation (str): Activation function to use in the hidden layers.
        dropout_p (float): Dropout probability applied to hidden layers.

    Attributes:
        in_features (int): Number of input features.
        hidden_group_size (int): Number of hidden features in each group.
        out_group_size (int): Number of output features in each group.
        n_branches (int): Number of branches.
        n_layers (int): Number of layers in each branch.
        branches (OrderedDict): Dictionary to store branch layers.
        nonlin (nn.Module): Activation function module.
        dropout (nn.Dropout): Dropout layer module.
        intake (RepeatLayer): A layer to repeat input along branches.

    Methods:
        forward(x):
            Perform forward pass through the branched linear architecture.

    Example:
        branched_linear = BranchedLinear(in_features=256, hidden_group_size=128,
                                         out_group_size=64, n_branches=4,
                                         n_layers=3, activation='ReLU', dropout_p=0.5)
        output = branched_linear(input_tensor)
    """
    
    def __init__(self, in_features, hidden_group_size, out_group_size, 
                 n_branches=1, n_layers=1, 
                 activation='ReLU', dropout_p=0.5):
        """
        Initialize the BranchedLinear module.

        Args:
            in_features (int): Number of input features.
            hidden_group_size (int): Number of hidden features in each group.
            out_group_size (int): Number of output features in each group.
            n_branches (int): Number of branches.
            n_layers (int): Number of layers in each branch.
            activation (str): Activation function to use in the hidden layers.
            dropout_p (float): Dropout probability applied to hidden layers.

        Returns:
            None
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers   = n_layers
        
        self.branches = OrderedDict()
        
        self.nonlin  = getattr(nn, activation)()                               
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features
        
        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, out_group_size, n_branches))
            else:
                setattr(self, f'branched_layer_{i+1}',  GroupedLinear(cur_size, hidden_group_size, n_branches))
            cur_size = hidden_group_size
            
    def forward(self, x):
        """
        Perform forward pass through the branched linear architecture.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        hook = self.intake(x)
        
        i = -1
        for i in range(self.n_layers-1):
            hook = getattr(self, f'branched_layer_{i+1}')(hook)
            hook = self.dropout( self.nonlin(hook) )
        hook = getattr(self, f'branched_layer_{i+2}')(hook)
            
        return hook


''' -- Malinois Model-- '''


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



def get_padding(kernel_size):
    left = (kernel_size - 1) // 2
    right= kernel_size - 1 - left
    return [ max(0,x) for x in [left,right] ]

class Malinois(nn.Module):
    
    def __init__(self, seq_len = 600, n_tokens = 4,
                 conv1_channels=300, conv1_kernel_size=19, 
                 conv2_channels=200, conv2_kernel_size=11, 
                 conv3_channels=200, conv3_kernel_size=7, 
                 n_linear_layers=1, linear_channels=1000, 
                 linear_activation='ReLU', linear_dropout_p=0.1163, 
                 n_branched_layers=3, branched_channels=140, 
                 branched_activation='ReLU', branched_dropout_p=0.5757, 
                 n_outputs = 1, # 3
                 use_batch_norm=True, use_weight_norm=False):
                                         
        super().__init__()        
        
        self.seq_len = seq_len
        self.n_tokens = n_tokens
        
        self.conv1_channels    = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)
        
        self.conv2_channels    = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        self.conv3_channels    = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)
        
        self.n_linear_layers   = n_linear_layers
        self.linear_channels   = linear_channels
        self.linear_activation = linear_activation
        self.linear_dropout_p  = linear_dropout_p
        
        self.n_branched_layers = n_branched_layers
        self.branched_channels = branched_channels
        self.branched_activation = branched_activation
        self.branched_dropout_p= branched_dropout_p
        
        self.n_outputs         = n_outputs
        self.use_batch_norm    = use_batch_norm
        self.use_weight_norm   = use_weight_norm
        
        self.pad1  = nn.ConstantPad1d(self.conv1_pad, 0.)
        self.conv1 = Conv1dNorm(self.n_tokens, 
                                self.conv1_channels, self.conv1_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad2  = nn.ConstantPad1d(self.conv2_pad, 0.)
        self.conv2 = Conv1dNorm(self.conv1_channels, 
                                self.conv2_channels, self.conv2_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        self.pad3  = nn.ConstantPad1d(self.conv3_pad, 0.)
        self.conv3 = Conv1dNorm(self.conv2_channels, 
                                self.conv3_channels, self.conv3_kernel_size, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                bias=True, 
                                batch_norm=self.use_batch_norm, 
                                weight_norm=self.use_weight_norm)
        
        self.pad4 = nn.ConstantPad1d((1,1), 0.)
        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)
        
        next_in_channels = self.conv3_channels * self.get_flatten_factor(self.seq_len)
        
        for i in range(self.n_linear_layers):
            
            setattr(self, f'linear{i+1}', 
                    LinearNorm(next_in_channels, self.linear_channels, 
                               bias=True, 
                               batch_norm=self.use_batch_norm, 
                               weight_norm=self.use_weight_norm)
                   )
            next_in_channels = self.linear_channels

        self.branched = BranchedLinear(next_in_channels, self.branched_channels, 
                                       self.branched_channels, 
                                       self.n_outputs, self.n_branched_layers, 
                                       self.branched_activation, self.branched_dropout_p)
            
        self.output  = GroupedLinear(self.branched_channels, 1, self.n_outputs)
        
        self.nonlin  = getattr(nn, self.linear_activation)()                               
        
        self.dropout = nn.Dropout(p=self.linear_dropout_p)
    
    def get_flatten_factor(self, input_len):
        hook = input_len
        hook = hook // 3
        hook = hook // 4
        
        return (hook + 2) // 4
    
    def encode(self, x):
        hook = self.nonlin( self.conv1( self.pad1( x ) ) )
        hook = self.maxpool_3( hook )
        hook = self.nonlin( self.conv2( self.pad2( hook ) ) )
        hook = self.maxpool_4( hook )
        hook = self.nonlin( self.conv3( self.pad3( hook ) ) )
        hook = self.maxpool_4( self.pad4( hook ) )        
        hook = torch.flatten( hook, start_dim=1 )
        return hook
    
    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout( 
                self.nonlin( 
                    getattr(self,f'linear{i+1}')(hook)
                )
            )
        hook = self.branched(hook)
        return hook
    
    def classify(self, x):
        output = self.output( x )
        return output
        
    def forward(self, x):
        
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output  = self.classify(decoded)
        return output
    


class Malinois_language:
    def __init__(self, 
                 length,
                 batch_size = 64,
                 model_name = "malinois",
                 epoch = 200,
                 patience = 50,
                 log_steps = 10,
                 save_steps = 20,
                 exp_mode = "direct"
                 ):
      
        self.model = Malinois(seq_len=length)
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