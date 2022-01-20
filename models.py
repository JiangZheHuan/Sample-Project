import os
import time
import math
import string
import random

import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.functional import F



class DNNModel(nn.Module):
    @classmethod
    def arg_list(cls):
        return ["input_size", "hidden_sizes", "n_classes", "dropout_p", "bi"]

    def __init__(self, input_size, hidden_sizes, n_classes, dropout_p=0.2, bi=False):
        """ """
        super(DNNModel, self).__init__()

        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes
        self.n_classes    = n_classes
        self.dropout_p    = dropout_p
        self.bi           = bi
        
        layer_sizes = [self.input_size] + self.hidden_sizes
        for n in range(1, len(layer_sizes)):
            setattr(self, f"fc_{n}", nn.Linear(
                in_features =layer_sizes[n-1],
                out_features=layer_sizes[n],))
        
        self.dropout = nn.Dropout(self.dropout_p)
        
        if self.bi:
            self.bi_classifier = nn.Linear(
                in_features=self.hidden_sizes[-1],
                out_features=1)
                
            self.multi_classifier = nn.Linear(
                in_features =self.hidden_sizes[-1],
                out_features=self.n_classes-1)
            
        else:
            self.classifier = nn.Linear(
                in_features =self.hidden_sizes[-1],
                out_features=self.n_classes)
        
    def forward(self, x, *, verbose=False):
        """ """
        n_b, n_s = x.shape
        if verbose:
            print("*"*10, "INPUT", "*"*10)
            print(x.shape)
            
        yhat = F.relu(self.fc_1(x))
        for n in range(2, len(self.hidden_sizes)+1):
            yhat = F.relu(getattr(self, f"fc_{n}")(yhat))
            
        if verbose:
            print("\n")
            print("*"*10, "Y_HAT", "*"*10)
            print(yhat.shape)
        
        yhat = self.dropout(yhat)
        out = self.classifier(yhat) if not self.bi else\
            [self.bi_classifier(yhat), self.multi_classifier(yhat)]

        if verbose:
            print("\n")
            print("*"*10, "OUTPUT", "*"*10)
            print(out.shape if not self.bi else [t.shape for t in out])
        return out


class CluModel():
    req_columns = [
            'hrank', 'arank', 'vpoints3h', 'vpoints3a', 'vpointspos3h', 'vpointspos3a', 'vpointsneg3h', 'vpointsneg3a',
            'instabilityscorehh', 'instabilityscoreaa', 'agspan3', 'hgspan3', 'points3aa', 'points3hh',
            'outcome1_label'
        ]
    
    def arg_list(cls):
        return self.params

    def __init__(self, n_comp, **kwargs):
        """ """
        super(CluModel, self).__init__()

        self.n_comp = n_comp
        self.params = {"n_comp": n_comp, **kwargs}
        
        self.gmm = GaussianMixture(n_components=self.n_comp)
        
    def __call__(self, df_x, *, verbose=False):
        """ """
        x = df_x[self.req_columns]
        n_b, n_s = x.shape
        if verbose:
            print("*"*10, "INPUT", "*"*10)
            print(x.shape)

        return self.gmm.fit(x)


class RocModel():
    req_columns = [
            'hrank', 'arank', 'vpoints3h', 'vpoints3a', 'vpointspos3h', 'vpointspos3a', 'vpointsneg3h', 'vpointsneg3a',
            'instabilityscorehh', 'instabilityscoreaa', 'agspan3', 'hgspan3', 'points3aa', 'points3hh'
        ]
    
    def arg_list(cls):
        return self.params

    def __init__(self, solver, **kwargs):
        """ """
        super(RocModel, self).__init__()

        self.solver = solver or 'lbfgs'
        self.params = {"n_comp": n_comp, **kwargs}
        
        self.log_reg = LogisticRegression(solver='lbfgs')
        
    def __call__(self, df, *, verbose=False):
        """ """
        x = df[self.req_columns]
        y = df['outcome1_label']

        n_b, n_s = x.shape
        if verbose:
            print("*"*10, "INPUT", "*"*10)
            print(x.shape)
            print(y.shape)

        return self.log_reg.fit(x, y)


class ReModel():
    req_columns = [
            'sid_id',
            'vptsingleh', 'vptsinglea', 'gspan', 'hrank', 'arank', 'relstrngthh', 'relstrngtha',
            'trndstrngthhn', 'trndstrngthhhn', 'trndstrngthan', 'trndstrngthaan', 'vpoints3h',
            'vpoints3a', 'vpointspos3h', 'vpointspos3a', 'vpointsneg3h', 'vpointsneg3a', 'instabilityscoreh',
            'instabilityscorehh', 'instabilityscorea', 'instabilityscoreaa', 'instabilitytrendh', 'instabilitytrenda',
            'oddsh', 'oddsd', 'oddsa'
        ]
    
    def arg_list(cls):
        return self.params

    def __init__(self, solver, **kwargs):
        """ """
        super(ReModel, self).__init__()

        self.solver = solver or 'lbfgs'
        self.params = {"n_comp": n_comp, **kwargs}
        
        self.log_reg = LogisticRegression(solver='lbfgs')
        
    def __call__(self, df, *, verbose=False):
        """ """
        x = df[self.req_columns]
        y = df['outcome1_label']

        n_b, n_s = x.shape
        if verbose:
            print("*"*10, "INPUT", "*"*10)
            print(x.shape)
            print(y.shape)

        return self.log_reg.fit(x, y)


class CNNModel(nn.Module):
    @classmethod
    def arg_list(cls):
        return ["input_size", "filter_sizes", "n_classes", "kernel_size", "dropout_p", "bi"]

    def __init__(self, input_size, filter_sizes, n_classes, kernel_size=3, dropout_p=0.2, bi=False):
        """ """
        super(CNNModel, self).__init__()

        self.input_size   = input_size
        self.filter_sizes = filter_sizes
        self.n_classes    = n_classes
        self.kernel_size  = kernel_size
        self.dropout_p    = dropout_p
        self.bi           = bi

        curr_out_size = self.input_size

        channel_sizes = [1] + self.filter_sizes
        for n in range(1, len(channel_sizes)):
            setattr(self, f"cnn_{n}", nn.Conv1d(
                in_channels =channel_sizes[n-1],
                out_channels=channel_sizes[n],
                kernel_size=self.kernel_size,))

            curr_out_size -= (self.kernel_size - 1)

            setattr(self, f"max_{n}", nn.MaxPool1d(
                kernel_size=2,
                stride     =2,
                padding    =1,))

            curr_out_size = (curr_out_size + 2) // 2

        self.dropout = nn.Dropout(self.dropout_p)
        
        if self.bi:
            self.bi_classifier = nn.Linear(
                in_features=self.filter_sizes[-1] * curr_out_size,
                out_features=1)
                
            self.multi_classifier = nn.Linear(
                in_features =self.filter_sizes[-1] * curr_out_size,
                out_features=self.n_classes-1)
            
        else:
            self.classifier = nn.Linear(
                in_features =self.filter_sizes[-1] * curr_out_size,
                out_features=self.n_classes)
        
    def forward(self, x, *, verbose=False):
        """ """
        x = x.unsqueeze(1)
        n_b, n_c, n_s = x.shape
        if verbose:
            print("*"*10, "INPUT", "*"*10)
            print(x.shape)
            
        yhat = F.relu(self.max_1(self.cnn_1(x)))
        for n in range(2, len(self.filter_sizes)+1):
            yhat = F.relu(getattr(self, f"max_{n}")(getattr(self, f"cnn_{n}")(yhat)))

        yhat = yhat.view(n_b, -1)
        if verbose:
            print("\n")
            print("*"*10, "Y_HAT", "*"*10)
            print(yhat.shape)

        yhat = self.dropout(yhat)
        out = self.classifier(yhat) if not self.bi else\
            [self.bi_classifier(yhat), self.multi_classifier(yhat)]

        if verbose:
            print("\n")
            print("*"*10, "OUTPUT", "*"*10)
            print(out.shape if not self.bi else [t.shape for t in out])
        return out


class RNNModel(nn.Module):
    @classmethod
    def arg_list(cls):
        return ["input_size", "hidden_size", "n_classes", "n_layers", "dropout_p", "bi"]

    def __init__(self, input_size, hidden_size, n_classes, n_layers=2, dropout_p=0.2, bi=False):
        """ """
        super(RNNModel, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.n_classes   = n_classes
        self.n_layers    = n_layers
        self.dropout_p   = dropout_p
        self.bi          = bi

        self.rnn = nn.RNN(
            input_size =1,
            hidden_size=self.hidden_size,
            num_layers =self.n_layers,
            batch_first=True,
            bidirectional=True)
        
        self.dropout = nn.Dropout(self.dropout_p)
        
        if self.bi:
            self.bi_classifier = nn.Linear(
                in_features=self.hidden_size,
                out_features=1)
                
            self.multi_classifier = nn.Linear(
                in_features =self.hidden_size,
                out_features=self.n_classes-1)
            
        else:
            
            self.classifier = nn.Linear(
                in_features =self.hidden_size,
                out_features=self.n_classes)
        
    def forward(self, x, *, verbose=False):
        """ """
        x = x.unsqueeze(-1)
        n_b, n_s, n_e = x.shape
        if verbose:
            print("*"*10, "INPUT", "*"*10)
            print(x.shape)
            
        yhat = F.relu(self.rnn(x, None)[0].view(n_b, n_s, 2, self.hidden_size))
        yhat = yhat[:, -1, 0, :] + yhat[:, 0, 1, :]
        if verbose:
            print("\n")
            print("*"*10, "Y_HAT", "*"*10)
            print(yhat.shape)

        yhat = self.dropout(yhat)
        out = self.classifier(yhat) if not self.bi else\
            [self.bi_classifier(yhat), self.multi_classifier(yhat)]

        if verbose:
            print("\n")
            print("*"*10, "OUTPUT", "*"*10)
            print(out.shape if not self.bi else [t.shape for t in out])
        return out


class CRNNModel(nn.Module):
    @classmethod
    def arg_list(cls):
        return ["input_size", "filter_sizes", "hidden_size", "n_classes", "kernel_size",
            "n_layers", "dropout_p", "inter", "bi"]

    def __init__(self, input_size, filter_sizes, hidden_size, n_classes, kernel_size=3,
        n_layers=2, dropout_p=0.2, inter=False, bi=False):
        """ """
        super(CRNNModel, self).__init__()

        self.input_size   = input_size
        self.filter_sizes = filter_sizes
        self.hidden_size  = hidden_size
        self.n_classes    = n_classes
        self.kernel_size  = kernel_size
        self.n_layers     = n_layers
        self.dropout_p    = dropout_p
        self.inter        = inter
        self.bi           = bi

        curr_out_size = self.input_size

        channel_sizes = [1] + self.filter_sizes
        for n in range(1, len(channel_sizes)):
            setattr(self, f"cnn_{n}", nn.Conv1d(
                in_channels =channel_sizes[n-1],
                out_channels=channel_sizes[n],
                kernel_size=self.kernel_size,))

            curr_out_size -= (self.kernel_size - 1)

            setattr(self, f"max_{n}", nn.MaxPool1d(
                kernel_size=2,
                stride     =2,
                padding    =1,))

            curr_out_size = (curr_out_size + 2) // 2

        if self.inter:
            self.inter_fc = nn.Linear(
                in_features =self.filter_sizes[-1],
                out_features=1)

        self.rnn = nn.RNN(
            input_size =1 if self.inter else self.filter_sizes[-1],
            hidden_size=self.hidden_size,
            num_layers =self.n_layers,
            batch_first=True,
            bidirectional=True)
        
        self.dropout = nn.Dropout(self.dropout_p)
        
        if self.bi:
            self.bi_classifier = nn.Linear(
                in_features=self.hidden_size,
                out_features=1)
                
            self.multi_classifier = nn.Linear(
                in_features =self.hidden_size,
                out_features=self.n_classes-1)
            
        else:
            self.classifier = nn.Linear(
                in_features =self.hidden_size,
                out_features=self.n_classes)
        
    def forward(self, x, *, verbose=False):
        """ """
        x = x.unsqueeze(1)
        n_b, n_c, n_s = x.shape
        if verbose:
            print("*"*10, "INPUT", "*"*10)
            print(x.shape)
            
        yhat = F.relu(self.max_1(self.cnn_1(x)))
        for n in range(2, len(self.filter_sizes)+1):
            yhat = F.relu(getattr(self, f"max_{n}")(getattr(self, f"cnn_{n}")(yhat)))

        yhat = yhat.permute(0, 2, 1)
        if self.inter:
            yhat = F.relu(self.inter_fc(yhat))
        
        n_b, n_s, n_e = yhat.shape
        if verbose:
            print("\n")
            print("*"*10, "LSTM INPUT", "*"*10)
            print(yhat.shape)
            
        yhat = F.relu(self.rnn(yhat, None)[0].view(n_b, n_s, 2, self.hidden_size))
        yhat = yhat[:, -1, 0, :] + yhat[:, 0, 1, :]
        if verbose:
            print("\n")
            print("*"*10, "Y_HAT", "*"*10)
            print(yhat.shape)
        
        yhat = self.dropout(yhat)
        out = self.classifier(yhat) if not self.bi else\
            [self.bi_classifier(yhat), self.multi_classifier(yhat)]

        if verbose:
            print("\n")
            print("*"*10, "OUTPUT", "*"*10)
            print(out.shape if not self.bi else [t.shape for t in out])
        return out


def load_model(model_cls, path):
    m_data = torch.load(path)
    m = model_cls(**{k: m_data[k] for k in model_cls.arg_list() if k in m_data})
    print(m.load_state_dict(m_data["state_dict"], False))
    loss_history = m_data["loss_history"]
    return m, loss_history


def save_model(m, loss_history, path):
    m_data = {k: getattr(m, k) for k in m.arg_list() if hasattr(m, k)}
    m_data["state_dict"] = m.state_dict()
    m_data["loss_history"] = loss_history
    torch.save(m_data, path)

