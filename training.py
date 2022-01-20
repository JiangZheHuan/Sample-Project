import os
import time
import math
import string
import random

import collections
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.functional import F



def train(model, data, num_iter, criterion, clip=5, lr=0.001, print_every=50,):
    """
    Input:
        model: A Pytorch's nn.Module class.
        data: Training data, containing both x and y.
        num_iter: Number of time to perform backward prop, (update parameters).
        criterion: A function that takes in (out, y) and returns the loss.
        clip: Value to clip gradients to. If None, clipping is not done.
        lr: Learning rate.
        print_every: Number of iterations to print averaged loss. If None, nothing
            is printed.
    
    Output:
        model: The trained model.
        costs: List of all the calculated loss.
    """
    model.train()
    
    costs = []
    running_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    curr_iter = 0
    while curr_iter<num_iter:
        for x, y in data:
            optimizer.zero_grad()

            # Initialise model's state and perform forward-prop
            out = model(x)

            # Calculate loss
            loss = criterion(out, y)#criterion(out.transpose(1, 2), y)
            costs.append(loss.item())
            running_loss += loss.item()

            # Calculate gradients and update parameters
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            curr_iter += 1
            if print_every and (curr_iter%print_every)==0:
                print("Iteration: {:{}}/{}, Loss: {:8.4f}".format(
                    curr_iter, int(math.log(num_iter, 10))+2, num_iter,
                    running_loss/float(print_every)))
                running_loss = 0
                
            if curr_iter>=num_iter:
                break
    return model, costs


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data.copy()
        self.n_features = len(self.data[0]) - 1
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        item = self.data[ix]
        return torch.tensor(item[:-1]), torch.tensor(item[-1]).long()


def keys_to_values(keys, _map, default):
    return [_map.get(key, default) for key in keys]

