import torch
import numpy as np
import pandas as pd
from torch import nn

import datasets

class NeuralNet(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        # declare initial layers:
        self.layers = nn.Sequential()
        # add layers based on dimensions given:
        for in_size, out_size in dimensions:
            # add linear layer w/ dimensions passed:
            self.layers.append(nn.Linear(in_size, out_size)) 
            # add ReLU layer after for stack:
            self.layers.append(nn.ReLU())

    def forward(self, features):
        # calculate:
        pred_labels = self.layers(features)
        return pred_labels

    def train_loop(
        self,
        data_loader,
        num_epochs, 
        optimizer,
        loss_fn,
    ):
        # iterate for epochs:
        for epoch in range(num_epochs):
            print(f"training epoch: {epoch}")
            # iterate for each batch:
            for features, labels in data_loader:
                # get predictions for batch:
                pred_labels = self(features)
                # get loss and backpropagate for gradients:
                loss = loss_fn(pred_labels, labels)
                loss.backward()
                # step optimizer to apply gradients and then reset those gradients w/ zero_grad:
                optimizer.step()
                optimizer.zero_grad()

    def test_loop(
            self, 
            data_loader, 
            loss_fn
    ):
        self.eval()
        # average loss value we return:
        avg_loss = 0
        total_examples = 0
        # don't need to calculate gradients so more efficient here:
        with torch.no_grad():
                # calculate loss for each batch:
                for features, labels in data_loader:
                    # get predictions, get loss for those predictions
                    pred_labels = self(features)
                    loss = loss_fn(pred_labels, labels)
                    avg_loss += loss
                    # increment total_examples:
                    total_examples += len(features)
        # divide to get average loss @ end:
        avg_loss /= total_examples
        return avg_loss
                    
class Perceptron(NeuralNet):
    def __init__(self, inp_size):
        ''' just a wrapper for NeuralNet so it can use inp_size instead of tuple of dimensions '''
        NeuralNet.__init__(self, [[inp_size, 1]])
