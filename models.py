import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy

import datasets

# 3 models - perceptron, basic Neural Net, Deep Neural Net
class Perceptron:
    def __init__(self, inp_size, loss_fn, lr):
        # set weights and biases - requires_grad so we can do backprop later:
        self.weights = torch.randn(inp_size, 1, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, features):
        # calculate:
        pred_labels = features @ self.weights + self.bias
        return pred_labels
    
    def train_on_batch(self, features, labels):
        # forward to get predictions:
        pred_labels = self.forward(features)
        # get loss using predicted and actual labels:
        loss = self.loss_fn(pred_labels, labels)
        # apply to each of them:
        loss.backward()
        w_grad, b_grad = self.weights.grad, self.bias.grad
        with torch.no_grad():
            self.weights += w_grad * self.lr
            self.bias += b_grad * self.lr
        # clear gradient so doesn't build up over time:
        self.weights.grad.zero_()
        self.bias.grad.zero_()

    def test_on_batch(self, features, labels):
        # no grad so doesn't interfere with gradients:
        with torch.no_grad():
            # forward to get predictions:
            pred_labels = self.forward(features)
            # get average loss between predicted and actual labels:
            loss = self.loss_fn(pred_labels, labels)
            avg_loss = loss.item() / len(features)
        return avg_loss 
        

    def test_on_loader(self, loader):
        # average loss we return @ the end:
        avg_loss = 0
        for batch_num, (batch_features, batch_labels) in enumerate(loader):
            #print(f"testing batch: {batch_num}")
            # get loss for this batch:
            batch_loss = self.test_on_batch(batch_features, batch_labels)
            avg_loss += batch_loss
        # divide to get average:
        avg_loss /=  len(loader)
        return avg_loss

    def train_on_loader(self, loader, num_epochs):
        for i in range(num_epochs):
            print(f"training epoch: {i}")
            for batch_num, (batch_features, batch_labels) in enumerate(loader):
                #print(f"training batch: {batch_num}")
                # train on batch:
                self.train_on_batch(batch_features, batch_labels)

# -- testing ---

if __name__ == "__main__":
        
    # test perceptron 
    model = Perceptron(
        inp_size = 5,
        loss_fn = nn.MSELoss(),
        lr = 0.01    
    )

    # test forward prop:
    features = torch.randn(5, 5)
    labels = torch.randn(5, 1)
    pred = model.forward(features)

    # test backpropagation is updated model parameters:
    original_weights, original_bias = deepcopy(model.weights), deepcopy(model.bias)
    model.train_on_batch(features, labels)
    new_weights, new_bias = model.weights, model.bias
    # see if backprop updated weights:
    if (original_weights == new_weights).all():
        raise Exception("weights weren't updated")
    # see if backprop updated bias:
    if (original_bias == new_bias).all():
        raise Exception("bias wasn't updated")
    
    # make sure test function works:
    loss_value = model.test_on_batch(features, labels)
    #print(loss_value)

    # --- APPLICATION ---
    # -- CHANGE MODEL AS WE TEST:
    model = Perceptron(
        inp_size = 29,
        loss_fn = nn.MSELoss(),
        lr = 0.01    
    )

    # get DataFrame from .csv, split into train and test datasets:
    data = pd.read_csv("creditcard.csv")
    partition_idx = int(len(data) * 0.8)
    # get all 28 feature columns without having to manually specify
    feature_cols = [x for x in data.columns if (x[0] == "V" or x == "Amount")]
    features = data[feature_cols]
    labels = data[["Class"]]
    # create individual datasets - use partition idx to split features and labels into test and train datasets:
    train_data = datasets.CSVDataset(features.iloc[:partition_idx], labels.iloc[:partition_idx])
    test_data = datasets.CSVDataset(features.iloc[partition_idx:], labels.iloc[partition_idx:])
    # pass to DataLoaders:
    train_loader = DataLoader(
        train_data, 
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=64,
        shuffle=True
    )
    
    # test on data:
    loss = model.test_on_loader(
        loader = test_loader
    )
    print(f"start loss: {loss}")
    
    # train on data:
    model.train_on_loader(
        loader = train_loader,
        num_epochs = 10
    )

    # test on data:
    loss = model.test_on_loader(
        loader = test_loader
    )
    print(f"end loss: {loss}")
