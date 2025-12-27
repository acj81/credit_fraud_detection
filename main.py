import torch
import numpy as np
import pandas as pd
import models
import datasets

from torch.utils.data import DataLoader
from torch import nn

if __name__ == "__main__":
        
    # test perceptron 
    model = models.Perceptron(
        inp_size = 28
    )

    # test forward prop:
    features = torch.randn(5, 28)
    labels = torch.randn(5, 1)
    pred = model(features)

    # --- APPLICATION ---
    # -- CHANGE MODEL AS WE TEST:
    model = models.Perceptron(
        inp_size = 29
    )
    model.to("cpu")

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
    
    # --- PERCEPTRON ---
    print(f"--- PERCEPTRON ---")

    # test on data:
    loss = model.test_loop(
        data_loader = test_loader,
        loss_fn = nn.MSELoss()
    )
    print(f"start loss: {loss}")
    
    # train on data:
    model.train_loop(
        data_loader = train_loader,
        num_epochs = 10,
        loss_fn = nn.MSELoss(),
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    )

    # test on data:
    loss = model.test_loop(
        data_loader = test_loader,
        loss_fn = nn.MSELoss()
    )
    print(f"end loss: {loss}")

    # --- 2-layer NN --- :
    print(f" --- 2-layer NN --- ")
    model = models.NeuralNet(
        dimensions = [[29, 14],[14, 1]]
    )

    # test on data:
    loss = model.test_loop(
        data_loader = test_loader,
        loss_fn = nn.MSELoss()
    )
    print(f"start loss: {loss}")
    
    # train on data:
    model.train_loop(
        data_loader = train_loader,
        num_epochs = 10,
        loss_fn = nn.MSELoss(),
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    )

    # test on data:
    loss = model.test_loop(
        data_loader = test_loader,
        loss_fn = nn.MSELoss()
    )
    print(f"end loss: {loss}")

    # --- 3-layer NN --- :
    print(f" --- 2-layer NN --- ")
    model = models.NeuralNet(
        dimensions = [[29, 14],[14,14],[14, 1]]
    )

    # test on data:
    loss = model.test_loop(
        data_loader = test_loader,
        loss_fn = nn.MSELoss()
    )
    print(f"start loss: {loss}")
    
    # train on data:
    model.train_loop(
        data_loader = train_loader,
        num_epochs = 10,
        loss_fn = nn.MSELoss(),
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    )

    # test on data:
    loss = model.test_loop(
        data_loader = test_loader,
        loss_fn = nn.MSELoss()
    )
    print(f"end loss: {loss}")
