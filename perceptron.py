# machine learning libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model.Perceptron

# so we can serialize models later
import pickle

# load dataset:
credit_df = pd.read_csv("creditcard_2023.csv")

# split into features and labels:
features = credit_df[["V1","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount",]]
labels = credit_df["Class"]

# model using Perceptron:
model = Perceptron()
model.fit(features, labels)

# serialize w/ pickle so we don't have to retrain constantly:
pickle.dump(model, open("/content/drive/My Drive/perceptron.pkl", "wb"))
