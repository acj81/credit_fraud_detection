import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import pickle

# model with Ridge Classification:
from sklearn.linear_model import RidgeClassifier

model = RidgeClassifier()
model.fit(features, labels)


# get accuracy for each:
pred_labels = model.predict(features)
accuracy = (pred_labels == labels).mean()

print(f"Ridge Classifier Accuracy: {accuracy}")

pickle.dump(model, open("/content/drive/My Drive/ridge_classifier.pkl", "wb"))
