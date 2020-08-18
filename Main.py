import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from Stock_data import *


BATCH_SIZE=32

model = keras.Sequential()

model.add(layers.LSTM(64, name="InputLSTMLayer")),
model.add(layers.Dense(100, activation="selu", name="InnerLayer"))
model.add(layers.Dense(4, name="OutputLayer"))

model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanAbsoluteError(),
    # List of metrics to monitor
    metrics=["accuracy"],
)


stock_data = get_stock_data("03/05/2016", "29/12/2019")

data = keras.preprocessing.timeseries_dataset_from_array(
    stock_data[:-1],
    stock_data[1:],
    stock_data.shape[1],
    batch_size=BATCH_SIZE)

history = model.fit(
  data,
  batch_size=BATCH_SIZE,
  epochs=100)

print(history.history)


plt.figure(figsize = (8,6))
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('loss')

plt.show()
