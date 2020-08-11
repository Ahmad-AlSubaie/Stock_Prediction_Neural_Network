import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models 

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from  IPython import display
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from Stock_data import *


model = keras.Sequential()

model.add(layers.Dense(60, activation="relu", input_shape=(None, 4*7*2), name="InputLayer"))
model.add(layers.Dense(30, activation="relu", name="InnerLayer"))
model.add(layers.Dense(4, name="OutputLayer"))

model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanSquaredLogarithmicError(),
    # List of metrics to monitor
    metrics=["accuracy"],
)


data_x, data_y = get_useable_data_lable_pair("03/01/2010", "29/12/2015")


val_x, val_y =  get_useable_data_lable_pair("15/05/2017", "15/05/2018")
    
print("-----------------------------------------------------")

history = model.fit(
  data_x, data_y,
  validation_data = (val_x, val_y),
  batch_size=64,
  epochs=100)


plt.figure(figsize = (8,6))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('loss')

plt.show()

pred_x, pred_y = get_useable_data_lable_pair("15/9/2018", "15/11/2018")

print(pred_y)

model.predict(pred_x)
