# IMPORTS
import tensorflow as tf
import numpy as np

## LOCAL IMPORTS
from init_datasets import testset, validset, trainset
from network import Affine_Network

# Load/Preprocess Datasets
print('*** PREPROCESSING ***')
trainset.preprocess_data()
testset.preprocess_data()
validset.preprocess_data()

# X, Y Process
train_x = []
train_y = []

for pair in trainset.images.values():
    s = np.array(pair[0])
    m = np.array(pair[1])
    p = [s, m]
    train_x.append(p)
for lab in trainset.labels.values():
    train_y.append(lab)


valid_x = []
valid_y = []

for pair in validset.images.values():
    s = np.array(pair[0])
    m = np.array(pair[1])
    p = [s, m]
    valid_x.append(p)
for lab in validset.labels.values():
    valid_y.append(lab)

train_x = np.array(train_x).reshape(-1, 512, 512, 2)
train_y = np.array(train_y)

valid_x = np.array(valid_x).reshape(-1, 512, 512, 2)
valid_y = np.array(valid_y)

# Define Model
model = Affine_Network()
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'mse')
history = model.fit(train_x, train_y, validation_data = (valid_x, valid_y), 
          epochs = 150, callbacks = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 8), batch_size = 10)

print(history)
