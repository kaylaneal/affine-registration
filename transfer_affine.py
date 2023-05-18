# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models, optimizers

## LOCAL IMPORTS
from init_datasets import trainset, validset
from transfer_network import model

# Data Loading and Preprocessing
print('*** LOADING DATA ***')
train_sx, train_mx = trainset.process_imgpairs()
train_sx = np.array(train_sx)
train_mx = np.array(train_mx)
train_y = np.array(trainset.process_labels())

valid_sx, valid_mx = validset.process_imgpairs()
valid_sx = np.array(valid_sx)
valid_mx = np.array(valid_mx)
valid_y = np.array(validset.process_labels())

# Training
print('*** TRAINING MODEL ***')

model.compile(optimizer = optimizers.Adam(learning_rate = 0.0001), loss = 'mse', metrics = ['mae'])
history = model.fit([train_sx, train_mx], [train_y[:, 0], train_y[:, 1], train_y[:, 2]], 
                    validation_data = ([valid_sx, valid_mx], [valid_y[:, 0], valid_y[:, 1], valid_y[:, 2]]),
                    batch_size = 32, epochs = 75)

models.save_model(model, 'transfer_affine')

# Plot Results
print('*** ANALYZING RESULTS ***')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model MSE Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig('transferlearn_figs/loss.png')
plt.clf()

fig = plt.figure(figsize = (12, 12))
fig.add_subplot(2, 1, 1)
plt.plot(history.history['rotation_loss'])
plt.plot(history.history['x_trans_loss'])
plt.plot(history.history['y_trans_loss'])
plt.title('Training Transformation MSE Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['rotation', 'x translation', 'y translation'], loc = 'upper left')

fig.add_subplot(2, 1, 2)
plt.plot(history.history['val_rotation_loss'])
plt.plot(history.history['val_x_trans_loss'])
plt.plot(history.history['val_y_trans_loss'])
plt.title('Validation MSE Transformation Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['rotation', 'x translation', 'y translation'], loc = 'upper left')

fig.tight_layout()
fig.savefig('transferlearn_figs/transformation_losses.png')