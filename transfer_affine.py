# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
## LOCAL IMPORTS
from init_datasets import trainset, validset
from transfer_network import model, base

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

model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mse', metrics = ['accuracy', 'mae'])
history = model.fit([train_sx, train_mx], train_y, 
                    validation_data = ([valid_sx, valid_mx], valid_y),
                    batch_size = 32, epochs = 30)

tf.keras.models.save_model(model, 'transfer_affine')

# Plot Results
print('*** ANALYZING RESULTS ***')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE Loss Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Squared Error')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig('transferlearn_figs/loss.png')
plt.clf()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE Metrics Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig('transferlearn_figs/mae.png')
plt.clf()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Curve')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig('transferlearn_figs/acc.png')
plt.clf()

mse_loss, acc, mae_loss = model.evaluate([valid_sx, valid_mx], valid_y)
print('Evaluation:')
print(f'\tAccuracy: {acc * 100:.2f}')
print(f'\tMean Squared Error: {mse_loss:.2f}')
print(f'\tMean Absolute Error: {mae_loss:.2f}')



## FINE TUNING
base.trainable = True

model.compile(optimizer = tf.keras.optimizers.Adam(1e-5), loss = 'mse', metrics = ['accuracy', 'mae'])
model.fit([train_sx, train_mx], train_y,
          validation_data = ([valid_sx, valid_mx], valid_y),
          batch_size = 32, epochs = 20)