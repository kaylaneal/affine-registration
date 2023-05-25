# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## LOCAL IMPORTS
from init_datasets import trainset, validset
from transfer_network import model, resnet

# Data Loading and Preprocessing
print('*** LOADING DATA ***')
trainset.process_data()
train_x = trainset.pairs
train_y = trainset.labels

if len(train_x) == len(train_y):
    print(f'Train Set Size: {len(train_x)} Samples')
else:
    print('Oopsie! Error in processing!')
    print(f'Train X: {len(train_x)}')
    print(f'Train Y: {len(train_y)}')

validset.process_data()
valid_x = validset.pairs
valid_y = validset.labels

if len(valid_x) == len(valid_y):
    print(f'Validation Set Size: {len(valid_x)} Samples')
else:
    print('Oopsie! Error in processing!')
    print(f'Valid X: {len(valid_x)}')
    print(f'Valid Y: {len(valid_y)}')

# Training
print('*** TRAINING MODEL ***')

lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 100, decay_rate = 1)
model.compile(optimizer = tf.keras.optimizers.Adam(lr_sched), loss = 'mse', metrics = ['mae'])

t = 0
for layer in model.layers:
    if layer.trainable == True:
        t += 1
print(f'{t}/{len(model.layers)} layers are trainable.')

history = model.fit(train_x, train_y, 
                    validation_data = (valid_x, valid_y),
                    batch_size = 32, epochs = 200,
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 10)])
                    
# tf.keras.models.save_model(model, 'transfer_model/transfer_affine')

# Plot Results
print('*** ANALYZING RESULTS ***')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE Loss Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Squared Error')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('transfer_model/transferlearn_figs/loss.png')
plt.clf()

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE Metrics Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('transfer_model/transferlearn_figs/mae.png')
plt.clf()

mse_loss, mae_loss = model.evaluate(valid_x, valid_y)
print('Evaluation:')
print(f'\tMean Squared Error: {mse_loss:.2f}')
print(f'\tMean Absolute Error: {mae_loss:.2f}')

'''
## FINE TUNING
print('** FINE TUNING **')

resnet.trainable = True

model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss = 'mse', metrics = ['mae'])

t = 0
for layer in model.layers:
    if layer.trainable == True:
        t += 1
print(f'{t}/{len(model.layers)} layers are trainable.')

ft_hist = model.fit(train_x, train_y,
          validation_data = (valid_x, valid_y),
          batch_size = 16, epochs = 30,
          callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)])

print('*** ANALYZING TUNED RESULTS ***')

plt.plot(ft_hist.history['loss'])
plt.plot(ft_hist.history['val_loss'])
plt.title('Fine Tune MSE Loss Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Squared Error')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('transfer_model/transferlearn_figs/ft_loss.png')
plt.clf()

plt.plot(ft_hist.history['mae'])
plt.plot(ft_hist.history['val_mae'])
plt.title('Fine Tune MAE Metrics Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('transfer_model/transferlearn_figs/ft_mae.png')
plt.clf()

mse_loss, mae_loss = model.evaluate(valid_x, valid_y)
print('Evaluation:')
print(f'\tMean Squared Error: {mse_loss:.2f}')
print(f'\tMean Absolute Error: {mae_loss:.2f}')
'''
tf.keras.models.save_model(model, 'transfer_model/transfer_affine')