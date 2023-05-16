# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers, models, losses, callbacks

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

lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001, decay_rate = 1, decay_steps = 40)
opt = optimizers.Adam(learning_rate = lr_schedule)

model.compile(optimizer = opt, loss = 'mse', metrics = ['accuracy'])
history = model.fit([train_sx, train_mx], train_y, 
                    validation_data = ([valid_sx, valid_mx], valid_y),
                    batch_size = 32, epochs = 50,
                    callbacks = [callbacks.EarlyStopping(patience = 5, restore_best_weights = True)])

models.save_model(model, 'transfer_affine')

# Plot Results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy Curve')
plt.savefig('transferlearn_figs/accuracy.png')

plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Mean Squared Error Loss Curve')
plt.savefig('transferlearn_figs/mse.png')

plt.clf()
