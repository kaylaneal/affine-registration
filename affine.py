# IMPORTS
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks, models, optimizers, losses, metrics, Input
import matplotlib.pyplot as plt

## LOCAL IMPORTS
from init_datasets import validset, trainset
from network import get_model

# Load Data
print('** LOADING DATASET **')
train_sx, train_mx = trainset.process_imgpairs()
train_y = trainset.process_labels()

valid_sx, valid_mx = validset.process_imgpairs()
valid_y = validset.process_labels()

train_sx = np.array(train_sx)
train_mx = np.array(train_mx)
train_y = np.array(train_y)

valid_sx = np.array(valid_sx)
valid_mx = np.array(valid_mx)
valid_y = np.array(valid_y)

# Check Data
'''
print(f'Training: \n\tX Static Shape: {train_sx.shape} \n\tX Moving Shape: {train_mx.shape} \n\tY Shape: {train_y.shape}')
print(f'Validation: \n\tX Static Shape: {valid_sx.shape} \n\tX Moving Shape: {valid_mx.shape} \n\tY Shape: {valid_y.shape}')
'''
# Load Model
print('** BUILDING MODEL **')
model = get_model(Input(shape = (256, 256, 3)), Input(shape = (256, 256, 3)))

lr_schedule = optimizers.schedules.ExponentialDecay(0.0001, decay_rate = 1, decay_steps = 500, staircase = True)
opt = optimizers.Adam(learning_rate = lr_schedule)
loss = losses.MeanSquaredLogarithmicError(name = 'log_mse')
metric = ['accuracy', metrics.RootMeanSquaredError(name = 'rmse')]

model.compile(optimizer = opt, loss = loss, metrics = metric)

# Train Model
print('** TRAINING **')
escb = callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
lcb = callbacks.CSVLogger('training_log.csv')

history = model.fit([train_sx, train_mx], train_y, 
                    validation_data = ([valid_sx, valid_mx], valid_y),
                    batch_size = 32, epochs = 250,
                    callbacks = [escb, lcb])
# model.summary()
models.save_model(model, 'multi_affine')
plot_model(model, to_file = 'figures/strat_model_vis.png', show_shapes = True)

# Results:
f = plt.figure()
f.add_subplot(111)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy Curve')
f.savefig('figures/SAN_acc_curve.png')

lf = plt.figure()
lf.add_subplot(111)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('MSLE Loss Curve')
lf.savefig('figures/SAN_loss_curve.png')

lf2 = plt.figure()
lf2.add_subplot(111)
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.legend(['training', 'validation'])
plt.title('RMSE Loss Curve')
lf2.tight_layout()
lf2.savefig('figures/SAN_metric_curve.png')

# print('Truth Values Training Set Pair 0:\n\t', train_y[0])
# print('Truth Values Training Set Pair 1:\n\t', train_y[1])