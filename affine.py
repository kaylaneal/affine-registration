# IMPORTS
import numpy as np
from tensorflow.keras.utils import plot_model, model_to_dot
from tensorflow.keras import callbacks, models, optimizers, losses, metrics
import matplotlib.pyplot as plt

## LOCAL IMPORTS
from init_datasets import validset, trainset
from network import Affine_Network

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

print(f'Training: \n\tX Static Shape: {train_sx.shape} \n\tX Moving Shape: {train_mx.shape} \n\tY Shape: {train_y.shape}')
print(f'Validation: \n\tX Static Shape: {valid_sx.shape} \n\tX Moving Shape: {valid_mx.shape} \n\tY Shape: {valid_y.shape}')

# Load Model
print('** BUILDING MODEL **')
model = Affine_Network()

lr_schedule = optimizers.schedules.InverseTimeDecay(0.0001, decay_rate = 1, decay_steps = 10)
opt = optimizers.Adam(learning_rate = lr_schedule)
loss = losses.MeanSquaredLogarithmicError(name = 'log_mse')
metric = ['accuracy', metrics.RootMeanSquaredError(name = 'rmse'), losses.MeanSquaredError(name = 'mse')]

model.compile(optimizer = opt, loss = loss, metrics = metric)

# Train Model
print('** TRAINING **')
callback = callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

history = model.fit([train_sx, train_mx], train_y, 
                    validation_data = ([valid_sx, valid_mx], valid_y),
                    batch_size = 18, epochs = 150,
                    callbacks = [callback])
# model.summary()
models.save_model(model, 'stratified_input_AN')
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
lf2.add_subplot(211)
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.legend(['training', 'validation'])
plt.title('RMSE Loss Curve')
lf2.add_subplot(212)
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.legend(['training', 'validation'])
plt.title('MSE Loss Curve')
lf2.tight_layout()
lf2.savefig('figures/SAN_metric_curve.png')

# print('Truth Values Training Set Pair 0:\n\t', train_y[0])
# print('Truth Values Training Set Pair 1:\n\t', train_y[1])

plot_model(model, to_file = 'figures/strat_model_vis.png', show_shapes = True)
model_to_dot(model, show_shapes = True)