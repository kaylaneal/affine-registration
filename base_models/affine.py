# IMPORTS
from PIL import Image
import numpy as np
from tensorflow.keras import callbacks, models, optimizers, losses, metrics
import matplotlib.pyplot as plt

## LOCAL IMPORTS
from init_datasets import validset, trainset
from initial_network import Affine_Network

## MODEL
print('** BUILDING MODEL **')
# Build Model:
model = Affine_Network()

# Define Inputs: X = pairs, Y = labels
X = []
for pair in trainset.pairs.values():
    s = Image.open(pair[0]).convert('L')
    m = Image.open(pair[1]).convert('L')
    s = np.array(s)
    m = np.array(m)
    s = s / 255.
    m = m / 255.

    input = [s, m]
    X.append(input)

Y = []
for label in trainset.pair_labels.values():
    Y.append(label)

# Define Validation Inputs
valid_X = []
valid_y = []

for pair in validset.pairs.values():
    s = Image.open(pair[0]).convert('L')
    m = Image.open(pair[1]).convert('L')
    s = np.array(s)
    m = np.array(m)
    s = s / 255.
    m = m / 255.

    input = [s, m]
    valid_X.append(input)

for label in validset.pair_labels.values():
    valid_y.append(label)

#print(f'X Samples = {len(X)}\nY Samples = {len(Y)}')
#print(X[0][1].shape) # 256, 256
X = np.array(X).reshape(-1, 256, 256, 2)
Y = np.array(Y)

valid_X = np.array(valid_X).reshape(-1, 256, 256, 2)
valid_y = np.array(valid_y)

# Compile Model
lr_sched = optimizers.schedules.InverseTimeDecay(0.001, decay_steps = 15, decay_rate = 1)
model.compile(optimizer = optimizers.Adam(learning_rate = lr_sched), loss = losses.MeanSquaredLogarithmicError(), metrics = ['accuracy', metrics.RootMeanSquaredError(name = 'rmse')])

# Train Model
print('\n** TRAINING **')
hist = model.fit(X, Y, validation_data = (valid_X, valid_y), 
                 batch_size = 32, epochs = 200,
                 callbacks = [callbacks.EarlyStopping(monitor = 'val_loss', patience = 8, restore_best_weights = True)])
# model.summary()
models.save_model(model, 'affine_model')

# Results:
f = plt.figure()
f.add_subplot(111)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy Curve')
f.savefig('figures/acc_curve.png')

lf = plt.figure()
lf.add_subplot(111)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('MSLE Loss Curve')
lf.savefig('figures/loss_curve.png')