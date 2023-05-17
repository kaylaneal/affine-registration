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

model.compile(optimizer = optimizers.Adam(), loss = 'mse', metrics = ['mae'])
history = model.fit([train_sx, train_mx], [train_y[:, 0], train_y[:, 1], train_y[:, 2]], 
                    validation_data = ([valid_sx, valid_mx], [valid_y[:, 0], valid_y[:, 1], valid_y[:, 2]]),
                    batch_size = 32, epochs = 75)

models.save_model(model, 'transfer_affine')

# Plot Results
print('*** ANALYZING RESULTS ***')
fig = plt.figure(figsize = (12, 12))

fig.add_subplot(3, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc = 'upper left')

fig.add_subplot(3, 1, 2)
plt.plot(history.history['rotation_loss'])
plt.plot(history.history['x_trans_loss'])
plt.plot(history.history['y_trans_loss'])
plt.title('Transformation Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['rotation', 'x translation', 'y translation'], loc = 'upper left')

fig.add_subplot(3, 1, 3)
plt.plot(history.history['val_rotation_loss'])
plt.plot(history.history['val_x_trans_loss'])
plt.plot(history.history['val_y_trans_loss'])
plt.title('VALIDATION Transformation Loss Curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['rotation', 'x translation', 'y translation'], loc = 'upper left')

fig.tight_layout()
fig.savefig('transferlearn_figs/loss_curves.png')

## Results
test_s = train_sx[:3]
test_m = train_mx[:3]
test_labels = train_y[:3]

pred = model.predict([test_s, test_m])

test_fig = plt.figure(figsize = (24, 24))

test_fig.add_subplot(1, 3, 1)
plt.imshow(np.concatenate((test_s[0], test_m[0])))
plt.axis('off')
plt.title(f'Truth: {test_labels[0]}\nPredicted: {np.asarray(pred[0])}')

test_fig.add_subplot(1, 3, 2)
plt.imshow(np.concatenate((test_s[1], test_m[1])))
plt.axis('off')
plt.title(f'Truth: {test_labels[1]}\nPredicted: {np.asarray(pred[1])}')

test_fig.add_subplot(1, 3, 3)
plt.imshow(np.concatenate((test_s[2], test_m[2])))
plt.axis('off')
plt.title(f'Truth: {test_labels[2]}\nPredicted: {np.asarray(pred[2])}')

test_fig.tight_layout()
test_fig.savefig('transferlearn_figs/test_imgs.png')

print(f'TEST IMAGE 1\nTruth Label: {test_labels[0]} --> Predicted Label: [{np.asarray(pred[0])}')
print(f'TEST IMAGE 2\nTruth Label: {test_labels[1]} --> Predicted Label: [{np.asarray(pred[1])}]')
print(f'TEST IMAGE 3\nTruth Label: {test_labels[2]} --> Predicted Label: [{np.asarray(pred[2])}]')
