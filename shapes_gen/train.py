# IMPORTS
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
## Local Imports
from model import model
from dataset import ShapeDataset

# Load Data
shape_dataset = ShapeDataset(csv = 'shapeset.csv')
shape_dataset.process()

X = [ p for p in shape_dataset.pairs.values() ]
X = np.array(X, dtype = np.float32)
y = [ l for l in shape_dataset.labels.values() ]
y = np.array(y, dtype = np.float32)

# Train Model
model.compile(optimizer = tf.keras.optimizers.Adam(0.0001), loss = 'mse', metrics = ['accuracy'])
hist = model.fit(X, y, 
                 validation_split = 0.3,
                 batch_size = 1, epochs = 100, 
                 callbacks = [tf.keras.callbacks.EarlyStopping(patience = 8, monitor = 'val_loss', restore_best_weights = True)])

model.save('shape_reg')
# model.summary()

# Plot History
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['training', 'validation'])

plt.tight_layout()
plt.savefig('figures/model_loss.png')

plt.clf()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['training', 'validation'])

plt.tight_layout()
plt.savefig('figures/model_acc.png')