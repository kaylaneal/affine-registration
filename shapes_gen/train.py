import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Local Imports
from model import model
from dataset import ShapeDataset

# Load Data
shape_dataset = ShapeDataset(csv = 'shape_dataset.csv')
shape_dataset.process()

X = [ p for p in shape_dataset.pairs.values() ]
X = np.array(X)
y = [ l for l in shape_dataset.labels.values() ]
y = np.array(y)

# Train Model
model.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = 'mse', metrics = ['accuracy'])
hist = model.fit(X, y, batch_size = 8, epochs = 100, callbacks = [tf.keras.callbacks.EarlyStopping(patience = 8, monitor = 'loss')])

model.save('shape_reg')

# Plot History
plt.plot(hist.history['loss'])
plt.title('Model Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('figures/model_loss.png')

plt.clf()

plt.plot(hist.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('figures/model_acc.png')