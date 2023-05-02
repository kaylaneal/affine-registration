# IMPORTS
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np

## LOCAL IMPORTS
from init_datasets import trainset

# Show Truth and Inputs:

figure = plt.figure()
figure.suptitle(f'Truth Label: \n{trainset.labels[0]}')
figure.add_subplot(121)
plt.imshow(Image.open(trainset.images[0][0]))
plt.title('Static Image')
plt.axis('off')
figure.add_subplot(122)
plt.imshow(Image.open(trainset.images[0][1]))
plt.title('Moving Image')
plt.axis('off')

figure.tight_layout()
figure.savefig('figures/trainset_0.png')

# Load Model:
model = tf.keras.models.load_model('AffineNet')

# Load Data:
trainset.preprocess_data()

x = []
y = []

x.append(np.array(trainset.images.get(0)))
y.append(np.array(trainset.labels.get(0)))

x = np.array(x).reshape(-1, 512, 512, 2)
y = np.array(y)

print('*** Predict ***')
pred = model.predict(x)

print(f'Prediction: {pred}')
print(f'Truth: {y}')
print(f'Actual Transformation: {trainset.labels.get(0)}')