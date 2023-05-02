# IMPORTS
import tensorflow as tf
import matplotlib.pyplot as plt

## LOCAL IMPORTS
from evaluate import test_x, test_y

# Load Model
model = tf.keras.models.load_model('AffineNet')


# Prediction
print('*** PREDICTION ***')

predictions = model.predict(test_x)

print(predictions)