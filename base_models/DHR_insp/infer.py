# IMPORTS
import tensorflow as tf

## LOCAL IMPORTS
from DHR_insp.evaluate import test_x

# Load Model
model = tf.keras.models.load_model('AffineNet')


# Prediction
print('*** PREDICTION ***')

predictions = model.predict(test_x)

print(predictions)