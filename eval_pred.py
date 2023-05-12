# IMPORTS
import tensorflow as tf
import numpy as np

## LOCAL IMPORTS
from init_datasets import testset

print('** LOADING TRAINED MODEL **')
model = tf.keras.models.load_model('multi_affine.h5')

print('** LOAD TESTING DATA **')
sx, mx = testset.process_imgpairs()
y = testset.process_labels()

sx = np.array(sx)
mx = np.array(mx)
y = np.array(y)

print('** PREDICT **')
predictions = model.predict([sx, mx], verbose = False)
#model.evaluate([sx, mx], y, verbose = 2)
print()

print('Truth Values\n', y)