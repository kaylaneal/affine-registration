# IMPORTS
import tensorflow as tf
import numpy as np

## LOCAL IMPORTS
from init_datasets import testset
from dataset import denormalize

print('** LOADING TRAINED MODEL **')
model = tf.keras.models.load_model('multi_affine')

print('** LOAD TESTING DATA **')
sx, mx = testset.process_imgpairs()
y = testset.process_labels()

sx = np.array(sx)
mx = np.array(mx)
y = np.array(y)

print('** PREDICT **')
predictions = model.predict([sx, mx], verbose = False)
model.evaluate([sx, mx], y, verbose = 2)

decoded_predictions = []
for p in predictions:
    px, py, pa = p[0], p[1], p[2]
    px = denormalize(px, testset.min_x, testset.max_x)
    py = denormalize(py, testset.min_y, testset.max_y)
    pa = denormalize(pa, testset.min_a, testset.max_a)

    decoded_predictions.append([px, py, pa])

print('\nAffine Parameters:')
for i in range(5):
    print()
    print(f'Truth Value Pair {i}: {testset.labels[i]}')
    print(f'Predicted Value Pair {i}: {decoded_predictions[i]}')

