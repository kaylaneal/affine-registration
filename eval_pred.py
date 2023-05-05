# IMPORTS
import tensorflow as tf
from PIL import Image
import numpy as np

## LOCAL IMPORTS
from create_datasets import testset

print('** LOADING TRAINED MODEL **')
model = tf.keras.models.load_model('affine_model')

test_X = []
test_y = []

for pair in testset.pairs.values():
    s = Image.open(pair[0]).convert('L')
    m = Image.open(pair[1]).convert('L')
    s = np.array(s)
    m = np.array(m)
    s = s / 255.
    m = m / 255.

    input = [s, m]
    test_X.append(input)

for label in testset.pair_labels.values():
    test_y.append(label)

test_X = np.array(test_X).reshape(-1, 256, 256, 2)
test_y = np.array(test_y)

print('** PREDICT **')
predictions = model.predict(test_X, verbose = False)
#model.evaluate(test_X, test_y, verbose = 2)


print(f'Truth Label 0: {test_y[0]}')
print(f'Predicted Label 0: {predictions[0]}')
print()
print(f'Truth Label 10: {test_y[10]}')
print(f'Predicted Label 10: {predictions[10]}')
