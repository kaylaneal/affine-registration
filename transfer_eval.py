# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

## LOCAL IMPORTS
from init_datasets import testset
from dataset import denormalize

## Data Loading and Preprocessing
print('** LOADING TEST DATA **')

test_s, test_m = testset.process_imgpairs()
test_s = np.array(test_s)
test_m = np.array(test_m)
test_y = testset.process_labels()

# Load Model
print('** LOADING TRAINING MODEL **')
model = load_model('transfer_affine')

predictions = model.predict([test_s, test_m])

## PLOTTING:
print('Plotting Test Results')
test_fig = plt.figure(figsize = (24, 24))

test_fig.add_subplot(131)
plt.imshow(np.concatenate((test_s[0], test_m[0])))
plt.title(f'Truth Label: {test_y[0]}')
plt.axis('off')

test_fig.add_subplot(132)
plt.imshow(np.concatenate((test_s[1], test_m[1])))
plt.title(f'Truth Label: {test_y[1]}')
plt.axis('off')

test_fig.add_subplot(133)
plt.imshow(np.concatenate((test_s[2], test_m[2])))
plt.title(f'Truth Label: {test_y[2]}')
plt.axis('off')

test_fig.tight_layout()
test_fig.savefig('transferlearn_figs/test_imgs.png')

print(f'TEST IMAGE 1\nTruth Label: {test_y[0]} --> Predicted Label: [{float(predictions[0][0]), float(predictions[1][0]), float(predictions[2][0])}]')
print(f'TEST IMAGE 2\nTruth Label: {test_y[1]} --> Predicted Label: [{float(predictions[0][1]), float(predictions[1][1]), float(predictions[2][1])}]')
print(f'TEST IMAGE 3\nTruth Label: {test_y[2]} --> Predicted Label: [{float(predictions[0][2]), float(predictions[1][2]), float(predictions[2][2])}]')

decoded_predictions = []
for p in predictions:
    px, py, pa = p[0], p[1], p[2]
    px = denormalize(px, testset.min_x, testset.max_x)
    py = denormalize(py, testset.min_y, testset.max_y)
    pa = denormalize(pa, testset.min_a, testset.max_a)

    decoded_predictions.append([px, py, pa])

print()
print(f'Truth Value: {testset.labels[0]}')
print(f'Decoded Prediction: [{float(decoded_predictions[0][0])}, {float(decoded_predictions[1][0])}, {float(decoded_predictions[2][0])}]')

