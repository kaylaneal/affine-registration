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
y = np.array(test_y)
# Load Model
print()
print('** LOADING TRAINING MODEL **')
model = load_model('transfer_affine')

print()
print('** EVALUATION **\n')
mse, acc, mae = model.evaluate([test_s, test_m], y)
print(f'\tMean Squared Error (LOSS): {mse:.2f}')
print(f'\tAccuracy: {(acc * 100):.0f}%')

print()
print('** PREDICTION **')
predictions = model.predict([test_s, test_m])

## PLOTTING:
print()
print('Plotting Test Results')

plt.imshow(np.concatenate((test_s[0], test_m[0]), axis = 1))
plt.suptitle('Testset Image Pair 0')
plt.title(f'Truth Label: {test_y[0]}')
plt.axis('off')
plt.savefig('transferlearn_figs/test_imgs.png')
plt.clf()

print(f'TEST IMAGE 1\nTruth Label: {test_y[0]} --> Predicted Label: [{float(predictions[0][0]), float(predictions[0][1]), float(predictions[0][2])}]')
print(f'TEST IMAGE 2\nTruth Label: {test_y[1]} --> Predicted Label: [{float(predictions[1][0]), float(predictions[1][1]), float(predictions[1][2])}]')
print(f'TEST IMAGE 3\nTruth Label: {test_y[2]} --> Predicted Label: [{float(predictions[2][0]), float(predictions[2][1]), float(predictions[2][2])}]')

decoded_predictions = []
for p in predictions:
    px, py, pa = p[0], p[1], p[2]
    px = denormalize(px, testset.min_x, testset.max_x)
    py = denormalize(py, testset.min_y, testset.max_y)
    pa = denormalize(pa, testset.min_a, testset.max_a)

    decoded_predictions.append([px, py, pa])

print()
print(f'Truth Value: {testset.labels[0]}')
print(f'Decoded Prediction: [{float(decoded_predictions[0][0])}, {float(decoded_predictions[0][1])}, {float(decoded_predictions[0][2])}]')
