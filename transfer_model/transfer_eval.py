# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

## LOCAL IMPORTS
from init_datasets import testset
from dataset import denormalize

## Data Loading and Preprocessing
print('** LOADING TEST DATA **')
testset.process_data()
test_x = testset.pairs
test_y = testset.labels

max_a = testset.max_a
max_x = testset.max_x
max_y = testset.max_y

min_a = testset.min_a
min_x = testset.min_x
min_y = testset.min_y

# Load Model
print()
print('** LOADING TRAINED MODEL **')
model = load_model('transfer_model/transfer_affine')

print()
print('** EVALUATION **\n')
model.evaluate(test_x, test_y)

print()
print('** PREDICTION **')
predictions = model.predict(test_x, verbose = 2)

# Prediction Exploration
truth, pred = [], []  

for l in test_y:

    lx = l[0]
    ly = l[1]
    la = l[2]

    lx = denormalize(lx, min_x, max_x)
    ly = denormalize(ly, min_y, max_y)
    la = denormalize(la, min_a, max_a)

    truth.append([lx, ly, la])
    
for p in predictions:

    px = p[0]
    py = p[1]
    pa = p[2]

    px = denormalize(px, min_x, max_x)
    py = denormalize(py, min_y, max_y)
    pa = denormalize(pa, min_a, max_a)

    pred.append([px, py, pa])

for i in range(5):
    print(f'Set {i}:\n\tTruth Value: {truth[i]}\n\tPredicted Value: {pred[i]}')

