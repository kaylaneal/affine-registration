# IMPORTS
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

# Load Model
print()
print('** LOADING TRAINED MODEL **')
model = load_model('transfer_model/transfer_affine')

print()
print('** EVALUATION **\n')
mse, mae = model.evaluate(test_x, test_y)
print(f'\tMean Squared Error: {mse:.2f}')

print()
print('** PREDICTION **')
predictions = model.predict(test_x)

print(test_y[0])
print(predictions[0])