# IMPORTS
import numpy as np
import tensorflow as tf

## LOCAL IMPORTS
from DHR_insp.init_datasets import testset

# Load / Preprocess Test Dataset
print('*** Preprocessing Testing Set ***')
testset.preprocess_data()

test_x = []
test_y = []

for pair in testset.images.values():
    s = np.array(pair[0])
    m = np.array(pair[1])
    
    test_x.append([s, m])
for label in testset.labels.values():
    test_y.append(label)

test_x = np.array(test_x).reshape(-1, 512, 512, 2)
test_y = np.array(test_y)

# Load Model
model = tf.keras.models.load_model('AffineNet')

# Evaluate on Test Dataset
print('*** Evaluate Model ***')
results = model.evaluate(test_x, test_y)
print(f'Test Loss: {results[0]:.2f}, Test Accuracy: {results[1] * 100:.1f}')