# IMPORTS
from tensorflow.keras.models import load_model
import numpy as np

## LOCAL IMPORTS
from init_datasets import testset

print('** LOADING TRAINED MODEL **')
model = load_model('stratified_input_AN')

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