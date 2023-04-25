# IMPORTS
import os
from PIL import Image
import numpy as np
import scipy.ndimage as nd
from tensorflow.keras import callbacks, models
import matplotlib.pyplot as plt

## LOCAL IMPORTS
from dataset import create_dataset
from inital_rotation import find_theta
from network import Affine_Network
import utils

# Data Paths:
json_1 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (1)/info.json'
images_1 = '../pearce-lab/registration/trainingdata/perfectpairs/image-sets (1)' 

# Create Dataset:
dataset = create_dataset(json_1)
masks = {}
transforms = {}
mc = 0

for i in dataset.pairs.items():
    print(f'\npair {i[0]} of {len(dataset.pairs.items())}')
    static = Image.open(os.path.join(images_1, i[1][0])).convert('L') # with convert('L') arrays are shape (256, 256)
    moving = Image.open(os.path.join(images_1, i[1][1])).convert('L')

    print('** Normalize, Pad, Smooth, Resample: **')
    static = np.asarray(static)
    static = 1 - utils.normalize(static)   
    
    moving = np.asarray(moving)
    moving = 1 - utils.normalize(moving)

    #print(f'Static Shape {static.shape}\nMoving Shape {moving.shape}')
    static, moving = utils.pad_images(static, moving)

    resample_factor = np.max(static.shape) / 512                    # max output size
    g_sigma = resample_factor / 1.25

    static = nd.gaussian_filter(static, g_sigma)
    moving = nd.gaussian_filter(moving, g_sigma)

    static = utils.resample_image(static, resample_factor)          # shape: (512, 512)
    moving = utils.resample_image(moving, resample_factor)          # shape: (512, 512)
    #print(f'Static Shape {static.shape}\nMoving Shape {moving.shape}')

    # Segment Background
    static = np.expand_dims(static, (0, -1))                        # Segmentation expects shape of (B, H, W, C)
    moving = np.expand_dims(moving, (0, -1))                        # New Shape: (1, 512, 512, 1)
    #print(f'Static Shape {static.shape}\nMoving Shape {moving.shape}')
    print('** SEGMENTING **')
    seg_static, seg_moving = utils.segmentation(static, moving)
    masks.update(
        { mc : [static - seg_static, moving - seg_moving] } )
    print('** INITIAL TRANSFORM **')
    initial_transformation = find_theta(static, moving)
    transforms.update(
        { mc : initial_transformation }
    )
    i[1][0] = static 
    i[1][1] = moving
    mc += 1

## DATA VIS
print('\n** DATA VIS **')
for i in range(4):
    plt.subplot(2, 2, (i+1))
    d = np.array(dataset.pairs.get(i)[0]).squeeze()
    plt.imshow(d, cmap = 'gray')
    plt.axis('off')
    plt.title(f'Static {i}')
plt.savefig('figures/static_example_inputs.png')

# Mask Pair
mask_fig = plt.figure()
mask_fig.add_subplot(121)
plt.imshow(np.squeeze(masks.get(0)[0], 0), cmap = 'gray')
plt.title('Static0 Mask')
plt.axis('off')
mask_fig.add_subplot(122)
plt.imshow(np.squeeze(masks.get(0)[1], 0), cmap = 'gray')
plt.title('Moving0 Mask')
plt.axis('off')
mask_fig.tight_layout()
mask_fig.savefig('figures/dataset1_pair0_masks.png')


## MODEL
print('\n** BUILDING MODEL **')
# Build Model:
model = Affine_Network()

# Define Inputs: X = pairs, Y = labels
X = []
for pair in dataset.pairs.values():
    s = np.array(pair[0])
    m = np.array(pair[1])
    input = [s, m]
    X.append(input)

Y = []
for label in dataset.pair_labels.values():
    Y.append(label)

#print(f'X Samples = {len(X)}\nY Samples = {len(Y)}')
#print(X[0][1].shape) # 1, 512, 512, 1
X = np.array(X).reshape(-1, 512, 512, 2)
Y = np.array(Y)

# Compile Model
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# Train Model
print('\n** TRAINING **')
hist = model.fit(X, Y, batch_size = 10, validation_split = 0.1, epochs = 100)
# model.summary()
models.save_model(model, 'models/affine_model')

# Results:
f = plt.figure()
f.add_subplot(111)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['training', 'validation'])
f.savefig('figures/acc_curve.png')