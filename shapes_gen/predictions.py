import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from dataset import ShapeDataset

model = tf.keras.models.load_model('shape_reg')

data = ShapeDataset('shape_dataset.csv')
data.process()

X = [p for p in data.pairs.values()]
X = np.array(X)
y = [l for l in data.labels.values()]
y = np.array(y)

preds = model.predict(X)

for i in range(4):
    print(f'IMAGE PAIR {i}')

    pa = int(data.denormalize_labels(preds[i][0], data.max_a, data.min_a))
    px = int(data.denormalize_labels(preds[i][1], data.max_x, data.min_x))
    py = int(data.denormalize_labels(preds[i][2], data.max_y, data.min_y))

    ta = data.denormalize_labels(y[i][0], data.max_a, data.min_a)
    tx = data.denormalize_labels(y[i][1], data.max_x, data.min_x)
    ty = data.denormalize_labels(y[i][2], data.max_y, data.min_y)

    print(f'Truth: [{ta}, {tx}, {ty}]')
    print(f'Predicted: [{pa}, {px}, {py}]')

    trans = Image.fromarray(X[i][1])
    trans = trans.rotate(angle = pa, translate = (px, py))

    plt.subplot(131)
    plt.imshow(X[i][0])
    plt.axis('off')
    plt.title('Static')
    plt.subplot(132)
    plt.imshow(X[i][1])
    plt.axis('off')
    plt.title('Moving')
    plt.subplot(133)
    plt.imshow(trans)
    plt.axis('off')
    plt.title('Transformed Moving')    
    plt.savefig(f'figures/imgpair{i}.png')

    print()

