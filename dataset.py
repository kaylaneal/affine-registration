# IMPORTS
import os
import pandas as pd
from PIL import Image
import numpy as np
'''
JSON KEYS // DATAFRAME COL NAMES:
deltaTheta
deltaXviewport
deltaYviewport
a
x
y
staticImage
movingImage
'''

# Images read in at size (256, 256)
    # Image Array at shape (256, 256, 4)
    # As grayscale ('L') shape is (256, 256)

class PerfectPairDataset:
    def __init__(self, json):
        self.json_df = pd.read_json(json)
        self.path = os.path.dirname(json)
        self.static = {}
        self.moving = {}
        self.labels = {}

        self.add_pairs()
    
    def add_pairs(self):
        for idx, r in self.json_df.iterrows():
            self.static.update({
                idx : os.path.join(self.path, r['staticImage'])
            })
            self.moving.update({
                idx : os.path.join(self.path, r['movingImage'])
            })
            self.labels.update({
                            idx : [r['x'], r['y'], r['a']]
                            })
    
    def process_imgpairs(self):
        staticX, movingX = [], []

        for idx in self.static.values():
            s = Image.open(idx).convert('RGB')
            s = np.array(s)
            s = s / 255.

            staticX.append(s)

        for idx in self.moving.values():
            m = Image.open(idx).convert('RGB')
            m = np.array(m)
            m = m / 255.

            movingX.append(m)

        return staticX, movingX
    
    def process_labels(self):
        x, y, a = [], [], []

        for i in self.labels.values():
            x.append(i[0])
            y.append(i[1])
            a.append(i[2])

        x = np.array(x)
        y = np.array(y)
        a = np.array(a)
        
        x = ((x - np.min(x)) / (np.max(x) - np.min(x)))
        y = ((y - np.min(y)) / (np.max(y) - np.min(y)))
        a = ((a - np.min(a)) / (np.max(a) - np.min(a)))

        label = []
        for idx in range(len(x)):
            label.append([x[idx], y[idx], a[idx]])
        
        return label
    
def create_dataset(json_file):
    dataset = PerfectPairDataset(json_file)

    return dataset
