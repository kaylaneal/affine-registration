# IMPORTS
import os
import pandas as pd
import numpy as np
from PIL import Image

'''
JSON KEYS // DATAFRAME COL NAMES:
a   x   y
staticImage     movingImage
'''

# Images read in at size (256, 256)
    # Image Array at shape (256, 256, 4)
    # RGB shape is (256, 256, 3)

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
        self.pairs, p = [], []

        for i in range(len(self.static.keys())):
            s = self.static.get(i)
            m = self.moving.get(i)

            s = Image.open(s).convert('RGB')
            s = np.array(s)
            m = Image.open(m).convert('RGB')
            m = np.array(m)

            #s = s / 255.
            #m = m / 255.

            p.append(np.stack([s, m], axis = 0))
        
        self.pairs = np.stack([ps for ps in p], axis = 0)
    
    def process_labels(self):
        x, y, a = [], [], []

        for i in self.labels.values():
            x.append(i[0])
            y.append(i[1])
            a.append(i[2])

        x = np.array(x)
        y = np.array(y)
        a = np.array(a)
        
        x, self.min_x, self.max_x = normalize(x)
        y, self.min_y, self.max_y = normalize(y)
        a, self.min_a, self.max_a = normalize(a)

        label = []
        for idx in range(len(x)):
            label.append(np.stack([x[idx], y[idx], a[idx]], axis = 0))
        
        self.labels = np.stack([l for l in label], axis = 0)
    
    def process_data(self):
        self.process_imgpairs()
        self.process_labels()

    
def create_dataset(json_file):
    dataset = PerfectPairDataset(json_file)

    return dataset

def normalize(data):
    min_d = float(min(data))
    max_d = float(max(data))

    norm = ((data - min_d) / (max_d - min_d))

    return norm, min_d, max_d

def denormalize(norm, min, max):
    denorm = (norm * (max - min) + min)

    return denorm