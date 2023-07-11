import pandas as pd
import cv2
import numpy as np

class ShapeDataset:
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

        self.pairs = {}
        self.labels = {}

        self.build()

    def build(self):
        for idx, row in self.df.iterrows():
            self.pairs.update( { idx : [ row[0], row[1] ] } )               # [Target, Moving]
            self.labels.update( { idx : [ row[2], row[3], row[4] ] } )      # [Theta Tx, Ty]
    
    def normalize_labels(self, data):
        mind = min(data)
        maxd = max(data)

        norm = ( (data - mind) / (maxd - mind) )
        return norm, mind, maxd
    
    def denormalize_labels(self, data, max, min):
        return (data * (max - min) + min)
    
    def process_images(self):
        ppairs = []

        for pair in self.pairs.values():
            t = pair[0]
            m = pair[1]

            t = cv2.imread(t)
            t = np.array(t)
            m = cv2.imread(m)
            m = np.array(m)

            ppairs.append([t, m])
        
        for i in range(len(ppairs)):
            self.pairs.update( {i : ppairs[i] } )
    
    def process_labels(self):
        a, x, y = [], [], []

        for i in self.labels.values():
            a.append(i[0])
            x.append(i[1])
            y.append(i[2])
        
        a = np.array(a)
        x = np.array(x)
        y = np.array(y)

        a, self.min_a, self.max_a = self.normalize_labels(a)
        x, self.min_x, self.max_x = self.normalize_labels(x)
        y, self.min_y, self.max_y = self.normalize_labels(y)

        for idx in range(len(x)):
            self.labels.update( { idx : [ a[idx], x[idx], y[idx] ] } )
    
    def process(self):
        self.process_images()
        self.process_labels()
