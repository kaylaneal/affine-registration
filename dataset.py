# IMPORTS
import os
import pandas as pd

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
        self.pairs = {}
        self.pair_labels = {}

        self.add_pairs()
    
    def add_pairs(self):
        for idx, r in self.json_df.iterrows():
            self.pairs.update({
                idx : [os.path.join(self.path, r['staticImage']), os.path.join(self.path, r['movingImage'])]
            })
            self.pair_labels.update({
                            idx : [r['x'], r['y'], r['a']]
                            })

def create_dataset(json_file):
    dataset = PerfectPairDataset(json_file)

    return dataset
