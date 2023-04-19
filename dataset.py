# IMPORTS
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
    def __init__(self, json_df:pd.DataFrame):
        assert json_df is not None, 'requires JSON!'
        self.json_df = json_df
        self.pairs = {}
        self.pair_labels = {}
        self.xya = {}

        self.add_pairs()
    
    def add_pairs(self):
        for idx, r in self.json_df.iterrows():
            self.pairs.update({
                idx : [r['staticImage'], r['movingImage']]
            })
            self.pair_labels.update({
                            idx : [r['deltaTheta'], r['deltaXviewport'], r['deltaYviewport']]
                            })
            self.xya.update({
                idx : (r['x'], r['y'], r['a'])
            })


def create_dataset(json_file):
    dataset_df = pd.read_json(json_file)
    
    dataset = PerfectPairDataset(dataset_df)

    return dataset
