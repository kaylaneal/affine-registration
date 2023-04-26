# IMPORTS
import os
import pandas as pd
from PIL import Image
import numpy as np
import scipy.ndimage as nd
## LOCAL IMPORTS
import utils

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
        self.images = {}
        self.labels = {}
        self.xya = {}

        self.add_pairs()
    
    def add_pairs(self):
        for idx, r in self.json_df.iterrows():
            self.images.update({
                idx : [os.path.join(self.path, r['staticImage']), os.path.join(self.path, r['movingImage'])]
            })
            self.labels.update({
                            idx : [r['deltaTheta'], r['deltaXviewport'], r['deltaYviewport']]
                            })
            self.xya.update({
                idx : (r['x'], r['y'], r['a'])
            })
    
    '''
    def segment_mask(self):
        for idx, p in self.pairs.items():
            static = p[0][0]
            moving = p[0][1]
            # Segment Background
            static = np.expand_dims(static, (0, -1))                        # Segmentation expects shape of (B, H, W, C)
            moving = np.expand_dims(moving, (0, -1))                        # New Shape: (1, 512, 512, 1)
            #print(f'Static Shape {static.shape}\nMoving Shape {moving.shape}')

            seg_static, seg_moving = utils.segmentation(static, moving)
            self.masks.update(
                { idx : [static - seg_static, moving - seg_moving] } )
    '''
    def preprocess_data(self):
        for idx, p in self.images.items():
            static = Image.open(p[0]).convert('L') # with convert('L') arrays are shape (256, 256)
            moving = Image.open(p[1]).convert('L')

            # Normalize, Pad, Smooth, Resample:
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

            self.images.update({
                idx : [static, moving]
            })
    

def create_dataset(json_file):    
    dataset = PerfectPairDataset(json_file)
    return dataset



