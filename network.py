# INPUTS
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential

## PARTIAL MODELS:
class Regression_Network(Model):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = Sequential([
            layers.Flatten(),
            layers.Dense(3)                # 256x256 to 3 variables
        ])
    
    def call(self, x):
        x = self.fc(x)
        return x

class Forward_Block(Model):
    def __init__(self, channels, pool = False):
        super(Forward_Block, self).__init__()
        
        self.pool = pool
        if self.pool:
            self.pool_layer = Sequential([
                layers.Conv2D(2*channels, kernel_size = 3, strides = 2, padding = 'same', kernel_regularizer = 'l2')
            ])
            self.layer = Sequential([
                layers.Conv2D(2*channels, kernel_size = 3, strides = 2, padding = 'same', kernel_regularizer = 'l2'),
                layers.BatchNormalization(),
                layers.PReLU(),
                layers.Conv2D(2*channels, kernel_size = 3, strides = 1, padding = 'same', kernel_regularizer = 'l2'),
                layers.BatchNormalization(),
                layers.PReLU(),
                layers.Dropout(0.2)
            ])
        else:
            self.layer = Sequential([
                layers.Conv2D(channels, kernel_size = 3, strides = 1, padding = 'same', kernel_regularizer = 'l2'),
                layers.BatchNormalization(),
                layers.PReLU(),
                layers.Conv2D(channels, kernel_size = 3, strides = 1, padding = 'same', kernel_regularizer = 'l2'),
                layers.BatchNormalization(),
                layers.PReLU(),
                layers.Dropout(0.2)
            ])

    def call(self, x):
        if self.pool:
            return self.pool_layer(x) + self.layer(x)
        else:
            return x + self.layer(x)
        
class Feature_Extractor(Model):
    def __init__(self):
        super(Feature_Extractor, self).__init__()

        self.input_layer = Sequential([
            layers.Conv2D(64, kernel_size = 7, strides = 2, padding = 'same', kernel_regularizer = 'l2')
        ])

        self.layer1 = Forward_Block(64, pool = True)
        self.layer2 = Forward_Block(128, pool = False)
        self.layer3 = Forward_Block(128, pool = True)
        self.layer4 = Forward_Block(256, pool = False)
        self.layer5 = Forward_Block(256, pool = True)
        self.layer6 = Forward_Block(512, pool = True)

        self.last_layer = Sequential([
            layers.Conv2D(512, kernel_size = 3, strides = 2, padding = 'same'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.Conv2D(256, kernel_size = 3, strides = 2, padding = 'same'),
            layers.BatchNormalization(),
            layers.PReLU(),
            layers.GlobalAveragePooling2D()
        ])

    def call(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.last_layer(x)
        return x
    
## AFFINE NETWORK
class Affine_Network(tf.keras.Model):
    def __init__(self):
        super(Affine_Network, self).__init__()

        self.feature_extractor = Feature_Extractor()
        self.regression_network = Regression_Network()
    
    def call(self, X):
        x = self.feature_extractor(X)
        x = self.regression_network(x)
        return x
