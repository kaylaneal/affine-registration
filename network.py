# IMPORTS
import tensorflow as tf
from tensorflow.keras import layers

def Feature_Extractor(x, channels):

    x = layers.Conv2D(channels, kernel_size = 3, strides = 2,
                      padding = 'same', kernel_regularizer = 'l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(channels, kernel_size = 3, strides = 2,
                      padding = 'same', kernel_regularizer = 'l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    return x

def Final_FELayer(x):

    x = layers.Conv2D(512, kernel_size = 3, strides = 2, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(256, kernel_size = 3, strides = 2, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    return x

def get_model(static_input, moving_input):
    
    s = layers.Conv2D(64, kernel_size = 7, strides = 2,
                      padding = 'same', kernel_regularizer = 'l2', activation = 'relu')(static_input)
    s = Feature_Extractor(s, 64)
    s = Feature_Extractor(s, 128)
    s = Feature_Extractor(s, 128)
    s = Feature_Extractor(s, 256)
    s = Feature_Extractor(s, 256)
    s = Feature_Extractor(s, 512)
    s = Final_FELayer(s)

    m = layers.Conv2D(64, kernel_size = 7, strides = 2,
                      padding = 'same', kernel_regularizer = 'l2', activation = 'relu')(moving_input)
    m = Feature_Extractor(m, 64)
    m = Feature_Extractor(m, 128)
    m = Feature_Extractor(m, 128)
    m = Feature_Extractor(m, 256)
    m = Feature_Extractor(m, 256)
    m = Feature_Extractor(m, 512)
    m = Final_FELayer(m)

    x = layers.Concatenate(axis = -1)([s, m])
    x = layers.Dense(3, activation = 'linear')(x)

    model = tf.keras.Model(inputs = [static_input, moving_input], outputs = x)
    return model
