# IMPORTS
import tensorflow as tf

# Build Transfer Model
print('*** BUILDING MODEL ***')

inputs = tf.keras.Input(shape = (2, 256, 256, 3), name = 'images')

# Base Model Instantiation
resnet = tf.keras.applications.resnet50.ResNet50(include_top = False, weights = 'imagenet', input_shape = (256, 256, 3))
resnet.trainable = False

# Base Model (processed) Output
rout1 = resnet(inputs[:, 0], training = False)
rout2 = resnet(inputs[:, 1], training = False)

rout1 = tf.keras.layers.GlobalAveragePooling2D()(rout1)
rout1 = tf.keras.layers.Dropout(0.4)(rout1)
rout1 = tf.keras.layers.Flatten()(rout1)

rout2 = tf.keras.layers.GlobalAveragePooling2D()(rout2)
rout2 = tf.keras.layers.Dropout(0.4)(rout2)
rout2 = tf.keras.layers.Flatten()(rout2)

x = tf.keras.layers.Concatenate()([rout1, rout2])

# Regression Head
'''
x = tf.keras.layers.Dense(2048, kernel_regularizer = 'l2')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dense(1024, kernel_regularizer = 'l2')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dense(512, kernel_regularizer = 'l2')(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dense(256, kernel_regularizer = 'l2')(x)
x = tf.keras.layers.LeakyReLU()(x)
'''
output = tf.keras.layers.Dense(3, name = 'output')(x)

model = tf.keras.Model(inputs = inputs, outputs = output)

# model.summary()
tf.keras.utils.plot_model(model, 'transfer_model/transferlearn_figs/rn_regress.png', show_shapes = True, show_layer_names = False)
