# IMPORTS
import tensorflow as tf

# Build Model
print('*** BUILDING MODEL ***')

# Base Model Instantiation
base = tf.keras.applications.resnet.ResNet50(include_top = False, input_shape = (256, 256, 3), weights = 'imagenet')
base.trainable = False

# tf.keras.utils.plot_model(base, 'transferlearn_figs/rn50BASE.png')

# Regression Head
input_s = tf.keras.Input(shape = (256, 256, 3))
input_m = tf.keras.Input(shape = (256, 256, 3))

res_s_out = base(input_s, training = False)
res_m_out = base(input_m, training = False)

x = tf.keras.layers.Concatenate(axis = -1)([res_s_out, res_m_out])

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(2048, activation = 'relu', kernel_regularizer = 'l2')(x)
x = tf.keras.layers.Dense(1024, activation = 'relu', kernel_regularizer = 'l2')(x)
x = tf.keras.layers.Dense(512, activation = 'relu', kernel_regularizer = 'l2')(x)
x = tf.keras.layers.Dense(256, activation = 'relu', kernel_regularizer = 'l2')(x)

x = tf.keras.layers.Dense(3, name = 'output')(x)

model = tf.keras.Model(inputs = [input_s, input_m], outputs = x)
#model.summary()
tf.keras.utils.plot_model(model, 'transferlearn_figs/rn_regress.png', show_shapes = True, show_layer_names = False)
