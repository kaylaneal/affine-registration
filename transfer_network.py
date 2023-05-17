# IMPORTS
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input, Model, layers

# Build Model
print('*** BUILDING MODEL ***')

# Base Model Instantiation
base = ResNet50(include_top = False, input_shape = (256, 256, 3), weights = 'imagenet')
base.trainable = False
plot_model(base, 'transferlearn_figs/rn50BASE.png')

# Regression Head
input_s = Input(shape = (256, 256, 3))
input_m = Input(shape = (256, 256, 3))

res_s_out = base(input_s, training = False)
res_m_out = base(input_m, training = False)

x = layers.Concatenate(axis = -1)([res_s_out, res_m_out])
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(2048, activation = 'relu')(x)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation = 'relu')(x)
x = layers.Dense(256, activation = 'relu')(x)

x_out = layers.Dense(1, name = 'x_trans')(x)
y_out = layers.Dense(1, name = 'y_trans')(x)
theta_out = layers.Dense(1, name = 'rotation')(x)

model = Model(inputs = [input_s, input_m], outputs = [x_out, y_out, theta_out])
#model.summary()
plot_model(model, 'transferlearn_figs/rn_regress.png', show_shapes = True, show_layer_names = False)
