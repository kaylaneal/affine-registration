import tensorflow as tf

# Build Model 
resnet = tf.keras.applications.resnet50.ResNet50(include_top = False, input_shape = (256, 256, 3))
resnet.trainable = False

inputs = tf.keras.Input(shape = (2, 256, 256, 3))

rout1 = resnet(inputs[:, 0], training = False)
rout2 = resnet(inputs[:, 1], training = False)

x = tf.keras.layers.Concatenate()([rout1, rout2])

x = tf.keras.layers.Conv2D(2048, kernel_size = 5)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(1024, kernel_size = 3)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(512, kernel_size = 1)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Conv2D(256, kernel_size = 1)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(3)(x)

model = tf.keras.Model(inputs, x)