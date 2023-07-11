# IMPORTS
import matplotlib.pyplot as plt
import tensorflow as tf

## LOCAL IMPORTS
from transfer_network import resnet
from transfer_affine import train_x, train_y, valid_x, valid_y

## FINE TUNING
print('** FINE TUNING **')

model = tf.keras.models.load_model('transfer_model/transfer_affine')
resnet.trainable = True
model.compile(optimizer = tf.keras.optimizers.Adam(1e-5), loss = 'mse', metrics = ['mae', 'accuracy'])

t = 0
for layer in model.layers:
    if layer.trainable == True:
        t += 1
print(f'{t}/{len(model.layers)} layers are trainable.')

ft_hist = model.fit(train_x, train_y,
          validation_data = (valid_x, valid_y),
          batch_size = 8, epochs = 30,
          callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)])

print('*** ANALYZING TUNED RESULTS ***')

plt.plot(ft_hist.history['loss'])
plt.plot(ft_hist.history['val_loss'])
plt.title('Fine Tune MSE Loss Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Squared Error')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('transfer_model/transferlearn_figs/ft_loss.png')
plt.clf()

plt.plot(ft_hist.history['mae'])
plt.plot(ft_hist.history['val_mae'])
plt.title('Fine Tune MAE Metrics Curve')
plt.xlabel('epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.savefig('transfer_model/transferlearn_figs/ft_mae.png')
plt.clf()

tf.keras.models.save_model(model, 'transfer_model/transfer_affine')