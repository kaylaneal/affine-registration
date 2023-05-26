from init_datasets import testset
import tensorflow as tf

testset.process_data()
x = testset.pairs
y = testset.labels
model = tf.keras.models.load_model('transfer_model/transfer_affine')

preds = model.predict(x)

print(preds)
