
from model import BiRNN
from options import options

import tensorflow as tf
import numpy as np

from train import train, train_step

options = options()
opts = options.parse()

biLSTM = BiRNN(opts)

#load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#make it so that the shape is [Batch, 1, 784] that can be fed into the LSTM module
x_train = x_train.reshape([-1, 1, 28 * 28]).astype(np.float32)
x_test = x_test.reshape([-1, 1, 28 * 28]).astype(np.float32)

#transform the input so that the values lie between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

#make the classes integers
y_train = tf.cast(y_train, tf.int64)
y_test = tf.cast(y_test, tf.int64)

#set the buffer / batch sizes and shuffle the data
BUFFER_SIZE = 60000
BATCH_SIZE = opts.batch_size

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(x_test.shape[0]).prefetch(1)

#define the error function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#define the optimizer
optimizer = tf.keras.optimizers.Adam(opts.lr, opts.beta1, opts.beta2)

#call the training function with training data
print("Train Data")
train(train_data, opts.epochs, biLSTM, optimizer)

#call the training function with test data
print("Test Data")
train(test_data, 1, biLSTM, optimizer, False)
