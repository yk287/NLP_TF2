
import tensorflow as tf
from tensorflow.keras import layers, Model

class BiRNN(Model):

    def __init__(self, opts):
        super(BiRNN, self).__init__()

        self.opts = opts

        lstm_fw = tf.keras.layers.LSTM(opts.lstmDim, input_shape = (None, 1, 784))
        lstm_bw = tf.keras.layers.LSTM(opts.lstmDim, input_shape = (None, 1, 784), go_backwards=True)

        self.biLSTM = tf.keras.layers.Bidirectional(lstm_fw, backward_layer= lstm_bw)
        self.dropout = tf.keras.layers.Dropout(self.opts.dropout)
        self.out = tf.keras.layers.Dense(self.opts.numClasses)

    def call(self, x, is_training=False):

        x = self.biLSTM(x)
        x = self.dropout(x)
        x = self.out(x)

        if not is_training:

            x = tf.nn.softmax(x)

        return x