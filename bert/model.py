
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

#run wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

class bertModel():

    def __init__(self, opts):

        self.bert = hub.KerasLayer(opts.model_loc, trainable=True)

class bertTrainer():
    def __init__(self, data, bert, opts):

        self.opts = opts
        self.data = data
        self.bert = bert.bert

    def initTrainer(self):

        self.init_optimizer()
        self.build_model()
        if self.opts.print_model:
            self.model.summary()

    def build_model(self):

        input_word_ids = Input(shape=(self.opts.max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(self.opts.max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(self.opts.max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = self.bert([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(self.opts.num_classes, activation='sigmoid')(clf_output)

        self.model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

        self.model.compile(self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def init_optimizer(self):

        train_data_size = len(self.data.X_train)
        steps_per_epoch = int(train_data_size / self.opts.batch_size)
        num_train_steps = steps_per_epoch * self.opts.epochs
        warmup_steps = int(self.opts.epochs * train_data_size * 0.1 / self.opts.batch_size)

        # creates an optimizer with learning rate schedule
        self.optimizer = nlp.optimization.create_optimizer(
            self.opts.lr, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    def train_model(self):

        self.initTrainer()

        checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

        self.train_history = self.model.fit(
            self.data.train_input, self.data.y_train,
            validation_split=self.opts.validation_split,
            epochs=self.opts.epochs,
            callbacks=[checkpoint],
            batch_size=self.opts.batch_size
        )
