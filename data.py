

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

#train test split
from sklearn.model_selection import train_test_split

class dataloader():

    def __init__(self, bert, opts):

        self.opts = opts
        self.bert = bert.bert

        self.preprocess()

    def init_tokenizer(self):

        vocab_file = self.bert.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    def read_data(self):

        self.data = pd.read_csv(self.opts.file_name)
        self.data['sentiment'] = np.where(self.data['sentiment'] == 'positive', 1, 0)

    def train_test_split(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['review'], self.data['sentiment'], test_size=self.opts.test_split,
                                                            random_state=self.opts.random_seed)

    def encode(self):

        self.train_input = self.bert_encode(self.X_train, self.tokenizer, self.opts.max_len)
        self.test_input = self.bert_encode(self.X_test, self.tokenizer, self.opts.max_len)

    def bert_encode(self, texts, tokenizer, max_len=512):

        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = tokenizer.tokenize(text)

            text = text[:max_len - 2]  # -2 to account for CLS and SEP
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)

            tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    def preprocess(self):

        self.init_tokenizer()
        self.read_data()
        self.train_test_split()
        self.encode()

