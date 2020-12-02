# at EDA get Tokenizer info for deciding hyperparameters
from tensorflow.keras.preprocessing.text import Tokenizer

# add padding to a tokenized sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

# input all the layers we might use
from tensorflow.keras.layers import (Embedding, Dense, Conv1D, MaxPooling1D,
Dropout, Activation, Input, Flatten, Concatenate, LSTM, GlobalAveragePooling1D)

# do not use sequential to build model
from tensorflow.keras import Model

# need specify lr in optiizer
from tensorflow.keras import optimizers

# use categorical to transform to one-hot coding
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


class BuildModels(object):
    def __init__(self):
        pass

    def nn_model(self, word_index, MAX_SEQ_LEN):
        """
        Input is just padded question sequence, add Embedding layer transorfrom it into word vector and build up a sentence
        """
        # *************Hyperparameters************
        # max sequence/sentence length
        MAX_Q_SEQ_LEN = 400
        MAX_A_SEQ_LEN = 1000
        # what is the word dimentsion for a single, for examplee, "thank" will have
        EMBEDDING_DIM = 100
        # *************Hyperparameters************

        model = None

        input_layer_1 = Input(shape=(MAX_SEQ_LEN,), dtype='float32')
        embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                  output_dim=EMBEDDING_DIM,
                                  input_length=MAX_SEQ_LEN
                                  )(input_layer_1)
        pooling_layer_3 = GlobalAveragePooling1D()(embed_layer_2)
        dense_layer_4 = Dense(units=32, activation='relu')(pooling_layer_3)
        output_layer_5 = Dense(units=10, activation='softmax')(dense_layer_4)
        model = Model(inputs=input_layer_1, outputs=output_layer_5, name='nn_model')
        model.summary()
        return model
