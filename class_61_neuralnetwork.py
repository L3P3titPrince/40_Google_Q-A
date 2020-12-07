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

# inhere hyperparameters
from class_31_hyperparameters import HyperParameters


class BuildModels(HyperParameters):
    def __init__(self):
        # we need state here again to make sure this class can inhereite parameters from partent class
        HyperParameters.__init__(self)


    def nn_model(self, word_index, part='q', type='classify', pretrain_matrix = None):
        """
        Input is just padded question sequence, add Embedding layer transorfrom
        it into word vector and build up a sentence

        Argus:
        -----
        part:str
            'q' means input is question part, we need use question hyperparameters
            'a' means input is answer part, we need use answer hyperparameters

        pretrain:str
            'Gloave' means we use glove pretrain
            'Random' means we use random initial Embedding
        """

        # we use this way to decide recall which hyperparameters
        MAX_SEQ_LEN = None
        OUTPUT_UNIT = None
        OUTPUT_ACT = None
        if part == 'q':
            MAX_SEQ_LEN = self.MAX_Q_SEQ_LEN
            OUTPUT_UNIT = self.Q_OUTPUT_UNIT
        elif part == 'a':
            MAX_SEQ_LEN = self.MAX_A_SEQ_LEN
            OUTPUT_UNIT = self.A_OUTPUT_UNIT
        else:
            print(f"Please indicate you want embedding question part or answer part")

        if self.TYPE == 'num':
            OUTPUT_ACT = 'linear'
        elif self.TYPE == 'classify':
            OUTPUT_UNIT = 10
            OUTPUT_ACT = 'softmax'

        model = None
        # input layer is fix, but embed_layer will change according to custom arguments
        input_layer_1 = Input(shape=(MAX_SEQ_LEN,), dtype='float32')
        # if we assign embedding matrix for arguments "pretrain_matrx", then we use pretrained Embedding
        if (pretrain_matrix is not None):
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                          output_dim=self.EMBEDDING_DIM,
                                          weights=[pretrain_matrix],
                                          input_length=MAX_SEQ_LEN,
                                          trainable=False,
                                          name="embedding_layer_2"
                                          )(input_layer_1)
        # then we use random initial embedding layer
        else:
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      input_length=MAX_SEQ_LEN,
                                      name = "embedding_layer_2"
                                      )(input_layer_1)
        pooling_layer_3 = GlobalAveragePooling1D()(embed_layer_2)
        dense_layer_4 = Dense(units=32, activation='relu')(pooling_layer_3)
        output_layer_5 = Dense(units=OUTPUT_UNIT, activation=OUTPUT_ACT)(dense_layer_4)
        model = Model(inputs=input_layer_1, outputs=output_layer_5, name='nn_model')
        model.summary()
        return model



    # def test_model(self, X_train, y_train, embedding_layer, part='q'):
    #     """
    #
    #     :return:
    #     """
    #     # we use this way to decide recall which hyperparameters
    #     # global MAX_SEQ_LEN
    #     if part == 'q':
    #         MAX_SEQ_LEN = self.MAX_Q_SEQ_LEN
    #         OUTPUT_UNIT = self.Q_OUTPUT_UNIT
    #     elif part == 'a':
    #         MAX_SEQ_LEN = self.MAX_A_SEQ_LEN
    #         OUTPUT_UNIT = self.A_OUTPUT_UNIT
    #     else:
    #         print(f"Please indicate you want embedding question part or answer part")
    #
    #     input_layer_1 = Input(shape=(1000, 100,))
    #     pooling_layer_3 = GlobalAveragePooling1D()(input_layer_1)
    #     dense_layer_4 = Dense(units=32, activation='relu')(pooling_layer_3)
    #     output_layer_5 = Dense(units=10, activation='softmax')(dense_layer_4)
    #     model = Model(inputs=input_layer_1, outputs=output_layer_5, name='nn_model')
    #     model.summary()
    #
    #     model = Model(pooling_layer_3, output_layer_5)
    #     # model.compile(loss='categorical_crossentropy',
    #     #               optimizer='rmsprop',
    #     #               metrics=['acc'])
    #     # model.fit(X_train, y_train, validation_data=(x_val, y_val),
    #     #           epochs=2, batch_size=128)
    #
    #     return model