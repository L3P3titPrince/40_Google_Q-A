from class_31_hyperparameters import HyperParameters

from tensorflow.keras.layers import (Embedding, Dense, Conv1D, MaxPooling1D,
                                     Dropout, Activation, Input, Flatten, Concatenate, LSTM, GlobalAveragePooling1D)
from tensorflow.keras import Model

class CNNModel(HyperParameters):
    """
    A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification
    https://arxiv.org/abs/1510.03820
    Convolutional Neural Networks for Sentence Classification https://arxiv.org/abs/1408.5882
    This is new way to using cnn to complete classification
    
    In one class, i will use different complie to 

    """
    def __init__self(self):
        """
        Most of hyperparameters come from Hyperparameters class
        :return:
        """
        # state here again to make sure this class can inhereite parameters from partent class
        HyperParameters.__init__(self)

    def normal_cnn(self, word_index, part = 'q', type = 'classify', pretrain_matrix = None):
        """
        This very simple CNN and we use it as our CNN benchmark

        :return:
        """
        # if i don't initial them in here, there will be a "unbound loacl varble" error
        MAX_SEQ_LEN = None
        output_units = None
        output_act = None
        if part == 'q':
            MAX_SEQ_LEN = self.MAX_Q_SEQ_LEN
        elif part == 'a':
            MAX_SEQ_LEN = self.MAX_A_SEQ_LEN
        else:
            print(f"Please indicate you want embedding question part or answer part")
        if type == 'num':
            output_units = 1
            output_act = 'linear'
        elif type == 'classify':
            output_units = 10
            output_act = 'softmax'

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
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        # filer =32, kernel_size =5. output will be (number of example=6709, MAX_SEQ_LEN-kneral_size+1 =400-5+1, filer)
        conv1d_layer_3 = Conv1D(filters=32, kernel_size=5, activation='relu', name='conv1D_layer_3')(embed_layer_2)
        pooling_layer_4 = MaxPooling1D(pool_size=5)(conv1d_layer_3)
        conv1d_layer_5 = Conv1D(filters=32, kernel_size=5, activation='relu', name='conv1D_layer_5')(pooling_layer_4)
        pooling_layer_6 = MaxPooling1D(pool_size=5)(conv1d_layer_5)
        flatten_layer_7 = Flatten()(pooling_layer_6)
        dense_layer_8 = Dense(units=128, activation='relu')(flatten_layer_7)
        output_layer_9 = Dense(output_units, activation=output_act)(dense_layer_8)
        model = Model(inputs=input_layer_1, outputs=output_layer_9, name='CNN_Model')
        model.summary()


    def n_gram_cnn(self,word_index, type = 'classify', pretrain_matrix = None):
        """

        Argus:
        -----
        word_index:dictionary
            we have two word_index, q_word_index is coming from question part, a_word_index is coming from answer
            And we use word_index to decide we you question or answer parameters

        type:str
            We need use this to judge this is regression problem or classify problem

        pretrain_matrix:tensor.weight
            if we assign
        :return:
        """
