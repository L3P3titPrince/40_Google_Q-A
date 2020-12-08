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
    def __init__(self):
        """
        Most of hyperparameters come from Hyperparameters class
        :return:
        """
        # state here again to make sure this class can inhereite parameters from partent class
        HyperParameters.__init__(self)

    def normal_cnn(self, word_index,  pretrain_matrix = None):
        """
        This very simple CNN and we use it as our CNN benchmark

        :return:
        """
        # if i don't initial them in here, there will be a "unbound loacl varble" error
        model = None
        # MAX_SEQ_LEN = None
        # OUTPUT_UNITS = None
        # OUTPUT_ACT = None
        if self.PART == 'q':
            MAX_SEQ_LEN = self.MAX_Q_SEQ_LEN
            # for question part caculation, our output will be six column numerical result, units = 6
            OUTPUT_UNITS = self.Q_OUTPUT_UNIT
        elif self.PART == 'a':
            MAX_SEQ_LEN = self.MAX_A_SEQ_LEN
            # for question part caculation, our output will be three column numerical result, units = 3
            OUTPUT_UNITS = self.A_OUTPUT_UNIT
        else:
            print(f"Please indicate you want embedding question part or answer part")

        if self.TYPE == 'num':
            # this is the final layer parameters. When we want to transform to numerical
            # if we want to use numerical, OUTPUT_UNIT will follow upper assign result
            # maybe we could change this activation function to sigmoid?
            OUTPUT_ACT = 'linear'
        elif self.TYPE == 'classify':
            # if we need use classify model, we need transfrom output layer Dense unit into ten classify
            OUTPUT_UNITS = 10
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
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        # filer =32, kernel_size =5. output will be (number of example=6709, MAX_SEQ_LEN-kneral_size+1 =400-5+1, filer)
        conv1d_layer_3 = Conv1D(filters=32, kernel_size=5, activation='relu', name='conv1D_layer_3')(embed_layer_2)
        pooling_layer_4 = MaxPooling1D(pool_size=5)(conv1d_layer_3)
        conv1d_layer_5 = Conv1D(filters=32, kernel_size=5, activation='relu', name='conv1D_layer_5')(pooling_layer_4)
        pooling_layer_6 = MaxPooling1D(pool_size=5)(conv1d_layer_5)
        flatten_layer_7 = Flatten()(pooling_layer_6)
        dense_layer_8 = Dense(units=128, activation='relu')(flatten_layer_7)
        output_layer_9 = Dense(OUTPUT_UNITS, activation=OUTPUT_ACT)(dense_layer_8)
        model = Model(inputs=input_layer_1, outputs=output_layer_9, name='Noram_CNN')
        model.summary()

        return model


    def n_gram_cnn(self,word_index, pretrain_matrix = None):
        """

        Argus:
        -----
        word_index:dictionary
            we have two word_index, q_word_index is coming from question part, a_word_index is coming from answer
            And we use word_index to decide we you question or answer parameters

        type:str
            We need use this to decide this is regression problem or classify problem

        pretrain_matrix:tensor.weight
            if we assign pre-trained matrix for this arguments, we will use pre-train embedding layers

        :return:model
        """
        model = None
        # if i don't initial them in here, there will be a "unbound loacl varble" error
        MAX_SEQ_LEN = None
        OUTPUT_UNITS = None
        OUTPUT_ACT = None
        if self.PART == 'q':
            MAX_SEQ_LEN = self.MAX_Q_SEQ_LEN
            # for
            OUTPUT_UNITS = self.Q_OUTPUT_UNIT
        elif self.PART == 'a':
            MAX_SEQ_LEN = self.MAX_A_SEQ_LEN
            OUTPUT_UNITS = self.A_OUTPUT_UNIT
        else:
            print(f"Please indicate you want embedding question part or answer part")

        if self.TYPE == 'num':
            # this is the final layer parameters. When we want to transform to numerical
            # if we want to use numerical, OUTPUT_UNIT will follow upper assign result
            # maybe we could change this activation function to sigmoid?
            OUTPUT_ACT = 'linear'
        elif self.TYPE == 'classify':
            # if we need use classify model, we need transfrom output layer Dense unit into ten classify
            OUTPUT_UNITS = 10
            OUTPUT_ACT = 'softmax'

        main_input = Input(shape = (MAX_SEQ_LEN,), dtype='int32', name = 'main_input')
        if (pretrain_matrix is not None):
            embed_layer = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      weights=[pretrain_matrix],
                                      input_length=MAX_SEQ_LEN,
                                      trainable=False,
                                      name="embedding_layer_pretrain"
                                      )(main_input)
            # then we use random initial embedding layer
        else:
            embed_layer = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      input_length=MAX_SEQ_LEN,
                                      name="embedding_layer_random"
                                      )(main_input)

        # define first unigram covlutional layer
        # filters maybe a hyperparameter we need to select
        # when kernel_size =1, it's a uni-gram model
        # embedding output dimension is (6079/NumofExample, 400/1000, 50), question =400, answer = 1000
        # (400 - stride) / kerneal_size + 1 = 400 and filter =64, so output shape is (batch_size, 400, 64)
        conv1d_1 = Conv1D(filters = 128, kernel_size = 1, activation='relu', name = 'conv_unigram')(embed_layer)
        # actulay, in pooling layer, we use max-pooling (check from paper)
        pool_1 = MaxPooling1D(pool_size = MAX_SEQ_LEN - 1 + 1, name = 'pool_unigram')(conv1d_1)
        # the pooling layer create output is the size(Num_of_Example, 1, filters) = (batch_size, 1, 128)
        # output of flatten is (64,)
        flat_1 = Flatten(name = 'flat_unigram')(pool_1)

        # create bigram convoluation structre
        conv1d_2 = Conv1D(filters = 128, kernel_size = 2, activation = 'relu', name = 'conv_bigram')(embed_layer)
        # pool_size = MAX_SEQ_LEN - kernel_size + 1
        pool_2 = MaxPooling1D(pool_size = MAX_SEQ_LEN -2 +1, name = 'pool_bigram')(conv1d_2)
        flat_2 = Flatten(name = "flat_bigram")(pool_2)

        # create trigram convoluation structure
        conv1d_3 = Conv1D(filters = 128, kernel_size = 2, activation='relu', name = 'conv_trigram')(embed_layer)
        # pooling layer
        pool_3 = MaxPooling1D(pool_size= MAX_SEQ_LEN - 3 + 1, name = 'pool_trigram')(conv1d_3)
        # flatten to one dimension, (batch_size, 64)
        flat_3 = Flatten(name = 'flat_trigram')(pool_3)

        # concatenate into one layer
        concat_layer_4 = Concatenate(name = 'concatenate_layer')([flat_1, flat_2, flat_3])
        # Add a dropout layer
        drop_layer_5 = Dropout(rate = 0.5, name = 'drop_layer')(concat_layer_4)
        # use dense create output layer
        output_layer_6 = Dense(units = OUTPUT_UNITS, activation= OUTPUT_ACT, name='output_layer')(drop_layer_5)
        # create model
        model = Model(inputs = main_input, outputs = output_layer_6, name = 'N_gram_CNN')
        model.summary()

        return model
