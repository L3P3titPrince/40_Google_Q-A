# import need layer
from tensorflow.keras.layers import Embedding, Dense, Dropout,  Input, LSTM
from tensorflow.keras import Model

from class_31_hyperparameters import HyperParameters

class RNNModel(HyperParameters):
    """
    Contain LSTM / LSTM + ATTENTION

    """
    def __init__(self):
        """

        """
        HyperParameters.__init__(self)

    def lstm(self, word_index, pretrain_matrix, trainable = False):
        """

        :return:
        """
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

        print(MAX_SEQ_LEN, OUTPUT_ACT, OUTPUT_UNITS)
        model = None
        # input layer is fix, but embed_layer will change according to custom arguments
        input_layer_1 = Input(shape=(MAX_SEQ_LEN,), dtype='float32')
        # if we assign embedding matrix for arguments "pretrain_matrx", then we use pretrained Embedding
        if (pretrain_matrix is not None):
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      weights=[pretrain_matrix],
                                      input_length=MAX_SEQ_LEN,
                                      trainable=trainable,
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        # then we use random initial embedding layer
        else:
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      input_length=MAX_SEQ_LEN,
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        lstm_layer_3 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(embed_layer_2)
        output_layer_4 = Dense(OUTPUT_UNITS, activation=OUTPUT_ACT)(lstm_layer_3)
        model = Model(inputs = input_layer_1, outputs = output_layer_4, name='LSTM_Model')
        model.summary()

        return model