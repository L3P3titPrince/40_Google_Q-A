# import need layer
from tensorflow.keras.layers import Embedding, Dense, Dropout,  Input, LSTM, GRU
from tensorflow.keras import Model
# draw the structure of this model
from tensorflow.keras.utils import model_to_dot, plot_model
# dispaly structure and saved picture
from IPython.display import Image,display


from class_31_hyperparameters import HyperParameters

class RNNModel(HyperParameters):
    """
    Contain LSTM / LSTM + ATTENTION

    """
    def __init__(self):
        """

        """
        HyperParameters.__init__(self)

    def lstm(self, word_index, pretrain_matrix):
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

        # print(MAX_SEQ_LEN, OUTPUT_ACT, OUTPUT_UNITS)
        model = None
        # input layer is fix, but embed_layer will change according to custom arguments
        input_layer_1 = Input(shape=(MAX_SEQ_LEN,), dtype='float32')
        # if we assign embedding matrix for arguments "pretrain_matrx", then we use pretrained Embedding
        if self.PRETRINA_MATIRX:
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      weights=[pretrain_matrix],
                                      input_length=MAX_SEQ_LEN,
                                      trainable=self.TRAIN_BOOLEAN,
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        # then we use random initial embedding layer
        else:
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      input_length=MAX_SEQ_LEN,
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        lstm_layer_3 = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embed_layer_2)
        dense_layer_4 = Dense(64, activation='relu')(lstm_layer_3)
        dense_layer_5 = Dense(32, activation='relu')(dense_layer_4)
        output_layer_6 = Dense(OUTPUT_UNITS, activation=OUTPUT_ACT)(dense_layer_5)
        model = Model(inputs = input_layer_1, outputs = output_layer_6, name='LSTM_Model')
        model.summary()

        return model


    def gru(self, word_index, pretrain_matrix):
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

        # print(MAX_SEQ_LEN, OUTPUT_ACT, OUTPUT_UNITS)
        model = None
        # input layer is fix, but embed_layer will change according to custom arguments
        input_layer_1 = Input(shape=(MAX_SEQ_LEN,), dtype='float32')
        # if we assign embedding matrix for arguments "pretrain_matrx", then we use pretrained Embedding
        if self.PRETRINA_MATIRX:
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      weights=[pretrain_matrix],
                                      input_length=MAX_SEQ_LEN,
                                      trainable=self.TRAIN_BOOLEAN,
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        # then we use random initial embedding layer
        else:
            embed_layer_2 = Embedding(input_dim=len(word_index) + 1,
                                      output_dim=self.EMBEDDING_DIM,
                                      input_length=MAX_SEQ_LEN,
                                      name="embedding_layer_2"
                                      )(input_layer_1)
        gru_layer_3 = GRU(128, dropout=0.2, recurrent_dropout=0.2)(embed_layer_2)
        dense_layer_4 = Dense(64, activation='relu')(gru_layer_3)
        dense_layer_5 = Dense(32, activation='relu')(dense_layer_4)
        output_layer_6 = Dense(OUTPUT_UNITS, activation=OUTPUT_ACT)(dense_layer_5)
        model = Model(inputs=input_layer_1, outputs=output_layer_6, name='GRU_Model')

        model.summary()

    # dot_img_file = '04_images/23_LSTM_model.png'
    # plot_model(model, to_file=dot_img_file, show_shapes=True)
    # display(Image(filename='04_images/23_LSTM_model.png'))

        return model