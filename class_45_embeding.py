# each step we need moniter its processing progress and running time
from time import time
# We can not isolate using Embeding, we  need first get intput a Input() layer and then Embeeding
from tensorflow.keras.layers import Embedding, Input
# need use this model to initial embedding layer
from tensorflow.keras import Model
# Inheriest some constant from hyerparameters
from class_31_hyperparameters import HyperParameters

class MultiEmbedding(HyperParameters):
    """
    For this embedding class, i will standardize a output content and shape.
    In put will be
    word_index: dictionary
        the pair for {word:index}
    padded

    """


    def __init__(self):
        """
        inhereit from HyperParameters Class
        :return:
        """
        # super(MultiEmbedding,self).__init__()
        HyperParameters.__init__(self)


    def glove_embedding(self, word_index, padded, embedding_matrix, part='q'):
        """
        Typically method is transform word vector first and find cooresponding word in a sentcen, using word vector to concatenate setence vector
        But i can't find a tokenize() function to using seperate dictionary to complete tokenziation process
        Embedding can do that. When you set trainale=False, the weights, embedding_matrix, will be change and mapping word vector by word

        Argus:
        -----
        word_index:dictionary
            provide a index <-> word mapping table

        padded:array
            (number of example, MAX_SEQ_LEN)

        embedding_matrix:dictionary
            provide a word <-> word vector mapping table
        """
        print("*" * 50, "Start Glove embedding process", "*" * 50)
        start_time = time()


        MAX_SEQ_LEN = None
        if part == 'q':
            MAX_SEQ_LEN = self.MAX_Q_SEQ_LEN
        elif part == 'a':
            MAX_SEQ_LEN = self.MAX_A_SEQ_LEN
        else:
            print(f"Please indicate you want embedding question part or answer part")


        input_layer = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
        embedding_layer = Embedding(input_dim = len(word_index) + 1,
                                    output_dim = self.EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQ_LEN,
                                    trainable=False)(input_layer)
        # (number of sample, MAX_SEQ_LEN, EMBEDING_DIM)
        model = Model(inputs=input_layer, outputs=embedding_layer)
        model.compile('rmsprop', 'mse')
        output_array = model.predict(padded)

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End Glove embedding() with {} seconds".format(cost_time), "*" * 40, end='\n\n')

        return output_array, embedding_layer

    def random_embedding(self, word_index, padded, part='q'):
        """
        This funciont is just random initialize word vectors by keras.embedding


        Argus:
        ------
        :param word_index:coming from tokenize process
        :param padded: coming from tokenize function

        part:string
            (q) represent question part, then we use hyperparameters of question
            (a) represent answer part, then we use answer question
        Returns:
        -------
        output_array:numpy.ndarray
            This is embedding version lookup, it transform tokenize lookup table into word vector lookup table
            I.e. input is tokenize document have shape (5000, 1000)
            5000 is sentences number, 1000 is MAX_SEQ_LEN (because padded to same length)
            output will be (5000, 1000, 100) if EMBEDDING_DIM = 100

        embedding_layer:Tensor

        """
        print("*" * 50, "Start Random embedding process", "*" * 50)
        start_time = time()

        MAX_SEQ_LEN = None
        # global MAX_SEQ_LEN
        if part == 'q':
            MAX_SEQ_LEN = self.MAX_Q_SEQ_LEN
        elif part == 'a':
            MAX_SEQ_LEN = self.MAX_A_SEQ_LEN
        else:
            print(f"Please indicate you want embedding question part or answer part")


        # this is the max lenght of sentence, typically is the column length of DataFrame
        input_layer = Input(shape = (MAX_SEQ_LEN,), dtype = 'int32')
        # first argu is the max word serial number from tokenize function,
        # i.e. 100001 menas reamin most 10000 frequency
        embedding_layer = Embedding(input_dim = len(word_index) + 1,
                                   output_dim = self.EMBEDDING_DIM,
                                   input_length = MAX_SEQ_LEN)(input_layer)
        model = Model(inputs = input_layer, outputs = embedding_layer)
        model.compile('rmsprop', 'mse')
        output_array = model.predict(padded)

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End Random embedding() with {} seconds".format(cost_time), "*" * 40, end='\n\n')


        return output_array, embedding_layer