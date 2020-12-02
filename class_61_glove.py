import numpy as np
import time

# see the loop progress
from tqdm import tqdm


class GloveVect(object):
    def __init__(self):
        """
        All the hyperparameters need initialize in this section
        """
        self.PATH = r"D:\Downloads\glove.6B\glove.6B.50d.txt"
        self.EMBEDDING_DIM = 50

    def glove_vect(self, word_index, df):
        """
        Argus:
        ------
        word_index:dict
            unique word dictionary from tokenize_data(). For this example, word_index contain a dictionary from 27014 words to a number
            which number represent frequence and ordered from most ferquency to least frequency

        Returns:
        --------
        """
        print("*" * 50, "Start glove_vect() process", "*" * 50)
        start_time = time.time()

        # key = words;  values = word vector
        embedding_index = {}
        # split by line and first word in line is the word represented, following 50 numbers is pre-trained word vector
        f = open(self.PATH, 'r', encoding='utf-8')
        # glove store in a line which contain values and word splited by whitespace
        for line in f:
            values = line.split()
            # first value is word
            word = values[0]
            # next result are glove embedding vector
            coefs = np.asarray(values[1:], dtype='float32')
            # embeddings_index will be a dict representing relationship between word and its pretrained 50 dimension vector
            embedding_index[word] = coefs
        f.close()

        # this is depended on which GLOVE was choosed
        self.EMBEDDING_DIM = 50
        # create a empty matrix to filled in glove, this matrix will be a 27015(27014+1) by 50 dimension matrix
        embedding_matrix = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))
        # take the key=word and value=i to iterate
        for word, i in tqdm(word_index.items()):
            # get the corresponding 50d vectro of "word" , for example 'the' (50,) vector
            embedding_vector = embedding_index.get(word)
            # if we can find word in GLOVE, he will have a None-Zero vector.
            # if we didn't find, this row/word will be np.zero
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        cost_time = round((time.time() - start_time), 4)
        print("*" * 40, "End glove_vect() with {} seconds".format(cost_time), "*" * 40, end='\n\n')
        return embedding_matrix, embedding_index

def glove_embedding(word_index, padded, embedding_matrix):
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
        print("*" * 50, "Start embedding process", "*" * 50)
        start_time = time.time()
        # max sequence/sentence length is 100
        MAX_SEQ_LEN = 500
        EMBEDDING_DIM = 50
        sequence_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQ_LEN,
                                    trainable=False)
        # (number of sample, MAX_SEQ_LEN, EMBEDING_DIM)
        output = embedding_layer(padded)

        cost_time = round((time.time() - start_time), 4)
        print("*" * 40, "End embedding() with {} seconds".format(cost_time), "*" * 40, end='\n\n')

        return output, embedding_layer