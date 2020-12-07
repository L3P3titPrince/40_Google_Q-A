import numpy as np
from time import time
# see the loop progress
from tqdm import tqdm
# use hyperparameters
from class_31_hyperparameters import HyperParameters

class GloveVect(HyperParameters):
    def __init__(self):
        """
        All the hyperparameters need initialize in this section
        """
        HyperParameters.__init__(self)

    def glove_vect(self, word_index):
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
        start_time = time()

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
        # self.EMBEDDING_DIM = 50
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

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End glove_vect() with {} seconds".format(cost_time), "*" * 40, end='\n\n')
        return embedding_matrix, embedding_index

