# most of data we will store in
import pandas as pd
import numpy as np

# use regex to extract text
import re
# using copy to duplicate
import copy
# corpus will be string format
import string
# the easiest way to get text is using bs4 to get only text
from bs4 import BeautifulSoup
#cacualte spearmen correcltion
from scipy import stats
# make picture
import matplotlib.pyplot as plt
# draw picture
import seaborn as sns


# at EDA get Tokenizer info for deciding hyperparameters
from tensorflow.keras.preprocessing.text import Tokenizer
# add padding to a tokenized sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences



# one of tokenize method
from sklearn.feature_extraction.text import TfidfVectorizer
# split data with random seed (37)
from sklearn.model_selection import train_test_split

# at EDA get Tokenizer info for deciding hyperparameters
from tensorflow.keras.preprocessing.text import Tokenizer
# add padding to a tokenized sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from class_31_hyperparameters import HyperParameters

class TokenizeData(HyperParameters):
    """
    Tokenize is only split sent
    """

    def __init__(self):
        HyperParameters.__init__(self)
    #     """
    #     """
    #     # using distribution to decide this parameters
    #     self.MAX_Q_SEQ_LEN = 400
    #     self.MAX_A_SEQ_LEN = 1000

    def tokenize_plot(self, question_cleaned_df, answer_cleaned_df):
        """
        decide how many words should be left for tokenzie() function

        Argus:
        -----
        question_cleaned_df
        """
        # do not set num_words at first time and to see how many unique words we have
        tokenizer_model = Tokenizer(split=' ', char_level=False, oov_token="<OOV>")
        # here is another trick, we need consider question and answer courpus into one unit
        # so we build a new big combination corpus,
        corpus_sum = question_cleaned_df['cleaned'] + answer_cleaned_df['cleaned']
        # use previous model to fit this large combination corpus
        tokenizer_model.fit_on_texts(corpus_sum)
        # get the word_index and word_count dictionary
        # word_index is the number corresponding to words by frequence. word_count is the sepcific words appeart times frequency
        word_index, word_count = tokenizer_model.word_index, tokenizer_model.word_counts
        print(f"we got unique {len(word_index)} words")
        MAX_WORD = 0
        for i in word_count.values():
            if i > 5:
                MAX_WORD += 1
        print(f"we have {MAX_WORD} words appear more than 5 times")

        # **************new tokenize process******************
        # due to bad performance, i remain all sentence and all word in this small dataset
        # MAX_WORD = 72070
        # start a new standard processs from begining, we use hyperparameter self.MAX_WORD not local MAX_WORD
        tokenizer_new = Tokenizer(num_words=self.MAX_WORD, split=' ', char_level=False, oov_token="<OOV>")
        # fit on combination corpus(question+answer). We can't just use one component to build tokenizer
        tokenizer_new.fit_on_texts(corpus_sum)
        # get new word_index and index_word. we use index_word(index:word) to restore orignial sentence by numerical sequence
        self.word_index, self.index_word = tokenizer_new.word_index, tokenizer_new.index_word
        # get question of sequence
        question_seq = tokenizer_new.texts_to_sequences(question_cleaned_df['cleaned'])
        # get sequence of answer
        answer_seq = tokenizer_new.texts_to_sequences(answer_cleaned_df['cleaned'])
        # get padded,
        self.question_padded = pad_sequences(question_seq, padding='post', maxlen=self.MAX_Q_SEQ_LEN, truncating='post')
        # using padded
        self.answer_padded = pad_sequences(answer_seq, padding='post', maxlen=self.MAX_A_SEQ_LEN, truncating='post')
        # **************new tokenize process******************
        question_cleaned_df['padded'] = list(self.question_padded)

        return self.question_padded, question_cleaned_df, self.answer_padded, self.word_index, self.index_word

    def sequence_to_text(self):
        """
        Use this function to convert padded sequnce back to text according to word_index
        """
        # create a empty list
        word_list = []
        for idx, i in enumerate(self.question_padded):
            # for every word in self.question_padded[0]=sentence, put it into a list
            words = np.array([self.index_word.get(word) for word in i])
            # insert into list
            word_list.append(words)
        # create a dictionary to build DataFrame
        dic = {"sequence_to_text": word_list}
        word_df = pd.DataFrame(dic)
        return word_df
