# most of data we will store in
import pandas as pd
import numpy as np

# use regex to extract text
import re
# using copy to duplicate
import copy
# recording each step runing time
import time
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


# one of tokenize method
from sklearn.feature_extraction.text import TfidfVectorizer
# split data with random seed (37)
from sklearn.model_selection import train_test_split

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


from class_41_preprocessdata import PreprocessData
from class_42_cleandata import CleanData
from class_43_tokenize import TokenizeData
from class_44_label import LabelProcess
from class_51_eda import EdaData
from class_68_splist import SplitData
from class_81_model import BuildModels
from class_89_complitfit import CompileFit

def main():
    """
    We use this function to call process one by one.
    """
    pre = PreprocessData()
    df_q_train_raw, X_question_df, X_answer_df, y_question_df, y_answer_df = pre.import_data("03_data/02_train.csv")
    # df_test_raw, X_q_test_df, X_a_test_df, y_q_test_df, y_a_test_df = pre.import_data("03_data/03_test.csv")

    clean = CleanData()
    question_cleaned_df = clean.clean_process(X_question_df, column_1='question')
    answer_cleaned_df = clean.clean_process(X_answer_df, column_1='answer')
    #     q_test_cleaned_df = clean.clean_process(X_q_test_df, column_1 ='question')
    #     a_test_cleaned_df = clean.clean_process(X_a_test_df, column_1='answer')

    eda_class = EdaData()
    # eda_class.question_plot(y_question_df)
    # question_padded have shape (6079,100) can be used in fewer embedding

    token_class = TokenizeData()
    # do not use '|', insteand we can use comma to next line and bracket to state they are together
    (question_padded, question_cleaned_df, answer_padded, word_index,
     index_word) = token_class.tokenize_plot(question_cleaned_df, answer_cleaned_df)
    # get question label
    #     y_q_label_df, y_a_label_df = token_class.label_feature(y_question_df, y_answer_df)

    # ********Using manuually categorical*************
    label_class = LabelProcess(y_question_df, y_answer_df)
    y_a_label_df = label_class.manual_calssify(y_answer_df.iloc[:, 0])

    #     q_test_padded, q_test_cleaned_df = eda_class.tokenize_plot(q_test_cleaned_df, a_test_cleaned_df)
    #     # get question label
    #     y_label_test_df = eda_class.label_feature(y_q_test_df)



    split_class = SplitData()
    # question part and answer part will be seperately split
    # X_q_train, X_q_val, y_q_train, y_q_val = SplitData(question_padded, y_q_label_df, test_size=0.2)
    X_a_train, X_a_val, y_a_train, y_a_val = split_class.split_data(answer_padded, y_a_label_df, test_size=0.2)




    model_class = BuildModels()
    nn_model = model_class.nn_model(word_index, MAX_SEQ_LEN = 1000)

    compile_class = CompileFit()
    #     history, model_2 = compile_fit(nn_model(word_index), X_q_train, X_q_val, y_q_train, y_q_val, loss_fun = 'mse', epoch_num=1)
    history_a, model_a = compile_class.compile_fit(nn_model,
                                     X_a_train, X_a_val, y_a_train, y_a_val, loss_fun='categorical_crossentropy',
                                     epoch_num=3)

    return (X_question_df, X_answer_df, y_question_df, y_answer_df,
            question_padded, question_cleaned_df, answer_cleaned_df,
            X_a_train, X_a_val, y_a_train, y_a_val, history_a, model_a, word_index)


if __name__=="__main__":
    (X_question_df, X_answer_df, y_question_df, y_answer_df,
     question_padded, question_cleaned_df, answer_cleaned_df,
     X_a_train, X_a_val, y_a_train, y_a_val, history_a, model_a, word_index) = main()
    print("over")
