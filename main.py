from class_31_hyperparameters import HyperParameters
from class_32_import_data import ImportData
from class_33_clean_data import CleanData
from class_34_tokenize import TokenizeData
from class_35_label import LabelProcess
from class_36_eda import EdaData
from class_45_embeding import MultiEmbedding
from class_43_glove import GloveVect
from class_45_splist_complie import SplitAndCompile
from class_61_neuralnetwork import BuildModels
from class_62_cnn import CNNModel
from class_63_rnn import RNNModel
from class_69_save import SaveModelHistory

def main():
    """
    We use this function to call process one by one.
    """

    # ***********************import******************************
    import_class = ImportData()
    # df_train_raw = (6079,41) raw data, which just read from csv file without changing anything
    # X_q_train_df = (6079, 6) only contain ['qa_id', 'question_title', 'question_body', 'category', 'host','question']
    # X_a_train_df = (6079,4) contain ['qa_id', 'answer', 'category', 'host']
    # y_q_train_df = (6079,21) numerical labels
    # y_a_train_df = (6079,9) numerical labels
    (df_train_raw, X_q_train_df, X_a_train_df, y_q_train_df,
     y_a_train_df) = import_class.import_data("03_data/02_train.csv")
    df_test_raw, X_q_test_df, X_a_test_df, y_q_test_df, y_a_test_df = import_class.import_data("03_data/03_test.csv")

    # ***********************clean******************************
    clean_class = CleanData()
    q_train_cleaned_df = clean_class.clean_process(X_q_train_df, column_1='question')
    a_train_cleaned_df = clean_class.clean_process(X_a_train_df, column_1='answer')
    q_test_cleaned_df = clean_class.clean_process(X_q_test_df, column_1 ='question')
    a_test_cleaned_df = clean_class.clean_process(X_a_test_df, column_1='answer')

    # ***********************tokenize*****************************
    token_class = TokenizeData()
    # do not use '|', insteand we can use comma to next line and bracket to state they are together
    # input q_train_cleaned_dataframe and output still dataframe with new colunm[padded]
    # question_train part_padded  = (6078, 400)
    (q_train_padded, q_train_cleaned_df, a_train_padded, word_index,
     q_index_word) = token_class.tokenize_data(q_train_cleaned_df, a_train_cleaned_df)


    # ********Using manuually categorical*************
    #***********From this part, we only consider classification version************
    # y_q_train_df
    label_class = LabelProcess(y_q_train_df, y_a_train_df)
    y_q_label_df, y_a_label_df, q_feature_col, a_feature_col = label_class.num_label()
    y_q_classify_list, y_q_classify_dict, y_a_classify_list, y_a_classify_dict = label_class.classify_label()


    # ********************EDA******************************
    # eda_class = EdaData()
    # eda_class.question_plot(y_question_df)
    # eda_class.answer_plot(y_answer_df)
    # q_train_padded have shape (6079,100) can be used in fewer embedding

    # #*********************Embedding****THIS PART ONLY WORD AS API*******************

    glove_class = GloveVect()
    embedding_matrix, embedding_index = glove_class.glove_vect(word_index)
    embed_class = MultiEmbedding()
    #first transform q_train_padded
    q_glove_output_array, q_glove_embedding_layer= embed_class.glove_embedding(word_index,
                                                                               q_train_padded,
                                                                               embedding_matrix,
                                                                               part = 'q')
    a_glove_output_array, a_glove_embedding_layer= embed_class.glove_embedding(word_index,
                                                                               a_train_padded,
                                                                               embedding_matrix,
                                                                               part = 'a')
    q_random_output, q_random_embedding_layer = embed_class.random_embedding(word_index,
                                                                             q_train_padded,
                                                                              part = 'q')
    a_random_output, a_random_embedding_layer = embed_class.random_embedding(word_index,
                                                                             a_train_padded,
                                                                             part = 'a')


    #******************Models*******************************

    model_class = BuildModels()
    compile_class = SplitAndCompile()
    save_class = SaveModelHistory()
    #***************Random Embedding Normal Neural Network****************
    # nn_model = model_class.nn_model(word_index)
    #     history, model_2 = compile_fit(nn_model(word_index), X_q_train, X_q_val, y_q_train, y_q_val, loss_fun = 'mse', epoch_num=1)
    # history, model = compile_class.compile_fit(nn_model,
    #                                  X_q_train, X_q_val, y_q_train, y_q_val, loss_fun='categorical_crossentropy',
    #                                  epoch_num=3)

    # # ***************Pretrain Glove Normal Neural Network****************(each model should have its own model part)
    # pretrain_nn = model_class.nn_model(word_index, part='q', type='classify', pretrain_matrix=embedding_matrix)
    # history, model = compile_class.compile_fit(pretrain_nn,
    #                                                X_q_train, X_q_val, y_q_train, y_q_val,
    #                                                loss_fun='categorical_crossentropy',
    #                                                epoch_num=3)
    # history_classify_df = save_class.write_csv(history, model, str_input='Question_Glove_NN_20')

    # ***************Question Pretrain Gloave Normal CNN Classify*******************
    # cnn_class= CNNModel()
    # cnn_model_1 = cnn_class.normal_cnn(word_index, pretrain_matrix=embedding_matrix)
    # history, model = compile_class.compile_fit(cnn_model_1,
    #                                            X_q_train, X_q_val, y_q_train, y_q_val,
    #                                            loss_fun='categorical_crossentropy',
    #                                            epoch_num=10)
    # history_classify_df = save_class.write_csv(history, model, str_input='Question_Glove_Normal_CNN_10')

    # cnn_class = CNNModel()
    # cnn_model_2 = cnn_class.n_gram_cnn(word_index, pretrain_matrix=embedding_matrix)
    # history, model = compile_class.compile_fit(cnn_model_2, q_train_padded, a_train_padded, y_q_label_df, y_a_label_df,
    #                                            y_q_classify_list, y_q_classify_dict,
    #                                            y_a_classify_list, y_a_classify_dict,
    #                                            epoch_num=3)
    # history_classify_df = save_class.write_csv(history, model)

    rnn_class = RNNModel()
    lstm_model_1 = rnn_class.lstm(word_index, pretrain_matrix=embedding_matrix, trainable=True)
    history, model = compile_class.compile_fit(lstm_model_1, q_train_padded, a_train_padded, y_q_label_df, y_a_label_df,
                                               y_q_classify_list, y_q_classify_dict,
                                               y_a_classify_list, y_a_classify_dict,
                                               epoch_num=5)
    history_classify_df = save_class.write_csv(history, model)

    #************************test part*****************************
    # 1.tokenize
    # 2.fit model
    # 3.get plot and analysis result
    #*************************END***************************



    return (df_train_raw, X_q_train_df, X_a_train_df, y_q_train_df, y_a_train_df, a_train_cleaned_df,
            q_train_padded, q_train_cleaned_df, a_train_padded, word_index, q_index_word,
            y_q_label_df, y_a_label_df, q_feature_col, a_feature_col,
            y_q_classify_list, y_q_classify_dict, y_a_classify_list, y_a_classify_dict,
            history, model)


if __name__=="__main__":
    """
    """
    (df_train_raw, X_q_train_df, X_a_train_df, y_q_train_df, y_a_train_df, a_train_cleaned_df,
     q_train_padded, q_train_cleaned_df, a_train_padded, word_index, q_index_word,
     y_q_label_df, y_a_label_df, q_feature_col, a_feature_col,
     y_q_classify_list, y_q_classify_dict, y_a_classify_list, y_a_classify_dict,
     history, model)= main()
    print("over")
