from class_31_hyperparameters import HyperParameters
from class_32_import_data import ImportData
from class_33_clean_data import CleanData
from class_34_tokenize import TokenizeData
from class_35_label import LabelProcess
from class_36_eda import EdaData
from class_42_embeding import MultiEmbedding
from class_43_glove import GloveVect
from class_45_splist import SplitData
from class_61_neuralnetwork import BuildModels
from class_65_complitfit import CompileFit
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
    (q_train_padded, q_train_cleaned_df, a_train_padded, q_word_index,
     q_index_word) = token_class.tokenize_plot(q_train_cleaned_df, a_train_cleaned_df)


    # ********Using manuually categorical*************
    #***********From this part, we only consider classification version************
    # y_q_train_df
    label_class = LabelProcess(y_q_train_df, y_a_train_df)
    y_q_label_df, y_a_label_df, feature_col_q, feature_col_a = label_class.num_label()
    y_q_table_0 = label_class.manual_calssify(y_q_label_df.iloc[:, 0])
    # y_a_label_df = label_class.manual_calssify(y_answer_df.iloc[:, 0])

    #     q_test_padded, q_test_cleaned_df = eda_class.tokenize_plot(q_test_cleaned_df, a_test_cleaned_df)
    #     # get question label
    #     y_label_test_df = eda_class.label_feature(y_q_test_df)

    # ********************EDA******************************
    # eda_class = EdaData()
    # eda_class.question_plot(y_question_df)
    # eda_class.answer_plot(y_answer_df)
    # question_padded have shape (6079,100) can be used in fewer embedding

    # #*********************Embedding****THIS PART NOT WORKING*******************
    # emb_class = MultiEmbedding()
    # # first transform question_padded
    # # output, embedding_layer = emb_class.embedding(word_index, question_padded, embedding_matrix)
    glove_class = GloveVect()
    embedding_matrix, embedding_index = glove_class.glove_vect(q_word_index)


    split_class = SplitData()
    # question part and answer part will be seperately split
    # If postfix is number, this label is one-hot for calssify
    X_q_train, X_q_val, y_q_train, y_q_val = split_class.split_data(q_train_padded, y_q_table_0, test_size=0.2)
    # if postfix is df, this label is numerical
    X_a_train, X_a_val, y_a_train, y_a_val = split_class.split_data(a_train_padded, y_a_label_df, test_size=0.2)


    #******************Models*******************************

    model_class = BuildModels()
    compile_class = CompileFit()
    save_class = SaveModelHistory()
    #***************Random Embedding Normal Neural Network****************
    # nn_model = model_class.nn_model(q_word_index, part = 'q')
    #     history, model_2 = compile_fit(nn_model(word_index), X_q_train, X_q_val, y_q_train, y_q_val, loss_fun = 'mse', epoch_num=1)
    # history_a, model_a = compile_class.compile_fit(nn_model,
    #                                  X_q_train, X_q_val, y_q_train, y_q_val, loss_fun='categorical_crossentropy',
    #                                  epoch_num=3)

    # ***************Pretrain Glove Normal Neural Network****************(each model should have its own model part)
    pretrain_nn = model_class.nn_model(q_word_index, part='q', pretrain_matrix=embedding_matrix)
    history_a, model_a = compile_class.compile_fit(pretrain_nn,
                                                   X_q_train, X_q_val, y_q_train, y_q_val,
                                                   loss_fun='categorical_crossentropy',
                                                   epoch_num=3)
    history_classify_df = save_class.write_csv(history_a, model_a, str_input='Question_Glove_NN_20')




    #************************test part*****************************
    # 1.tokenize
    # 2.fit model
    # 3.get plot and analysis result
    #*************************END***************************



    return (df_train_raw, X_q_train_df, X_a_train_df, y_q_train_df, y_a_train_df, a_train_cleaned_df,
            q_train_padded, q_train_cleaned_df, a_train_padded, q_word_index, q_index_word,
            y_q_label_df, y_a_label_df, feature_col_q, feature_col_a, y_q_table_0,
            X_q_train, X_q_val, y_q_train, y_q_val,
            history_a, model_a)


if __name__=="__main__":
    """
    """
    (df_train_raw, X_q_train_df, X_a_train_df, y_q_train_df, y_a_train_df, a_train_cleaned_df,
     q_train_padded, q_train_cleaned_df, a_train_padded, q_word_index, q_index_word,
     y_q_label_df, y_a_label_df, feature_col_q, feature_col_a, y_q_table_0,
     X_q_train, X_q_val, y_q_train, y_q_val,
     history_a, model_a)= main()
    print("over")
