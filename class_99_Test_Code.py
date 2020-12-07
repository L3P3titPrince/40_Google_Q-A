# 1. In tf official document, MAX_WORD should not more than 999, but that way can not
# represent enough words to transfomr the sentence meaning

# 2. how to change self.MAX_Q_SEN_LEN from other class and restore or affect original class hyperparameter
# for example, we want to change OUTPUT_UNIT in class_35_label, in the meantime, class_62_cnn can receive this change
# Is there any way Child class can affect Parent class
# self.preprocess.max_features https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py

# 3.hyparameter choose we can check the paper https://arxiv.org/abs/1510.03820

"""
1.using question_user_page as benchmark, split quetsion_title or question_body into train_test, we believe different website have different type questions, so we can make evaluate and predict model

2.Using unsupervise learning to cluster question into differnt type, culster is depending on data preprocessing granularity. smaller grandularity, more spase cluster

3.After i embedding these sentence, you can use KNN SVM to do unsuperviese cluster

4.Using categore to cluster by CNN(n-gram / Glove / miniGPT）

5.Generage numerical value by former data and compart to anser_well_written

6.extract the root url like photo.stackchange.com to try to classfiy it with some argothrim, same question is to catgory column

7.If the result is not good enough, try to use url to grab more data to analysis

8.The data for this competition includes questions and answers from various StackExchange properties. Your task is to predict target values of 30 labels for each question-answer pair.

The list of 30 target labels are the same as the column names in the sample_submission.csv file. Target labels with the prefix question_ relate to the question_title and/or question_body features in the data. Target labels with the prefix answer_ relate to the answer feature.

9.for each dataframe maybe we need add category, and that will imporove performance

10.Stopword is meaningful for answer sequence, and so as punctuation. Try to only eliminate useless punctuatinon like '\`' but remain '?'and '!'
11.embedding is random initial word vector, but we can use Glove to import pretrain to impove performance

12.evalution part try to use BLEU score

13. After pre-trian, continue training 


"""





# for now, we only use 0 - 6 to represt each arrary, next step, this will be function into class LabelProcess
# y_q_array_0 = label_class.manual_calssify(y_q_label_df.iloc[:, 0])
# y_q_array_1 = label_class.manual_calssify(y_q_label_df.iloc[:, 1])
# y_q_array_2 = label_class.manual_calssify(y_q_label_df.iloc[:, 2])
# y_q_array_3 = label_class.manual_calssify(y_q_label_df.iloc[:, 3])
# y_q_array_4 = label_class.manual_calssify(y_q_label_df.iloc[:, 4])
# y_q_array_5 = label_class.manual_calssify(y_q_label_df.iloc[:, 5])
# # y_a_label_df = label_class.manual_calssify(y_answer_df.iloc[:, 0])
#     q_test_padded, q_test_cleaned_df = eda_class.tokenize_plot(q_test_cleaned_df, a_test_cleaned_df)
#     # get question label
#     y_label_test_df = eda_class.label_feature(y_q_test_df)




from time import time
# need specify lr in optiizer
from tensorflow.keras import optimizers

class CompileFit(object):

    def __init__(self):
        pass

    def compile_fit(self, model_input, X_train, X_val, y_train, y_val, type='num', epoch_num=3):
        """
        """
        start_time = time()
        print("*" * 40, "Start {} Processing".format(model_input._name), "*" * 40)

        model = model_input

        learning_rate = 1e-2
        opt_adam = optimizers.Adam(lr=learning_rate, decay=1e-5)
        model.compile(loss=loss_fun,
                      optimizer=opt_adam,
                      metrics=['accuracy'])
        # batch_size is subjected to my GPU and GPU memory, after testing, 32 is reasonable value size.
        # If vector bigger, this value should dercrease
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epoch_num, batch_size=32, verbose=1)
        # dic = ['loss', 'accuracy', 'val_loss','val_accuracy']
        history_dict = [x for x in history.history]
        # model.predict(train_features[:10])


        #     METRICS = [
        #           metrics.TruePositives(name='tp'),
        #           metrics.FalsePositives(name='fp'),
        #           metrics.TrueNegatives(name='tn'),
        #           metrics.FalseNegatives(name='fn'),
        #           metrics.CategoricalAccuracy(name='accuracy'),
        #           metrics.Precision(name='precision'),
        #           metrics.Recall(name='recall'),
        #           metrics.AUC(name='auc'),
        #           F1Score(num_classes = int(y_train.shape[1]), name='F1')
        #     ]

        #     print("*"*50)
        #     x_axis = list(range(epoch_num))
        #     # loss
        #     plt.plot(x_axis, history.history[history_dict[0]], color = 'r', lw = 2, label = history_dict[0])
        #     # val_loss
        #     plt.plot(x_axis, history.history[history_dict[10]], color = 'y', lw = 2, label = history_dict[10])
        #     # accuracy
        #     plt.plot(x_axis, history.history[history_dict[5]], color = 'b', lw = 2, label = history_dict[5])
        #     # validataion_accuracy
        #     plt.plot(x_axis, history.history[history_dict[15]], color = 'k', lw = 2, label = history_dict[15])
        #     plt.title(model_input._name)
        #     plt.legend()
        #     plt.xlabel('Epochs')
        #     # plt.ylabel(str(dic[i]))
        #     plt.show()

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End {} with {} seconds".format(model_input._name, cost_time), "*" * 40, end='\n\n')
        return history, model
