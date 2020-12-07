import numpy as np
# use categorical to transform to one-hot coding
from tensorflow.keras.utils import to_categorical



class LabelProcess(object):
    """
    Originally, we use tartget numerial as our label, which are
    """

    def __init__(self, y_question_df, y_answer_df):
        self.y_question_df = y_question_df
        self.y_answer_df = y_answer_df

    def num_label(self):
        """
        In future, i will use arguritem to filter column. For now, i do it manually
        As i test use pure numerical algorithem to calcualte

        For labels of question, we ignore the following parts
        since the results of these labels are almost coming to one category (1 or 0).
        For examples:
        question_conversational
        question_not_really_a_question
        question_tpye_compare

        They all have obviously meanings so that they will have inefficient evaluation about questions.
        Therefore, we are looking forward to finding some labels which have more balanced results.
        We choose several labels as follows.

        question_asker_intent_understanding:
        This label represents the level that people can understand the questions’intent.

        question_expect_short_answer:
        This label represents the level that people expect the short answers.

        question_has_commonly_accepted_answer:
        This label represents the level that the question has a commonly accepted answer.

        question_interestingness_others:
        This label represents the interestingness of people expect asker.

        question_interestingness_self:
        This label represents the interestingness of asker himself or herself.

        question_well_written:
        This label represents the level that the question can be well written.

        Not used 'question_body_critical'

        """
        # first try these labels, these labels are distribute average and easy for interpretation
        self.q_feature_col = ['question_asker_intent_understanding',
                         'question_expect_short_answer',
                         'question_has_commonly_accepted_answer',
                         'question_interestingness_others',
                         'question_interestingness_self',
                         'question_well_written'
                         ]
        # extract the dataframe of these columns
        self.y_q_label_df = self.y_question_df[self.q_feature_col]

        # extract the answer label features
        self.a_feature_col = ['answer_type_instructions',
                         'answer_satisfaction',
                         'answer_type_reason_explanation'
                         ]
        self.y_a_label_df = self.y_answer_df[self.a_feature_col]

        return self.y_q_label_df, self.y_a_label_df, self.q_feature_col, self.a_feature_col

    def auto_classify(self):
        """
        This result maybe get better result but not palusible.
        Because we don't konw how this data collect and why they got this pattern result
        So we can not make sure new data will still classify like 0.5 0.633

        input will be a column of label feature and output will be a 10 dimension to_categrical matrix
        """
        y_q_label_df = to_categorical(self.y_question_df.iloc[:, 0], num_classes=4)
        y_a_label_df = to_categorical(self.y_answer_df.iloc[:, 0], num_classes=5)

        return y_q_label_df, y_a_label_df

    def manual_calssify(self, label_col):
        """
        This function will manually segement numerical
        """
        # initial empty matrix
        categorical = np.zeros((len(label_col), 10), dtype='float32')
        for idx, label in enumerate(label_col):
            if 0 <= label < 0.1:
                categorical[idx, 0] = 1
            elif 0.1 <= label < 0.2:
                categorical[idx, 1] = 1
            elif 0.2 <= label < 0.3:
                categorical[idx, 2] = 1
            elif 0.3 <= label < 0.4:
                categorical[idx, 3] = 1
            elif 0.4 <= label < 0.5:
                categorical[idx, 4] = 1
            elif 0.5 <= label < 0.6:
                categorical[idx, 5] = 1
            elif 0.6 <= label < 0.7:
                categorical[idx, 6] = 1
            elif 0.7 <= label < 0.8:
                categorical[idx, 7] = 1
            elif 0.8 <= label < 0.9:
                categorical[idx, 8] = 1
            elif 0.9 <= label <= 1.0:
                categorical[idx, 9] = 1
            else:
                print('ERROR', label)

        # test part, if our calcuatlion is correct, all row should be included and have exactly number one
        # If correct, nothing happen, if condition return false, AssertionError is raised
        assert np.sum(categorical, axis=1).sum() == len(label_col)
        #         (unique, counts) = np.unique(test_13, return_counts=True)
        return categorical

    def classify_label(self):
        """
        Use this function to transform each classify column result into list

        :return:
        """
        # we need iterate each
        y_q_classify_list = []
        y_q_classify_dict = {}
        for idx, col in enumerate(self.q_feature_col):
            # for each column, we transfrom them into one-hot calssification columns
            y_q_array = self.manual_calssify(self.y_q_label_df.iloc[:, idx])
            # store them into a list
            y_q_classify_list.append(y_q_array)
            # get another containor dictionary
            y_q_classify_dict[col] = self.manual_calssify(self.y_q_label_df.iloc[:, idx])

        y_a_classify_list = []
        y_a_classify_dict = {}
        for idx, col in enumerate(self.a_feature_col):
            # for each column, we transfrom them into one-hot calssification columns
            y_a_array = self.manual_calssify(self.y_a_label_df.iloc[:, idx])
            # store them into a list
            y_a_classify_list.append(y_a_array)
            # get another containor dictionary
            y_a_classify_dict[col] = self.manual_calssify(self.y_a_label_df.iloc[:, idx])

        return y_q_classify_list, y_q_classify_dict, y_a_classify_list, y_a_classify_dict






        """
        # convert input dimension into np.array int format
        y = np.array(y, dtype='int')
        # get input_shape of y. Be caution, we accept high dimension narray as input
        input_shape = y.shape
        # if input_shape is exist, and last dimension is 1 and dimension of input_shape more than 2, then we only need the non-one part
        # for exmaple, if intput dimension last number is 1, then we think the use dimension is not include this one.
        # the final input_shape will only be dimesion from number to second to last (3,3,1) -> (3,3)
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        # flattern the y into contiguous array
        y = y.ravel()
        # if we don't specifity assign the classification number, the number of class of will max of y blus one 
        if not num_classes:
            num_classes = np.max(y) + 1
        # n is the number of examples
        n = y.shape[0]
        # create an empty matrix with (n,classes) dimension, n is number of exmaple,
        categorical = np.zeros((n, num_classes), dtype=dtype)
        #
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
        """