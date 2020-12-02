import numpy as np
# use categorical to transform to one-hot coding
from tensorflow.keras.utils import to_categorical



class LabelProcess(object):
    """

    """

    def __init__(self, y_question_df, y_answer_df):
        self.y_question_df = y_question_df
        self.y_answer_df = y_answer_df

    def num_label(self):
        """
        In future, i will use arguritem to filter column. For now, i do it manually
        As i test use pure numerical algorithem to calcualte

        """
        # first try these labels
        feature_col_q = ['question_asker_intent_understanding',
                         'question_body_critical',
                         'question_expect_short_answer',
                         'question_interestingness_others'
                         ]
        y_q_label_df = self.y_question_df[feature_col_q]

        # extract the answer label features
        feature_col_a = ['answer_type_instructions']
        y_a_label_df = self.y_answer_df[feature_col_a]

        return y_q_label_df, y_a_label_df

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