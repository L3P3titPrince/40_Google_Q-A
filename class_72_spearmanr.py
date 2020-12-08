from tensorflow.keras.callbacks import Callback


import pandas as pd
import numpy as np
# plot image
import matplotlib.pyplot as plt
# use spearman correcticon to evaluate how good our model
from scipy.stats import spearmanr


class PredictCallback(Callback):
    """

    """
    def __init__(self, X_data, y_label, model):
        """

        :param X_data: X_val
        :param y_true: y_val
        :param model:
        """
        Callback.__init__(self)
        self.X_data = X_data
        self.y_label = y_label
        self.model = model

    def SpearmanCorrCoeff(self, y_true, y_pred):
        """
        We use spearmanr to calculate result.
        For now, we only use validation to get result

        Argus:
        -----
        y_true:DataFrame
            In here, most of time y_true will be y_val (y_q_label_df / y_a_label_df)
            In the future, we need use test data inhere to evaluate result
        y_pred:numpy.array
            This augums come from model prediction, typicall is model.predict(q_train_padded))


        Return:
        ------
            spearman correlated result is an average result. We need caculate each column and
            then get the mean for this
        """
        # initial score
        overall_score = 0
        # iterate for each column
        for col in range(y_true.shape[1]):
            overall_score += spearmanr(y_true.iloc[:, col], y_pred[:, col]).correlation
        return np.round(overall_score / y_true.shape[1], 3)


    def on_epoch_end(self, epoch, logs = {}):
        """

        :param epoch:
        :param logs:
        :return:
        """
        y_pred = self.model.predict(self.X_data)
        print('\nVal_Spearman_Score - ' + str(self.SpearmanCorrCoeff(self.y_label, y_pred)))