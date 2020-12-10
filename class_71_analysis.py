import pandas as pd
import numpy as np
# plot image
import matplotlib.pyplot as plt
# use spearman correcticon to evaluate how good our model
from scipy.stats import spearmanr

from class_31_hyperparameters import HyperParameters

class AnalysisAndPlot(HyperParameters):
    """
    This class include numerical score, classify plot functions
    """
    def __init__(self, history):
        HyperParameters.__init__(self)
        # self.shape = None
        self.history = history


    def plot_history(self):
        """

        :return:
        """
        if self.TYPE == 'num':
            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch

            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Abs Error [MPG]')
            plt.plot(hist['epoch'], hist['mae'],
                     label='Train Error')
            plt.plot(hist['epoch'], hist['val_mae'],
                     label='Val Error')
            plt.ylim([0, 0.5])
            plt.legend()
            plt.savefig(f'04_images/{self.NAME_STR}_mae.png', dpi=150, format='png')
            # plt.savefig(f'/googledrive/MyDrive/04_images/{self.NAME_STR}_mae.png', dpi=150, format='png')

            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Square Error [$MPG^2$]')
            plt.plot(hist['epoch'], hist['mse'],
                     label='Train Error')
            plt.plot(hist['epoch'], hist['val_mse'],
                     label='Val Error')
            plt.ylim([0, 0.2])
            plt.legend()
            plt.savefig(f'04_images/{self.NAME_STR}_mse.png', dpi=150, format='png')
            # plt.savefig(f'/googledrive/MyDrive/04_images/{self.NAME_STR}_mse.png', dpi=150, format='png')
            plt.show()

        elif self.TYPE == 'classify':
            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch

            # ****loss plost *******************
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.plot(hist['epoch'], hist['loss'],
                     label='Train loss')
            plt.plot(hist['epoch'], hist['val_loss'],
                     label='Val loss')
            plt.ylim([0, 3])
            plt.legend()
            plt.savefig(f'04_images/{self.NAME_STR}_loss.png', dpi=150, format='png')
            # plt.savefig(f'/googledrive/MyDrive/04_images/{self.NAME_STR}_loss.png', dpi=150, format='png')

            # ****************accuracy plot******************
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.plot(hist['epoch'], hist['accuracy'],
                     label='Train Accuracy')
            plt.plot(hist['epoch'], hist['val_accuracy'],
                     label='Val Accuracy')
            plt.ylim([0, 1])
            plt.legend()
            plt.savefig(f'04_images/{self.NAME_STR}_acc.png', dpi=150, format='png')
            # plt.savefig(f'/googledrive/MyDrive/04_images/{self.NAME_STR}_acc.png', dpi=150, format='png')
            plt.show()




    def plot_history_classify(self):
        """
        This function is used for plot classification self.history result
        :param self.history:
        :return:
        """
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch

        # ****loss plost *******************
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.plot(hist['epoch'], hist['loss'],
                 label='Train loss')
        plt.plot(hist['epoch'], hist['val_loss'],
                 label='Val loss')
        plt.ylim([0, 3])
        plt.legend()
        plt.savefig(f'04_images/{self.NAME_STR}_loss.png', dpi=150, format='png')
        # plt.savefig(f'/googledrive/MyDrive/04_images/{self.NAME_STR}_loss.png', dpi=150, format='png')

        # ****************accuracy plot******************
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(hist['epoch'], hist['accuracy'],
                 label='Train Accuracy')
        plt.plot(hist['epoch'], hist['val_accuracy'],
                 label='Val Accuracy')
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(f'04_images/{self.NAME_STR}_acc.png', dpi=150, format='png')
        # plt.savefig(f'/googledrive/MyDrive/04_images/{self.NAME_STR}_acc.png', dpi=150, format='png')
        plt.show()
        return hist

    def plot_history_num(self):
        """

        :param self.history:
        :return:
        """
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
                 label='Val Error')
        plt.ylim([0, 0.5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                 label='Val Error')
        plt.ylim([0, 0.2])
        plt.legend()
        plt.show()



    def SpearmanCorrCoeff(self, y_true, y_pred):
        """
        We use spearmanr to calculate result.
        For now, we only use validation to get result

        Argus:
        -----
        y_true:DataFrame
            In here, most of time y_true will be y_q_label_df / y_a_label_df.
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


    def spearmanr_score(self, model, q_train_padded, y_q_label_df):
        """
        """
        #     print(model, y_q_label_df.head(5))
        # only use first column
        y_true = y_q_label_df
        # only get first predict
        y_pred = model.predict(q_train_padded)
        #     print(y_true.shape, y_pred.shape)
        assert y_true.shape == y_pred.shape
        print('\nValidation Score - ' + str(self.SpearmanCorrCoeff(y_q_label_df, model.predict(q_train_padded))))
    #     scores_2 = stats.spearmanr(y_a_val, test_predictions)







    # def get_scores(y_true, y_pred) -> Dict[str, float]:
    #     # y_true, y_pred: np.ndarray with shape (sample_size, num_targets)
    #     assert y_true.shape == y_pred.shape
    #     assert y_true.shape[1] == num_targets
    #     scores = {}
    #     for target_name, i in zip(target_names, range(y_true.shape[1])):
    #         scores[target_name] = scipy.stats.spearmanr(y_true[:, i], y_pred[:, i])[0]
    #     return scores
