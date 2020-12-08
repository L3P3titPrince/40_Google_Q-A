import pandas as pd
# plot image
import matplotlib.pyplot as plt
# use spearmen correcticon to evaluate how good our model
from scipy.stats import spearmanr

from class_31_hyperparameters import HyperParameters

class AnalysisAndPlot(HyperParameters):
    """
    This class include numerical score, classify plot functions
    """
    def __init__(self, history):
        HyperParameters.__init__(self)
        self.history = history


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
        plt.savefig('04_images/15_N-gram_CNN_Classify_20Epochs_loss.png', dpi=150, format='png')

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
        plt.savefig('04_images/15_N-gram_CNN_Classify_20Epochs_acc.png', dpi=150, format='png')
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



    def SpearmanCorrCoeff(A, B):
        overall_score = 0
        for index in range(A.shape[1]):
            overall_score += spearmanr(A.iloc[:, index], B[:, index]).correlation
        return np.round(overall_score / A.shape[1], 3)

    def spearmanr_score(model, q_train_padded, y_q_label_df):
        """
        """
        #     print(model, y_q_label_df.head(5))
        # only use first column
        y_true = y_q_label_df
        # only get first predict
        y_pred = model.predict(q_train_padded)
        #     print(y_true.shape, y_pred.shape)
        assert y_true.shape == y_pred.shape
        print('\nValidation Score - ' + str(SpearmanCorrCoeff(y_q_label_df, model.predict(q_train_padded))))
    #     scores_2 = stats.spearmanr(y_a_val, test_predictions)







    # def get_scores(y_true, y_pred) -> Dict[str, float]:
    #     # y_true, y_pred: np.ndarray with shape (sample_size, num_targets)
    #     assert y_true.shape == y_pred.shape
    #     assert y_true.shape[1] == num_targets
    #     scores = {}
    #     for target_name, i in zip(target_names, range(y_true.shape[1])):
    #         scores[target_name] = scipy.stats.spearmanr(y_true[:, i], y_pred[:, i])[0]
    #     return scores
