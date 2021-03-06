import numpy as np
#cacualte spearmen correcltion
from scipy import stats
# make picture
import matplotlib.pyplot as plt
# draw picture
import seaborn as sns
# use hyperparameters
from class_31_hyperparameters import HyperParameters


class EdaData(HyperParameters):
    """
    explorer this data structure
    The parameters need to be decided by eda:
        self.MAX_WORD = ???
    """

    def __init__(self):
        """
        """
        # using distribution to decide this parameters
        # self.MAX_SEQ_LENGTH = 400
        HyperParameters.__init__(self)
    # def q_distribution_plot():
    #     # from the plot we can see
    #     plt.scatter(y_answer_df.index, y_answer_df.iloc[:, 0])
    #     plt.show()

    def question_plot(self, df):
        """
        Due to different column number, we need to
        """
        #
        fig, axes = plt.subplots(7, 3, figsize=(18, 15))
        axes = axes.ravel()
        bins = np.linspace(0, 1, 20)

        for i, col in enumerate(df.columns):
            ax = axes[i]
            sns.histplot(df[col], label=col, kde=False, bins=bins, ax=ax)
            # ax.set_title(col)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 6079])
        plt.tight_layout()
        # plt.show() will release memory, so we need save file before show()
        plt.savefig('04_images/10_question_plot.png', dpi=150, format='png')
        plt.show()
        plt.close()

    def answer_plot(self, df):
        """
        Due to different column number, we need to
        """
        #
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.ravel()
        bins = np.linspace(0, 1, 20)

        for i, col in enumerate(df.columns):
            ax = axes[i]
            sns.histplot(df[col], label=col, kde=False, bins=bins, ax=ax)
            # ax.set_title(col)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 6079])
        plt.tight_layout()
        plt.savefig('04_images/11_answer_plot.png', dpi = 250, format='png')
        plt.show()
        plt.close()

    def eda_length(self, df, col_name):
        """
        use statisc and plot to determine hyperparameters, such as MAX_SEQ_LEN, TOP_WORDS
        Arugs:
        ------

        """
        # get the 'AUTHOR' column sentence length
        sentence_len = [len(x) for x in df[df.columns[6]]]
        sentence_len_arr = np.array(sentence_len)
        # change the type to numpy array and get 95%/90%/85%th percentile of the data value
        print(f"95%th percentile of {col_name} sentence lenght is {np.percentile(sentence_len_arr, 95)}")
        print(f"90%th percentile of {col_name} sentence lenght is {np.percentile(sentence_len_arr, 90)}")
        print(f"85%th percentile of {col_name} sentence lenght is {np.percentile(sentence_len_arr, 85)}")
        print(f"80%th percentile of {col_name} sentence lenght is {np.percentile(sentence_len_arr, 80)}")
        print(f"70%th percentile of {col_name} sentence lenght is {np.percentile(sentence_len_arr, 70)}")
        print(f"50%th percentile of {col_name} sentence lenght is {np.percentile(sentence_len_arr, 50)}")
        # get sorted
        sentence_len.sort()
        plt.plot(np.arange(len(sentence_len)), sentence_len)
        plt.title("Sentence Length Distribution")
        plt.xlabel("Sentence Amount")
        plt.ylabel("Sentence Length")
        plt.show()
        print(
            f"For MAX_SEQ_LEN, 1000 words in one sentce will reasonable to preseve 90% sentence and delete outlier value")

        return None
