import numpy as np
#cacualte spearmen correcltion
from scipy import stats
# make picture
import matplotlib.pyplot as plt
# draw picture
import seaborn as sns



class EdaData(object):
    """
    explorer this data structure
    """

    def __init__(self):
        """
        """
        # using distribution to decide this parameters
        self.MAX_SEQ_LENGTH = 400

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
        plt.show()
        plt.close()

    def answer_plot(self, df):
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
        plt.show()
        plt.close()

    def eda_length(df):
        """
        use statisc and plot to determine hyperparameters, such as MAX_SEQ_LEN, TOP_WORDS
        Arugs:
        ------

        """
        # get the 'AUTHOR' column sentence length
        sentence_len = [len(x) for x in df[df.columns[0]]]
        sentence_len_arr = np.array(sentence_len)
        # change the type to numpy array and get 95%/90%/85%th percentile of the data value
        print(f"95%th percentile of {df.columns[0]} sentence lenght is {np.percentile(sentence_len_arr, 95)}")
        print(f"90%th percentile of {df.columns[0]} sentence lenght is {np.percentile(sentence_len_arr, 90)}")
        print(f"85%th percentile of {df.columns[0]} sentence lenght is {np.percentile(sentence_len_arr, 85)}")
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