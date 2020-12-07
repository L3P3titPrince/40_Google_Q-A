import pandas as pd

class SaveModelHistory(object):
    """

    """
    def __init__(self):
        """

        """
        self.history_classify_df = pd.DataFrame(
            columns=['loss', 'accuracy', 'val_loss', 'val_accuracy', 'epoch', 'model_features'])

    def write_csv(self, history, model, str_input):
        """
        Use this function to restore history into csv. Next time, we can easily recall and plot former result
        """
        # save each indudial model in h5 format
        #     model.save("06_models/11_Normal_Glove_NN_20_model.h5")
        # transform current history dictionary into dataframe
        history_df = pd.DataFrame(history.history)
        # add epoch column
        history_df['epoch'] = history.epoch
        # add ['model_features'] to help identify each model parameter choose
        history_df['model_features'] = str(model.name) + "_" + str_input
        #     print(history_df)
        # append into old dataframe
        #     print(history_classify_df)
        #     history_classify_df.append(history_df)
        #     print(history_classify_df)
        frames = [history_df, self.history_classify_df]
        # concatanate old and new dataframe into one
        self.history_classify_df = pd.concat(frames)
        # write in tof file
        self.history_classify_df.to_csv(f"05_files/11_hisotry_classify_{str_input}_df.csv")

        return self.history_classify_df



