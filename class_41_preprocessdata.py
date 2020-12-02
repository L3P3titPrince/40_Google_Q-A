import pandas as pd

class PreprocessData(object):
    """
    Because this data have two components, so we first preprocess it and then return raw and corpus to do EDA
    We also need to use preprocee to estimate hyperparameters
    """

    def __init__(self):
        None

    def import_data(self, path):
        """
        Arugs:
        ------
        path:string
            directory of file you want to read

        Return:
        ------


        """
        # read the raw unpreprocess data into df_raw
        df_raw = pd.read_csv(path)
        # first we need extract the X(data) part and y(label) part.
        # In this dataset, columns from "qa_id" to "host" will be X(data)
        # columns from "question_asker_intent_understanding" to "answer_will_written" are human label result which is numerical betwenn [0,1]
        #
        X_df = df_raw.iloc[:, 0:10]
        y_df = df_raw.iloc[:, 11:]

        # we classify question_title and question_body in X_question_df, question_ related columns into y_question_df,
        # this is X y for one task
        # we classify answer in X_answer, answer_ related column into y_answer_df. This is X and y for another task
        # maybe we need sometime consider questoin and answer together
        # i believe user info have no contribution with this task
        # construct DataFrame
        self.X_question_df = df_raw.loc[:, ['qa_id', 'question_title', 'question_body', 'category', 'host']]
        self.X_answer_df = df_raw.loc[:, ['qa_id', 'answer', 'category', 'host']]

        # initial label list
        y_question_list = []
        y_answer_list = []
        for idx, i in enumerate(y_df.columns):
            if "question_" in i:
                y_question_list.append(i)
            elif "answer_" in i:
                y_answer_list.append(i)
            else:
                continue

        # extract label of question
        self.y_question_df = df_raw[y_question_list]
        self.y_answer_df = df_raw[y_answer_list]
        # So, for now, we have two pair of dataset, firt is (X_question_df + y_question_df). Second is (X_answer_df + y_answer_df)

        # for question part, i think we need a new column for merge title and body, but we still reserve seperate column
        self.X_question_df['question'] = self.X_question_df['question_title'] + self.X_question_df['question_body']

        return df_raw, self.X_question_df, self.X_answer_df, self.y_question_df, self.y_answer_df