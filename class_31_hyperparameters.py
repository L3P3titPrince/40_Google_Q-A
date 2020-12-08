class HyperParameters(object):
    """
    This class will be used to transmit hypereparametes between class.
    Most of class will inherit this class and hyerparameters.
    If this class want to change personal hyperparameters, it can modify in __init__()
    Most of value of hyperparamters come from EDA part. Others defined by human

    You can also use this paramter to control we use numerical or classify to calcualte
    """
    def __init__(self):
        """
        It can be designed to accept arguments from outside in main() function
        Also, you can modify hyperparameters just in here
        """
        # max length of sentence, question/answer sequence have different max length
        self.MAX_Q_SEQ_LEN = 400
        self.MAX_A_SEQ_LEN = 1000

        # how many words you want to reserve.
        # Because some words appear less than 5 times in whole corpus
        # so sometimes we think they can not transfrom useful infomation
        # if we remain all of word in question and answer, the max word number in tokenize is 72070
        # acoording to tf official document, this
        self.MAX_WORD = 10000

        # the vector dimesnsion for Embedding(). This can influence running speed
        # higher dimensin will take more time than lower dimesnions
        self.EMBEDDING_DIM = 50

        # If we choose six column in class_35_label, we will assign unit = 6
        self.Q_OUTPUT_UNIT = 6
        self.A_OUTPUT_UNIT = 3

        # reading GLOVE PATH
        self.PATH = r"D:\Downloads\glove.6B\glove.6B.50d.txt"

        # if type == 'num',
        # First class SplistData will use numerical ylabel = y_q_label_df / y_a_label_df
        # the target y_q_label and y_a_label_df will be 6 column and 3 columns numreical result
        # then each modle will change its output layer to 6 or 3 Dense unit and transform activation
        # Last step, complie part will switch to MSE as loss function and ['mae', 'mse'] as metrics
        self.TYPE = 'num'
        # First class SplitData will use labeled data y_q_array_0 / y_q_array_label
        # the classify label will be (# of example, 10) for each column features
        # PRECAUTION, each column feature have its own label,
        # i.e., in question part, we need run six times, because we have six different columns
        # then each model will change its output layer to 10 unit softmax for classification
        # Last stop, compile fun will switch to categorical-crossentorya and ['accuracy'] as metrix
        # self.TYPE = 'classify'


        # ADD data process part hypereparameters, different part use differnt process arguments
        self.PART = 'q'
        # self.PART = 'a'


        # this parameter is used to name model_save file, plot picture and history save file name
        self.NAME_STR = '15_q_num_glove_n-gram_lstm_3'
