class HyperParameters(object):
    """
    This class will be used to transmit hypereparametes between class.
    Most of class will inherit this class and hyerparameters.
    If this class want to change personal hyperparameters, it can modify in __init__()
    Most of value of hyperparamters come from EDA part. Others defined by human
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
        self.MAX_WORD = 70000

        # the vector dimesnsion for Embedding(). This can influence running speed
        # higher dimensin will take more time than lower dimesnions
        self.EMBEDDING_DIM = 50

