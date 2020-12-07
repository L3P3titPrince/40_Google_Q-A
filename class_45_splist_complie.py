from time import time
# split data with random seed (1024)
from sklearn.model_selection import train_test_split
# need specify lr in optiizer
from tensorflow.keras import optimizers
# need use this to identify loss function
from tensorflow.keras import losses
# need hyperparameters
from class_31_hyperparameters import HyperParameters



class SplitAndCompile(HyperParameters):
    """
    This function contain
        1. split data into train and valistion (In this task we have individual test dataset)
        2. use num_classical() to choose numerical data or classfiy data
        3.
    """
    def __init__(self):
        """

        """
        HyperParameters.__init__(self)



    def split_data(self, X_vector, y_vector, test_size=0.2):
        """
        this is only for padded data split
        """
        print("*" * 50, "Start train_test_split", "*" * 50)
        start_time = time()
        X_train, X_val, y_train, y_val = train_test_split(X_vector, y_vector, test_size=test_size,
                                                          random_state=1024)

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 37)

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End embedding() with {} seconds".format(cost_time), "*" * 40, end='\n\n')
        return X_train, X_val, y_train, y_val



    def compile_fit(self, model_input, q_train_padded, a_train_padded, y_q_label_df, y_a_label_df,
                    y_q_classify_list, y_q_classify_dict, y_a_classify_list, y_a_classify_dict,
                    epoch_num=3):
        """
        This function is used to switch between numrical. The switch controled by hyperparameters self.TYPE
        When self.TYPE == 'num', input will be q_train_padded and y_q_label_df (others are same)
        Meanwhile, switch to ['MSE'] as loss and ['mse', 'mae'] as metrics

        When self.TYPE == 'classify', input will be q_train_padded and y_q_classify_list[0] etc.
        Meanwhile, swith to ['categorical_crossentropy'] as loss and ['accuracy'] as metrics

        """
        start_time = time()
        print("*" * 40, "Start {} Processing".format(model_input._name), "*" * 40)
        # loss_fun = 'categorical_crossentropy'
        # loss_fun = 'MSE' #MeanSquaredError
        # loss_fun = '

        loss_fun = None
        metrics_fun = None
        # becase large data input, we want to process automaticaly. So set this arugs to choose
        # question process or answer process automatically
        if self.PART == 'q':
            print("Start processing question part")
            # start to decide complie parameters
            if self.TYPE == 'num':
                print("Start numerical output")
                # call split
                X_train, X_val, y_train, y_val = self.split_data(q_train_padded, y_q_label_df, test_size=0.2)
                loss_fun = losses.MeanSquaredError()
                metrics_fun = ['mse', 'mae']
            elif self.TYPE == 'classify':
                print("Start classify output")
                X_train, X_val, y_train, y_val = self.split_data(q_train_padded, y_q_classify_list[0], test_size=0.2)
                loss_fun = losses.CategoricalCrossentropy()
                metrics_fun = ['accuracy']
            else:
                print("UNKNOW self.TYPE")

        elif self.PART == 'a':
            print("Start processing answer part")
            if self.TYPE == 'num':
                print("Start numerical output")
                # call split
                X_train, X_val, y_train, y_val = self.split_data(a_train_padded, y_a_label_df, test_size=0.2)
                loss_fun = losses.MeanSquaredError()
                metrics_fun = ['mse', 'mae']
            elif self.TYPE == 'classify':
                print("Start classify output")
                X_train, X_val, y_train, y_val = self.split_data(a_train_padded, y_a_classify_list[0], test_size=0.2)
                loss_fun = losses.CategoricalCrossentropy()
                metrics_fun = ['accuracy']
            else:
                print("UNKNOW self.TYPE")

        learning_rate = 1e-3
        opt_adam = optimizers.Adam(lr=learning_rate, decay=1e-5)
        model_input.compile(loss=loss_fun, optimizer=opt_adam, metrics=metrics_fun)
        # batch_size is subjected to my GPU and GPU memory, after testing, 32 is reasonable value size.
        # If vector bigger, this value should dercrease
        history = model_input.fit(X_train, y_train, validation_data=(X_val, y_val),
                                  epochs=epoch_num, batch_size=32, verbose=1)
        # dic = ['loss', 'accuracy', 'val_loss','val_accuracy']
        history_dict = [x for x in history.history]
        # model_input.predict(train_features[:10])

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End {} with {} seconds".format(model_input._name, cost_time), "*" * 40, end='\n\n')

        return history, model_input


