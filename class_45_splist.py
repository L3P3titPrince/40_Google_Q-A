import time
# split data with random seed (1024)
from sklearn.model_selection import train_test_split


class SplitData(object):
    """

    """
    def split_data(self, X_vector, y_vector, test_size=0.2):
        """
        this is only for padded data split
        """
        print("*" * 50, "Start train_test_split", "*" * 50)
        start_time = time.time()
        X_train, X_val, y_train, y_val = train_test_split(X_vector, y_vector, test_size=test_size,
                                                          random_state=1024)

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 37)

        cost_time = round((time.time() - start_time), 4)
        print("*" * 40, "End embedding() with {} seconds".format(cost_time), "*" * 40, end='\n\n')
        return X_train, X_val, y_train, y_val

