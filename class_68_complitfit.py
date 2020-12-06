from time import time
# need specify lr in optiizer
from tensorflow.keras import optimizers

class CompileFit(object):

    def __init__(self):
        pass

    def compile_fit(self, model_input, X_train, X_val, y_train, y_val, loss_fun, epoch_num=3):
        """
        """
        start_time = time()
        print("*" * 40, "Start {} Processing".format(model_input._name), "*" * 40)

        model = model_input
        #     METRICS = [
        #           metrics.TruePositives(name='tp'),
        #           metrics.FalsePositives(name='fp'),
        #           metrics.TrueNegatives(name='tn'),
        #           metrics.FalseNegatives(name='fn'),
        #           metrics.CategoricalAccuracy(name='accuracy'),
        #           metrics.Precision(name='precision'),
        #           metrics.Recall(name='recall'),
        #           metrics.AUC(name='auc'),
        #           F1Score(num_classes = int(y_train.shape[1]), name='F1')
        #     ]

        learning_rate = 1e-2
        opt_adam = optimizers.Adam(lr=learning_rate, decay=1e-5)
        model.compile(loss=loss_fun,
                      optimizer=opt_adam,
                      metrics=['accuracy'])
        # batch_size is subjected to my GPU and GPU memory, after testing, 32 is reasonable value size.
        # If vector bigger, this value should dercrease
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epoch_num, batch_size=32, verbose=1)
        # dic = ['loss', 'accuracy', 'val_loss','val_accuracy']
        history_dict = [x for x in history.history]
        # model.predict(train_features[:10])

        #     print("*"*50)
        #     x_axis = list(range(epoch_num))
        #     # loss
        #     plt.plot(x_axis, history.history[history_dict[0]], color = 'r', lw = 2, label = history_dict[0])
        #     # val_loss
        #     plt.plot(x_axis, history.history[history_dict[10]], color = 'y', lw = 2, label = history_dict[10])
        #     # accuracy
        #     plt.plot(x_axis, history.history[history_dict[5]], color = 'b', lw = 2, label = history_dict[5])
        #     # validataion_accuracy
        #     plt.plot(x_axis, history.history[history_dict[15]], color = 'k', lw = 2, label = history_dict[15])
        #     plt.title(model_input._name)
        #     plt.legend()
        #     plt.xlabel('Epochs')
        #     # plt.ylabel(str(dic[i]))
        #     plt.show()

        cost_time = round((time() - start_time), 4)
        print("*" * 40, "End {} with {} seconds".format(model_input._name, cost_time), "*" * 40, end='\n\n')
        return history, model
