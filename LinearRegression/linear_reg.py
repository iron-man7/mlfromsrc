
import numpy as np
import time


class LinearRegression:
    def __init__(self, lr=0.0001, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def init_wandb(self, n_features):
        '''
        initialize weights and biases : Now sets it to 0.
        added random
        '''
        self.weights = np.random.rand(n_features)
        self.bias = 0

    def loss(self, y_pred, y):
        loss = np.mean(y_pred - y) ** 2
        return loss

    def acc(self, y_pred, y):
        # r2
        corrm = np.corrcoef(y_pred, y)
        acc = corrm[0, 1] ** 2
        return acc


    def gradient_dec(self, X, y, pred_y):
        n, _ = X.shape

        # derivative terms for weights and bias
        w_grad = (1 / n) * np.dot(X.T, (pred_y - y))
        b_grad = (1 / n) * sum(pred_y - y)

        # decide step direction
        self.weights -= self.lr * w_grad
        self.bias -= self.lr * b_grad


    def fit(self, X, y, show_progress=True):
        n_samples, n_features = X.shape
        self.init_wandb(n_features)

        #print(self.weights, self.bias)

        for step in range(self.epochs):
            # feed forword, one perceptron
            pred_y = np.dot(X, self.weights) + self.bias

            # loss/cost function calculation [ simple gradient ]
            self.gradient_dec(X, y, pred_y)

            if show_progress and step % 100 == 0:
                #loss = np.mean(pred_y - y)**2
                # r2
                #corrm = np.corrcoef(pred_y, y)
                #acc = corrm[0, 1] ** 2
                loss = self.loss(pred_y, y)
                acc  = self.acc(pred_y, y)
                print("iter: ", step, "\tloss: ", loss, "\tacc: ", acc)
                #print("iter : {}".format(step), end='\t')
                #print("New weights : {}".format(self.weights))
                #print("New bias: {}".format(self.bias))
                #print("Loss : {}".format(np.mean(pred_y - y)**2))
                #time.sleep(1)



    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
