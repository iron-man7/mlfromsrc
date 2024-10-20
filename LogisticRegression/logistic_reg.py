
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.epochs = None
        self.weights = None
        self.bias = None

    def init_wandb(self, n_features):
        '''
        create weight vector of size inputs
        one bias term
        '''
        self.weights = np.random.rand(n_features)
        self.bias = 0


    def activation(self, x):
        '''
        using softmax function for calculation

        generates range between 0 to 1
        '''
        return 1 /( 1 + np.exp(-x))

    def acc(self, y_pred, y):
        '''
        bare minimum, verifiying how many are correct
        '''
        return np.sum(y_pred == y) / len(y)

    def gradient_dec(self, X, y_pred, y):
        # derivative terms, unchanged from linear reg
        n_samples, _ = X.shape
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # ** 2
        db = (1 / n_samples) * np.sum(y_pred - y) # ** 2

        # choosing the direction
        self.weights -= dw * self.lr
        self.bias -= db * self.lr

    def fit(self, X, y, epochs, show_progress=True):
        '''
        Trying linear regression output -> softmax
        '''
        n_samples, n_features = X.shape
        self.init_wandb(n_features)

        for step in range(epochs):
            # y = mx +b thingy
            y_pred = np.dot(X, self.weights) + self.bias

            # softmax
            y_pred = self.activation(y_pred)

            self.gradient_dec(X, y_pred, y)

            if show_progress and step % 10 == 0:
                loss = np.mean(y_pred - y) ** 2
                acc = self.acc(y_pred, y)
                print("iter: ", step, "\tloss: ", loss, "\tacc: ", acc)


    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return self.activation(y_pred)
