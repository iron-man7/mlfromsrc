
import numpy as np

class SVM:
    '''
    I think implementation will only limited to/work for linear and few problem statements.
    '''
    def __init__(self, lr=0.0001, lambda_prm=0.01):
        self.lr = lr
        self.lambda_param = lambda_prm
        self.epochs = None
        self.weights = None
        self.bias = None


    def init_wandb(self, n_features):
        self.weights = np.random.rand(n_features)
        self.bias = 0


    def fit(self, X, y, epochs=100):
        self.epochs = epochs
        n_samples, n_features = X.shape

        self.init_wandb(n_features)

        # change all param between -1 to 1
        y_ = np.where(y <= 0, -1, 1)

        for step in range(self.epochs):
            for idx, i in enumerate(X):
                # make a decision from the condition
                condition = y_[idx] * (np.dot(i, self.weights) - self.bias) >= 1

                # decide step
                if condition:
                    # only when dot product is larger than 1
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

            #not implementing progress/training prints.


    def predict(self, X):
        #pred = np.dot(self.weights, X) - self.bias
        pred = np.dot(X, self.weights) - self.bias
        return np.sign(pred) # its fancy way of denoting -1 for x < 0, 0 for x == 0, 1 for x > 0


