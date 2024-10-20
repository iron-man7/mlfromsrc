
import numpy as np
from sklearn import datasets
from linear_reg import LinearRegression


# generate more continuous data
X, y = datasets.make_regression(
    n_samples=500, n_features=2, noise=15, random_state=15
)


#print(X[:5], y[:5])

model = LinearRegression(epochs=1000)

# training
model.fit(X[:-5], y[:-5])

# prediction
pred = model.predict(X[-5:])
print("actual: {}".format(y[-5:]))
print("predictions: {}".format(pred))
# not relevant
#print("Accuracy: {}".format(np.sum(pred == y[-5:]) / len(y[-5:])))

# seen r2 in use
corr_m = np.corrcoef(y[-5:], pred)
corr = corr_m[0, 1]
print('Acc: {}'.format(corr ** 2))

