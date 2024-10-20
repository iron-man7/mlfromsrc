
import numpy as np
from sklearn import datasets
from knn import KNN

iris = datasets.load_iris()

X, y = iris.data, iris.target

#print(X[:5], y[:5])

model = KNN(5)
model.fit(X[:-5], y[:-5])

pred = model.predict(X[-5:])

print("pred: ", pred)
print("expected: ", y[-5:])

acc = np.sum(pred == y[-5:]) / len(y[-5:])
print('acc: ', acc)

