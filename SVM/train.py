
import numpy as np
from svm import SVM
from sklearn import datasets

X, y = datasets.make_blobs(
    #n_samples=100, n_features=3, centers=2, cluster_std=1.05, random_state=40
    n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40
)

y = np.where(y == 0, -1, 1)


model = SVM()
model.fit(X[:-5], y[:-5])

pred = model.predict(X[-5:])

print("pred", pred)
print("expected", y[-5:])

acc = np.sum(pred == y[-5:]) / len(y[-5:])
print("acc", acc)
