
import numpy as np
import time

from logistic_reg import LogisticRegression

# dataset import is pending
from sklearn import datasets

breast_cancer_ds = datasets.load_breast_cancer()
X, y = breast_cancer_ds.data, breast_cancer_ds.target

#print(X[:2], y[:20])
#time.sleep(5)

#X = np.random.rand(100, 2)
#y = np.random.rand(100)

model = LogisticRegression()

model.fit(X[:-5], y[:-5], epochs=200)

pred = model.predict(X[-5:])
pred_cls = [1 if i > 0.5 else 0 for i in pred]

print(pred_cls)
print(y[-5:])
print('acc: ', np.sum(pred_cls == y[-5:]) / len(y[-5:]))
