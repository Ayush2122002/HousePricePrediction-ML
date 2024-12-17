# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:50:20 2024

@author: avarsh1
"""

import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X,y = mnist["data"],mnist["target"]


import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X.iloc[0]


some_digit_image = some_digit.values.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

# converting to int
y = y.astype(np.uint8)

Xtrain,Xtest,ytrain,ytest = X[:60000],X[60000:],y[:60000],y[60000:]

# determine 5
ytrain_5= (ytrain==5)
ytest_5 = (ytest==5)

"""Stochastic Gradient Descent - training model"""
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(Xtrain,ytrain_5)
sgd_clf.predict([some_digit])

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, Xtrain, ytrain_5, cv=3,scoring="accuracy")

### Confusion Matrix
from sklearn.model_selection import cross_val_predict
ytrain_pred = cross_val_predict(sgd_clf, Xtrain, ytrain_5, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(ytrain_5,ytrain_pred)

from sklearn.metrics import precision_score, recall_score
precision_score(ytrain_5, ytrain_pred)
recall_score(ytrain_5, ytrain_pred)








