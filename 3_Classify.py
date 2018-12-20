# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 22:05:35 2018

@author: Administrator
"""

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original',data_home='./datasets')
mnist

X,y = mnist['data'],mnist['target']
X.shape
y.shape

import matplotlib
import matplotlib.pyplot as plt
some_digit_image = X[36000].reshape((28,28))
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.title('the pic is '+y[36000].__str__())

X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]

import numpy as np
shuffle_index = np.random.permutation(60000)
X_train,y_train = X_train[shuffle_index],y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
