import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Maths heavily inspired by https://towardsdatascience.com/multiclass-logistic-regression-from-scratch-9cc0007da372

onehot = OneHotEncoder()

def get_data(filename):
    df = pd.read_csv(filename, header=None)
    x = df.values.T[:-1].T
    y = df.values.T[-1]
    return x.astype(np.int64), y


class Softmax:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test

        classDict = defaultdict(int)

        classIdx = 0
        for idx, c in enumerate(y_train):
            if c not in classDict:
                classDict[c] = classIdx
                classIdx += 1
            y_train[idx] = np.int64(classDict[c])
        
        for idx, c in enumerate(y_test):
            y_test[idx] = np.int64(classDict[c])

        self.y_train = y_train
        self.y_test = y_test
        self.onehot_train = onehot.fit_transform(y_train.reshape(-1, 1))
        self.w = np.zeros((x_train.shape[1], self.onehot_train.shape[1]))


    def grad(self):
        n = x_train.shape[0]
        xw = self.x_train @ self.w
        smax = softmax(-xw, axis=1)
        return (x_train.T @ (self.onehot_train - smax)) / n  + 0.02 * self.w

        
    def fit(self):
        # just gradient descent 10k times
        for i in range(10000):
            self.w -= 0.1 * self.grad()
    
    def predict(self):
        fw = self.x_test @ self.w
        smax = softmax(-fw, axis=1)
        return np.argmax(smax, axis=1)





x, y = get_data("fertility.csv")

cnt = 0
tot = 0
for i in range(5):

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    predictor = Softmax(x_train, x_test, y_train, y_test)
    predictor.fit()
    cnt += np.sum(predictor.predict() == y_test)
    tot += len(y_test)
    # print(np.sum(predictor.predict() == y_test) / len(y_test) * 100)
print(cnt / tot * 100)