import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(filename):
    df = pd.read_csv(filename, header=None)
    x = df.values.T[:-1].T
    y = df.values.T[-1]

    trans = {}
    c = 0
    for i, val in enumerate(y):
        if val not in trans:
            trans[val] = c
            c += 1
        y[i] = trans[val]

    return x.astype(np.float64), y.astype(np.float64)

class LogisticRegression():
    def __init__(self, x_train, x_test, y_train, y_test, lr=0.5, epoch=10000):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.bias = 0
        self.weight = np.zeros(x_train.shape[1])
        self.lr = lr
        self.epoch = epoch

    def fit(self):
        for i in range(self.epoch):
            self.grad_desc()

    def grad_desc(self):
        Y = 1 / (1 + np.exp(-(self.x_train.dot(self.weight) + self.bias)))    
        dw = (1/self.x_train.shape[0])*np.dot(self.x_train.T, (Y - self.y_train))
        db = (1/self.x_train.shape[0])*np.sum(Y - self.y_train)
        self.weight -= self.lr * dw
        self.bias -= self.lr * db


    def predict(self, X):
        p = 1 / (1 + np.exp(-(X.dot(self.weight) + self.bias))) 
        p = np.where(p >= 0.5, 1, 0)
        return p



x, y = get_data("fertility.csv")
cnt = 0
tot = 0
for i in range(5):


    x_train, x_test, y_train, y_test = train_test_split(x, y)

    pred = LogisticRegression(x_train, x_test, y_train, y_test)
    pred.fit()
    for idx, row in enumerate(x_test):
        if pred.predict(row) == y_test[idx]: cnt += 1
    tot += len(y_test)


print(cnt / tot * 100)