from sklearn.model_selection import train_test_split
import pandas as pd
from math import sqrt, exp, pi
from collections import defaultdict


def get_data(filename):
    df = pd.read_csv(filename, header=None)
    x = df.values.T[:-1].T
    y = df.values.T[-1]
    return x.tolist(), y.tolist()


class NaiveBayes:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.classes = defaultdict(list)
        self.probOfClass = defaultdict(int)

        for i in range(len(y_train)):
            self.probOfClass[y_train[i]] += 1
            self.classes[y_train[i]].append(x_train[i])

        for c in self.probOfClass:
            self.probOfClass[c] /= len(x_train)
        self.means = {}
        self.devs = {}
        for c in self.classes:
            self.means[c] = [0 for i in range(len(x_train[0]))]
            self.devs[c] = [0 for i in range(len(x_train[0]))]
            for feature in range(len(x_train[0])):
                featureValues = [row[feature] for row in self.classes[c]]
                self.means[c][feature] = self.mean(featureValues)
                self.devs[c][feature] = self.dev(featureValues)

    def mean(self, vector):
        return sum(vector) / len(vector)

    def dev(self, vector):
        # sample variance
        avg = self.mean(vector)
        variance = sum(((x - avg) ** 2 for x in vector)) / (len(vector) - 1)
        return sqrt(variance)

    def probability(self, x, mean, dev):
        if mean == 0 or dev == 0: return 1
        frac1 = 1 / (mean * sqrt(2 * pi))
        frac2 = -(((x - mean) / dev) ** 2) / 2
        return frac1 * exp(frac2)

    # takes feature vector and returns most probable class
    def predict(self, features):
        # p(class | x1, x2...) = p(x1|class) * p(x2|class) * ... * p(class)
        # for each feature, calculate p(xi|class) and multiply
        # p(xi|class) = gaussian distribution of ith feature that was part of class
        maxProb = 0
        bestClass = 0
        for c in self.classes:
            prob = self.probOfClass[c]
            for feature in range(len(self.x_train[0])):
                prob *= self.probability(features[feature], self.means[c][feature], self.devs[c][feature])
            if prob > maxProb:
                maxProb = prob
                bestClass = c
        
        return bestClass

    def predict_and_give_accuracy(self):
        count = 0
        for i in range(len(self.x_test)):
            result = self.predict(self.x_test[i])
            if (result == self.y_test[i]):
                count += 1
        
        return count, len(self.x_test)



x, y = get_data("fertility.csv")

cnt = 0
tot = 0
for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    predictor = NaiveBayes(x_train, x_test, y_train, y_test)
    a, b = predictor.predict_and_give_accuracy()
    cnt += a
    tot += b

print(cnt / tot * 100)
# print(predictor.predict_and_give_accuracy())