from sklearn.model_selection import train_test_split
import pandas as pd
from math import sqrt


def get_data(filename):
    df = pd.read_csv(filename, header=None)
    x = df.values.T[:-1].T
    y = df.values.T[-1]
    return x.tolist(), y.tolist()

def get_data_new(filename):
    df = pd.read_csv(filename, header=None)
    x = df.values.T[1:].T
    y = df.values.T[0]
    return x.tolist(), y.tolist()



class KNN:
    def __init__(self, x_train, x_test, y_train, y_test, k):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.k = k
    
    def euclidean_distance(self, vector1, vector2):
        Sum = 0
        for i in range(len(vector1)):
            Sum += (vector2[i] - vector1[i])**2
        
        return sqrt(Sum)

    def predict(self, features):
        distances = []
        for index, vector in enumerate(self.x_train):
            distances.append((self.euclidean_distance(vector, features), self.y_train[index]))
        
        distances.sort(key=lambda x: x[0])
        count = {}
        mxm = 0
        best = None
        for i in range(self.k):
            if distances[i][1] not in count:
                count[distances[i][1]] = 0
            count[distances[i][1]] += 1
            if count[distances[i][1]] > mxm:
                mxm = count[distances[i][1]]
                best = distances[i][1]
        return best

    def predict_and_give_accuracy(self):
        count = 0
        for i in range(len(self.x_test)):
            result = self.predict(self.x_test[i])
            if (result == self.y_test[i]):
                count += 1
        
        # print(f"{count / len(self.x_test) * 100.0}% accuracy")
        return count, len(self.x_test)






x, y = get_data("fertility.csv")

# x_train, x_test, y_train, y_test = train_test_split(x, y)


# predictor = KNN(x_train, x_test, y_train, y_test, 3)
# predictor.predict_and_give_accuracy()

cnt = 0
tot = 0
for i in range(5):

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    predictor = KNN(x_train, x_test, y_train, y_test, 3)
    a, b = predictor.predict_and_give_accuracy()
    cnt += a
    tot += b

print(cnt / tot * 100)


# res = []

# for k in range(1, 101):
#     count = 0
#     total = 0
#     for i in range(100):
#         x_train, x_test, y_train, y_test = train_test_split(x, y)
#         predictor = KNN(x_train, x_test, y_train, y_test, k)
#         a, b = predictor.predict_and_give_accuracy()
#         count += a
#         total += b
#     # print(f"{count / total * 100}")
#     res.append(count / total * 100)

# print(res)
