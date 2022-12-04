import random
from csv import reader
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Testing functions adapted from https://machinelearningmastery.com/implement-random-forest-scratch-python/

class Node:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


def get_data(filename):
    df = pd.read_csv(filename, header=None)
    x = df.values.T[1:].T
    y = df.values.T[0]
    for idx, val in enumerate(y):
        np.append(x[idx], val)
        # x[idx].append(val)
    return x.tolist()

def file_to_data(filename):
    # turning the csv into a python 2D list
    data = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row: continue
            data.append(row)

    # converting members of the list from strings to floats
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            data[i][j] = float(data[i][j])

    # convert the classes to integers
    classes = {}
    x = 0
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes[data[i][-1]] = x
            x += 1
        data[i][-1] = classes[data[i][-1]]

    return data

def entropy(vector):
    # entropy is the measure of the purity of a split
    count = {}
    n = len(vector)
    for x in vector:
        if x not in count:
            count[x] = 0
        count[x] += 1 / n
    
    e = 0
    for c in count:
        e += -count[c] * math.log2(count[c])
    
    return e


def information_gain(parent, left, right):
    main = [row[-1] for row in parent]
    l = [row[-1] for row in left]
    r = [row[-1] for row in right]
    return entropy(main) - (len(l) * entropy(l) + len(r) * entropy(r)) / len(main)


def get_split(data, index, threshold):
    left = []
    right = []
    for row in data:
        if row[index] < threshold:
            left.append(row)
            continue
        right.append(row)
    return left, right

def getValue(cases):
    counts = {}
    for case in cases:
        if case[-1] not in counts:
            counts[case[-1]] = 0
        counts[case[-1]] += 1
    
    mxm = 0
    # most popular result that showed in the given set
    mostPopular = 0
    for result in counts:
        if counts[result] > mxm:
            mxm = counts[result]
            mostPopular = result
    return mostPopular
    
def best_split(data, n_features):
    classVals = [{case[-1] for case in data}]
    features = set()
    # best feature
    bF = -1
    # best gain
    bG = -1
    # best threshold
    bT = -1
    # best split
    bL = None
    bR = None
    while len(features) < n_features:
        index = random.randrange(len(data[0]) - 1)
        if index not in features:
            features.add(index)

    for index in features:
        for row in data:
            left, right = get_split(data, index, row[index])
            gain = information_gain(data, left, right)
            if (gain > bG):
                bG = gain
                bF = index
                bL = left 
                bR = right
                bT = row[index]
    
    return bF, bT, bL, bR

def split(node, left, right, max_depth, min_size, n_features, depth):
    if not left or not right:
        node.left = getValue(left + right)
        node.right = getValue(left + right)
        return
    if depth >= max_depth:
        node.left = getValue(left)
        node.right = getValue(right)
        return
    
    if len(left) <= min_size:
        node.left = getValue(left)
    else:
        # split further
        feature, threshold, l, r = best_split(left, n_features)
        node.left = Node(feature, threshold, None, None)
        split(node.left, l, r, max_depth, min_size, n_features, depth + 1)

    if len(right) <= min_size:
        node.right = getValue(right)
    else:
        # split further
        feature, threshold, l, r = best_split(right, n_features)
        node.right = Node(feature, threshold, None, None)
        split(node.right, l, r, max_depth, min_size, n_features, depth + 1)


def build(dataset, max_depth, min_size, n_features):
    feature, threshold, l, r = best_split(dataset, n_features)
    root = Node(feature, threshold, None, None)
    split(root, l, r, max_depth, min_size, n_features, 1)
    return root

def predict(node, case):
    if (case[node.feature] < node.threshold):
        if isinstance(node.left, int):
            return node.left
        return predict(node.left, case)
    else:
        if isinstance(node.right, int):
            return node.right
        return predict(node.right, case)


def subsample(data, ratio):
    sample = []
    n_sample = math.ceil(len(data) * ratio)
    while len(sample) < n_sample:
        case = random.randrange(len(data))
        sample.append(data[case])
    return sample

# checking the predictions of all the trees, and then returning the most voted prediction
def predict_vote(trees, case):
    count = {}
    for tree in trees:
        prediction = predict(tree, case)
        if prediction not in count:
            count[prediction] = 0
        count[prediction] += 1
    
    maxCount = 0
    mostPopular = 0
    for prediction in count:
        if count[prediction] > maxCount:
            maxCount = count[prediction]
            mostPopular = prediction
    return mostPopular


x = file_to_data("fertility.csv")
# x = get_data("abalone.csv")
trees = []
# actual = [row[-1] for row in x]
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build(sample, max_depth, min_size, n_features)
        trees.append(tree)
    
    predictions = []
    correct = 0
    for i, case in enumerate(test):
        prediction = predict_vote(trees, case)
        predictions.append(prediction)
    
    return predictions

    # get the prediction of all the trees
    # return the predictions


def cross_validation_split(data, n_folds):
	data_split = []
	data_copy = data[:]
	fold_size = int(len(data) / n_folds)
	for i in range(n_folds):
		fold = []
		while len(fold) < fold_size:
			index = random.randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)
	return data_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# printed = False
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
    folds = cross_validation_split(data, n_folds)
    scores = []
    for fold in folds:
        train_set = folds[:]
        
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = row[:]
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores







n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(math.sqrt(len(x[0])-1))

ls = []
for n_trees in (1, 5, 10):
    scores = evaluate_algorithm(x, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    ls.append(sum(scores) / len(scores))
    print(sum(scores) / len(scores))
	# print('Trees: %d' % n_trees)
	# print('Scores: %s' % scores)
	# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))





