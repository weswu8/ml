#!/usr/bin/python
# -*- coding: utf-8 -*-
########################################################################
# Name:
# 		knn
# Description:
# 		knn
# Author:
# 		wesley wu
# Python:
#       3.5
# Version:
#		1.0
########################################################################
import csv as csv
import random as random
import math as math
import operator as operator


# load the data and split it into two data set
def prepare_dataset(filename ):
    # read the data from csv into the list
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        orig_data = list(reader)

   # Divide the data into two parts, one for training and the other for verification
    random.shuffle(orig_data)
    train_data = orig_data[:int(0.7 * 30)]
    val_data = orig_data[int(0.7 * 30):]
    return train_data, val_data


# calculate the distance
def calculate_euclidean_distance(s1,s2):
    ed = 0.0
    # exclude the label
    for i in range(len(s1) - 1):
        ed += pow((float(s1[i]) - float(s2[i])), 2)  # euclidean distance
    ed = math.sqrt(ed)
    return ed


# find the nearest neighbors, the number of neighbors is defined by k value
def find_neighbors(dataset, sample, k):
    distances = []
    for x in range(len(dataset)):
        dist = calculate_euclidean_distance(dataset[x],sample )
        distances.append((dataset[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# do the prediction, the sample point will evaluate the vote of the k nearest neighbors
def knn_prediction(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    # the class with largest voted numbers
    return sortedVotes[0][0]


def getAccuracy(valset, predictions):
    correct = 0
    for x in range(len(valset)):
        if valset[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(valset))) * 100.0


# main loop begin here
train_data, val_data = prepare_dataset("knn.csv")
# generate predictions
predictions=[]
# the number of nearest neighbors
k = 3
for x in range(len(val_data)):
    neighbors = find_neighbors(train_data, val_data[x], k)
    #print("x:{} and it's neighbors:{}".format(val_data[x],neighbors))
    result = knn_prediction(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(val_data[x][-1]))
accuracy = getAccuracy(val_data, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
print(calculate_euclidean_distance(train_data[0],train_data[1]))