import numpy as np
from collections import defaultdict


def createDataSet():
    group = np.array([[1.0, 1.1],
                      [1.0, 1.0],
                      [0, 0],
                      [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    return group, labels


def knn_classify(X, data, labels, k):
    # Get row number of data
    dataSize = data.shape[0]

    # np.tile: Repeat X dataSize times to shape(dataSize,1)
    # Compute difference between X and each data
    diff = np.tile(X, (dataSize, 1)) - data

    # Square difference
    sqDiff = diff ** 2

    # Square sum
    sqDistances = sqDiff.sum(axis=1)

    # Get distance between X and each data
    distance = sqDistances ** 0.5
    print('Distance: ', distance)

    # Get the sorted index by distance
    sorted_index = distance.argsort()
    print('Sorted index by distance: ', sorted_index)

    # Init dict to compute labels
    label_count = defaultdict(int)

    # Count the top K labels
    for i in range(k):
        label = labels[sorted_index[i]]

        label_count[label] += 1

    # Sort the dict by label count
    label_count_rank = sorted(label_count.items(), key=lambda item: item[1], reverse=True)
    print('Sorted label count: ', label_count_rank)

    return label_count_rank[0][0]


# Test
group, labels = createDataSet()
X = [2, 2]
result = knn_classify(X, group, labels, 3)
print('Prediction: ', result)
