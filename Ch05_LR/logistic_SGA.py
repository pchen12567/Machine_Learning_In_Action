"""
@Coding: uft-8 *
@Time: 2019-05-20 20:02
@Author: Ryne Chen
@File: logistic_SGA.py
"""
import numpy as np
import matplotlib.pyplot as plt
import random


# Function to load data
def load_data():
    data_matrix = []
    label_list = []

    with open('./data/testSet.txt', 'r') as f:
        for line in f.readlines():
            line_data = line.strip().split()
            data_matrix.append([1.0, float(line_data[0]), float(line_data[1])])
            label_list.append(int(line_data[2]))

    return data_matrix, label_list


# Function to define sigmoid
def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


# Function to implement  gradient ascent
def gradient_ascent(input_data, label_list):
    # Change input data to type numpy matrix
    x_matrix = np.mat(input_data)

    # Change label shape from 1*m to m*1
    label_matrix = np.mat(label_list).transpose()

    # Get shape of input data
    m, n = np.shape(x_matrix)

    # Init learning rate alpha
    alpha = 0.001

    # Set maximum iteration times
    max_cycles = 500

    # Init weights with shape n*1
    weights = np.ones((n, 1))

    for k in range(max_cycles):
        # m*n dot n*1 = m*1
        h = sigmoid(x_matrix * weights)

        # Vector subtraction, shape = m*1
        error = (label_matrix - h)

        # Vector addition, n*m dot m*1 = n*1
        weights = weights + alpha * x_matrix.transpose() * error

    # Return weights with shape 1*n
    return weights.transpose()


# Function to display best fit regression line
def plotBestFit(weights):
    dataMat, labelMat = load_data()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# Function to implement stochastic gradient ascent
def stoc_grad_ascent(data_matrix, labels):
    # Get shape of input data matrix
    m, n = np.shape(data_matrix)

    # Init learning rate alpha
    alpha = 0.01

    # Init weights, type = numpy array
    weights = np.ones(n)

    for i in range(m):
        # Scalar
        h = sigmoid(sum(data_matrix[i] * weights))

        # Scalar
        error = labels[i] - h

        # Scalar
        weights = weights + alpha * error * data_matrix[i]

    return weights


# Optimize stochastic gradient ascent
def opt_stoc_gra_ascent(data_matrix, labels, num_iter=150):
    # Get shape of input data matrix
    x_matrix = data_matrix.copy()
    m, n = np.shape(x_matrix)

    # Init weights, type = numpy array
    weights = np.ones(n)

    # Maximum iteration times = num_iter
    for j in range(num_iter):
        for i in range(m):
            # Set learning rate alpha
            # Decrease with iteration times
            alpha = 4 / (1.0 + j + i) + 0.01

            # Select sample index randomly
            rand_index = int(random.uniform(0, len(x_matrix)))

            # Scalar
            h = sigmoid(sum(x_matrix[rand_index] * weights))

            # Scalar
            error = labels[rand_index] - h

            # Scalar
            weights = weights + alpha * error * x_matrix[rand_index]

            list(x_matrix).pop(rand_index)

    return weights


def main():
    data_matrix, label_list = load_data()
    # weights = gradient_ascent(data_matrix, label_list)
    # weights = stoc_grad_ascent(np.array(data_matrix), label_list)
    weights = opt_stoc_gra_ascent(np.array(data_matrix), label_list, 500)
    plotBestFit(weights)
    print(weights)


if __name__ == '__main__':
    main()
