"""
@Coding: uft-8 *
@Time: 2019-05-20 20:02
@Author: Ryne Chen
@File: logistic.py 
"""
import numpy as np


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


def main():
    data_matrix, label_list = load_data()
    weights = gradient_ascent(data_matrix, label_list)
    print(weights)


if __name__ == '__main__':
    main()
