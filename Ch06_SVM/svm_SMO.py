"""
@Coding: uft-8 *
@Time: 2019-05-26 17:13
@Author: Ryne Chen
@File: svm_SMO.py 
"""

import random
import numpy as np
from Ch06_SVM import support


# Function to parse source data
def load_data(file_name):
    # Init container
    data_matrix = []
    label_matrix = []

    with open(file_name, 'r') as f:
        for line in f.readlines():
            line_array = line.strip().split('\t')

            # Get each sample data
            data_matrix.append([float(line_array[0]), float(line_array[1])])

            # Get each sample label
            label_matrix.append(float(line_array[2]))

    return data_matrix, label_matrix


# Auxiliary function to get a random index of alpha
def select_Jrand(i, m):
    """
    :param i: index of first alpha
    :param m: number of total alpha
    :return: random index of alpha except the first one
    """
    # Init j
    j = i

    while j == i:
        # Get a random index of alpha except the first one
        j = int(random.uniform(0, m))

    return j


# Auxiliary function to adjust alpha
def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# Function to implement simple SMO algorithm
def smo_simple(data_in, labels, C, tolerance, max_iter):
    """
    :param data_in: Source data set
    :param labels: Category labels
    :param C: Constant
    :param tolerance: Max error tolerance
    :param max_iter: Max iteration times
    :return:
    """
    # Change source data to type matrix, shape = m * n
    data_matrix = np.mat(data_in)
    # Change labels to type matrix and transpose, shape = m * 1
    label_matrix = np.mat(labels).transpose()

    # Init b
    b = 0

    # Get shape of data
    m, n = np.shape(data_matrix)

    # Init alphas matrix with zeros, shape = m * 1
    alphas = np.mat(np.zeros((m, 1)))

    # Init iteration time with 0
    iteration = 0

    while iteration < max_iter:
        # Init alpha pairs change time with 0
        alpha_pairs_changed = 0

        for i in range(m):
            # Calculate first alpha i prediction value
            fXi = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b

            # Get error value of alpha i
            Ei = fXi - float(label_matrix[i])

            # Condition if alpha can be optimized
            if (label_matrix[i] * Ei < -tolerance and alphas[i] < C) or \
                    (label_matrix[i] * Ei > tolerance and alphas[i] > 0):

                # Get the second alpha j
                j = select_Jrand(i, m)

                # Calculate second alpha j prediction value
                fXj = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b

                # Get error value of alpha j
                Ej = fXj - float(label_matrix[j])

                # Get init first alpha i
                alpha_I_old = alphas[i].copy()

                # Get init second alpha j
                alpha_J_old = alphas[j].copy()

                # Get L and H to adjust alpha, between 0 and C
                if label_matrix[i] != label_matrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue

                # Calculate the value for modifying alpha j
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                      data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T

                if eta >= 0:
                    print('eta >= 0')
                    continue

                alphas[j] -= label_matrix[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)

                if abs(alphas[j] - alpha_J_old) < 0.00001:
                    print('j not moving enough')
                    continue

                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_J_old - alphas[j])

                b1 = b - Ei - label_matrix[i] * (alphas[i] - alpha_I_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_J_old) * data_matrix[i, :] * data_matrix[j, :].T

                b2 = b - Ej - label_matrix[i] * (alphas[i] - alpha_I_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_matrix[j] * (alphas[j] - alpha_J_old) * data_matrix[j, :] * data_matrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2

                alpha_pairs_changed += 1

                print('iteration: {}, i: {}, pairs changed {}.'.format(iteration, i, alpha_pairs_changed))

        if alpha_pairs_changed == 0:
            iteration += 1
        else:
            iteration = 0
        print('iteration number: {}.'.format(iteration))

    return b, alphas


def smo_P(data_in, labels, C, tolerance, max_iter, k_tup=('lin', 0)):
    oS = support.optStruct(np.mat(data_in), np.mat(labels).transpose(), C, tolerance)

    iteration = 0

    entire_set = True
    alpha_pairs_changed = 0

    while (iteration < max_iter) and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(oS.m):
                alpha_pairs_changed += support.inner_L(i, oS)
                print('fullSet, iteration: {}, i: {}, pairs changed: {}.'.format(iteration, i, alpha_pairs_changed))
            iteration += 1

        else:
            non_bound_is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]

            for i in non_bound_is:
                alpha_pairs_changed += support.inner_L(i, oS)
                print('non-bound, iteration: {}, i: {}, pairs changed: {}.'.format(iteration, i, alpha_pairs_changed))
            iteration += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print('iteration number: {}'.format(iteration))

    return oS.b, oS.alphas


# Function to get W
def cal_W(alphas, data_array, labels):
    X = np.mat(data_array)
    label_matrix = np.mat(labels).transpose()
    m, n = np.shape(X)
    W = np.zeros((n, 1))
    for i in range(m):
        W += np.multiply(alphas[i] * label_matrix[i], X[i, :].T)
    return W


def main():
    file_name = './data/testSet.txt'
    data_matrix, label_matrix = load_data(file_name)
    # print(label_matrix)
    # b, alphas = smo_simple(data_matrix, label_matrix, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas > 0])
    # print(np.shape(alphas[alphas > 0]))
    # for i in range(100):
    #     if alphas[i] > 0:
    #         print(data_matrix[i], label_matrix[i])

    b, alphas = smo_P(data_matrix, label_matrix, 0.6, 0.001, 40)
    w = cal_W(alphas, data_matrix, label_matrix)
    # print(w)
    dataMat = np.mat(data_matrix)
    print('Prediction: ', dataMat[0] * np.mat(w) + b)
    print('Real: ', label_matrix[0])


if __name__ == '__main__':
    main()
