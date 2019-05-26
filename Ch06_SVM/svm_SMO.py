"""
@Coding: uft-8 *
@Time: 2019-05-26 17:13
@Author: Ryne Chen
@File: svm_SMO.py 
"""

import random


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


def main():
    file_name = './data/testSet.txt'
    data_matrix, label_matrix = load_data(file_name)
    print(label_matrix)


if __name__ == '__main__':
    main()
