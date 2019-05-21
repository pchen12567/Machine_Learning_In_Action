"""
@Coding: uft-8 *
@Time: 2019-05-21 16:03
@Author: Ryne Chen
@File: colic_test.py 
"""

from Ch05_LR import logistic_SGA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Function to normalize the raw data
def norm_data():
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = pd.read_csv('./data/horseColicTraining.txt', sep='\t', header=None)
    train_norm = pd.DataFrame(scaler.fit_transform(train.iloc[:, :-1]))
    train_norm = pd.concat([train_norm, train[21]], axis=1)
    train_norm.to_csv('./data/horseColicTraining_norm.txt', sep='\t', header=None, index=False)

    test = pd.read_csv('./data/horseColicTest.txt', sep='\t', header=None)
    test_norm = pd.DataFrame(scaler.fit_transform(test.iloc[:, :-1]))
    test_norm = pd.concat([test_norm, test[21]], axis=1)
    test_norm.to_csv('./data/horseColicTest_norm.txt', sep='\t', header=None, index=False)


# Function to get taget label
def classify_vect(X, weights):
    prob = logistic_SGA.sigmoid(sum(X * weights))

    if prob > 0.5:
        return 1
    else:
        return 0


def colicTest():
    # Init containers of train data and label
    train_data = []
    train_labels = []

    # Read train data
    with open('./data/horseColicTraining_norm.txt', 'r') as train_file:
        for line in train_file.readlines():
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))

            # Save train factor data
            train_data.append(line_arr)
            # Save train label
            train_labels.append(float(curr_line[21]))

    # Calculate train data weights
    train_weights = logistic_SGA.opt_stoc_gra_ascent(np.array(train_data), train_labels, 500)

    # Init error counter
    error_count = 0

    # Init test number
    num_test = 0

    # Read test data
    with open('./data/horseColicTest_norm.txt', 'r') as test_file:
        for line in test_file.readlines():
            num_test += 1
            curr_line = line.strip().split()
            line_arr = []

            # Get test factor data
            for i in range(21):
                line_arr.append((float(curr_line[i])))

            # Compare pred label with real label
            if int(classify_vect(np.array(line_arr), train_weights)) != int(curr_line[21]):
                error_count += 1

    # Calculate error rate
    error_rate = (float(error_count) / num_test)
    print('The error rate of this test is: {}'.format(error_rate))

    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0

    for k in range(num_tests):
        error_sum += colicTest()

    print('After {} iterations the average error rate is: {}'.format(num_tests, error_sum / float(num_tests)))


def main():
    # norm_data()
    multi_test()


if __name__ == '__main__':
    main()
