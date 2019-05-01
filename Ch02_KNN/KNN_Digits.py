import numpy as np
import os
from Ch02_KNN import KNN


# Switch image file to vector
def img2vector(file):
    # Init result vector
    result_vector = np.zeros((1, 1024))

    with open(file) as f:

        # Read the first 32 lines of file
        for i in range(32):
            line = f.readline()

            # Get the first 32 characters of each line
            for j in range(32):
                result_vector[0, 32 * i + j] = int(line[j])

    return result_vector


def classify_digits(train_dir, test_dir):
    # Init digit labels list
    digit_labels = []

    # Get train files
    train_files = os.listdir(train_dir)

    # Get number of train files
    m_train = len(train_files)

    # Init train matrix
    train_matrix = np.zeros((m_train, 1024))

    # Read train files one by one
    for i in range(m_train):
        # Get train file name
        train_file_name = train_files[i]

        # Get train file label from file name which position is the first
        train_digit_label = int(train_file_name.split('.')[0].split('_')[0])

        # Save the labels to label list
        digit_labels.append(train_digit_label)

        # Switch the image file to vector
        train_matrix[i, :] = img2vector('{}/{}'.format(train_dir, train_file_name))

    # Get test files
    test_files = os.listdir(test_dir)

    # Init error counter
    count = 0

    # Get number of test files
    m_test = len(test_files)

    # Read test files one by one
    for i in range(m_test):

        # Get test file name
        test_file_name = test_files[i]

        # Get test file real label from file name which position is the first
        test_digit_label = int(test_file_name.split('.')[0].split('_')[0])

        # Switch the image file to vector
        test_vector = img2vector('{}/{}'.format(test_dir, test_file_name))

        # Implement KNN classifier to predict test file label
        classify_result = KNN.knn_classify(X=test_vector, data=train_matrix, labels=digit_labels, k=3)

        # Compare the prediction and real label of test files
        # If not match, error counter +1
        if classify_result != test_digit_label:
            count += 1

        print('The classifier came back with: {}, the real answer is: {}'.format(classify_result, test_digit_label))

    print('The total number of errors is: {}'.format(count))
    print('The total error rate is: {}%'.format((count / m_test) * 100))


def main():
    train_dir = './data/trainingDigits'
    test_dir = './data/testDigits'

    classify_digits(train_dir, test_dir)


if __name__ == '__main__':
    main()
