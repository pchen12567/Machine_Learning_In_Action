"""
@Coding: uft-8 *
@Time: 2019-05-13 21:05
@Author: Ryne Chen
@File: bayes.py 
"""

import numpy as np


# Function to create sample data
def load_data():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 1 is abusive, 0 not
    class_vec = [0, 1, 0, 1, 0, 1]

    return posting_list, class_vec


# Function to get all unique vocabularies in documents
def create_vocab_list(documents):
    # Init container
    vocab_set = set([])

    # Add unique vocabulary to container
    for doc in documents:
        vocab_set = vocab_set | set(doc)

    # Sorted result
    vocab_list = sorted(vocab_set)

    # Return the unique vocabularies list
    return vocab_list


# Function to get document vector set of target one according to unique vocabulary list
def words_2_vec_set(vocab_list, input_doc):
    # Init document vector with zeros
    result_vec = np.zeros(len(vocab_list))

    # Scan target document and set vector value to 1 if it exists in unique vocabulary list
    for word in input_doc:
        if word in vocab_list:
            result_vec[vocab_list.index(word)] = 1
        else:
            print('the word {} is not in my vocabulary.'.format(word))

    # Return target vector
    return result_vec


# Function to get document vector bag of target one according to unique vocabulary list
def words_2_vec_bag(vocab_list, input_doc):
    # Init document vector with zeros
    result_vec = np.zeros(len(vocab_list))

    # Scan target document and increase vector value by 1 if it exists in unique vocabulary list
    for word in input_doc:
        if word in vocab_list:
            result_vec[vocab_list.index(word)] += 1

    # Return target vector
    return result_vec


# Function to train data with label
def trainNB(train_matrix, category_label):
    # Get number of total document
    total_doc = len(train_matrix)

    # Get number of total words in vocabulary
    words_number = len(train_matrix[0])

    # Calculate probability of abusive document
    p_abusive = sum(category_label) / float(total_doc)

    # Init numerator vector
    p_0_numerator = np.ones(words_number)
    p_1_numerator = np.ones(words_number)

    # Init denominator
    p_0_denominator = 2
    p_1_denominator = 2

    # Scan all documents
    for i in range(total_doc):
        # If label of document is abusive
        if category_label[i] == 1:
            # Vector addition
            p_1_numerator += train_matrix[i]
            # Get total words of label = 1
            p_1_denominator += sum(train_matrix[i])

        # If label of document is not abusive
        else:
            # Vector addition
            p_0_numerator += train_matrix[i]
            # Get total words of label = 0
            p_0_denominator += sum(train_matrix[i])

    # Calculate word vector probability of label = 0
    p_0_vec = np.log(p_0_numerator / p_0_denominator)

    # Calculate word vector probability of label = 1
    p_1_vec = np.log(p_1_numerator / p_1_denominator)

    return p_0_vec, p_1_vec, p_abusive


# Build classifier
def classifyNB(doc_vec, p_0_vec, p_1_vec, p_class_1):
    # Get probability of target doc if label = 1
    p1 = sum(doc_vec * p_1_vec) + np.log(p_class_1)

    # Get probability of target doc if label = 0
    p0 = sum(doc_vec * p_0_vec) + np.log(1 - p_class_1)

    if p1 > p0:
        return 1
    else:
        return 0


def main():
    posting_list, class_vec = load_data()
    my_vocab_list = create_vocab_list(posting_list)
    print(my_vocab_list)

    # post_vec_1 = words_2_vec_set(my_vocab_list, posting_list[0])
    # post_vec_2 = words_2_vec_set(my_vocab_list, posting_list[3])
    # print(post_vec_1)
    # print(post_vec_2)

    train_matrix = []
    for post in posting_list:
        train_matrix.append(words_2_vec_set(my_vocab_list, post))

    p_0_vec, p_1_vec, p_abusive = trainNB(train_matrix, class_vec)
    # print('p_abusive: ', p_abusive)
    # print('p_0_vec: ', p_0_vec)
    # print('p_1_vec: ', p_1_vec)

    input_1 = ['love', 'my', 'dalmation']
    test_1 = words_2_vec_set(my_vocab_list, input_1)
    print('{} vector is: {}'.format(input_1, test_1))

    input_2 = ['stupid', 'garbage']
    test_2 = words_2_vec_set(my_vocab_list, input_2)
    print('{} vector is: {}'.format(input_2, test_2))

    print('{} classified as: {}'.format(input_1, classifyNB(test_1, p_0_vec, p_1_vec, p_abusive)))
    print('{} classified as: {}'.format(input_2, classifyNB(test_2, p_0_vec, p_1_vec, p_abusive)))


if __name__ == '__main__':
    main()
