"""
@Coding: uft-8 *
@Time: 2019-05-15 15:13
@Author: Ryne Chen
@File: spam_email.py 
"""

import re
import random
from Ch04_NB import bayes


# Function to parse input text
def text_parse(sentence):
    # Set pattern
    reg_ex = re.compile(r'\W+')

    # Split text by pattern
    tokens = reg_ex.split(sentence)

    # Get token list with specified rules: lower() and len() > 2
    token_list = [token.lower() for token in tokens if len(token) > 2]

    return token_list


def spam_test():
    # Init container to save files
    file_list = []
    # Init container to save total words
    total_words = []
    # Init container ot save labels
    label_list = []

    # Scan directory to save data
    for i in range(1, 26):
        with open('./data/email/spam/6.txt', 'rb') as f_spam:
            # Read spam email
            content = f_spam.read().decode('utf-8', errors='ignore')
            # Parse file and get words list
            token_list = text_parse(content)
            # Save spam email file
            file_list.append(token_list)
            # Save spam email words
            total_words.extend(token_list)
            # Set label of spam email as 1
            label_list.append(1)

        with open('./data/email/ham/6.txt', 'rb') as f_ham:
            # Read normal email
            content = f_ham.read().decode('utf-8', errors='ignore')
            # Parse file and get words list
            token_list = text_parse(content)
            # Save normal email file
            file_list.append(token_list)
            # Save normal email words
            total_words.extend(token_list)
            # Set label of normal email as 0
            label_list.append(0)

    # print(len(file_list))
    # print(len(total_words))

    # Get unique vocabulary list of total words
    vocab_list = bayes.create_vocab_list(file_list)

    # Init training file index as 50
    training_file_index = list(range(50))
    # print(len(training_file_index))

    # Init test file index list
    test_file_index = []

    # Take 10 file index for test randomly
    for i in range(10):
        # Take email index randomly
        rand_index = int(random.uniform(0, len(training_file_index)))

        # Save index to text file index list
        test_file_index.append(rand_index)

        # Delete index in training file index list
        training_file_index.pop(rand_index)

    # print(len(training_file_index))
    # Init train vector matrix
    train_matrix = []
    # Init train labels list
    train_labels = []

    # Scan the remaining 40 files
    for train_index in training_file_index:
        # Get train file vector
        train_file_vec = bayes.words_2_vec_set(vocab_list, file_list[train_index])
        # Save to train vector matrix
        train_matrix.append(train_file_vec)
        # Get train file label
        train_labels.append(label_list[train_index])

    # Train data by naive bayes
    p_0_vec, p_1_vec, p_spam = bayes.trainNB(train_matrix, train_labels)

    # Init error counter
    counter = 0

    # Scan 10 test files
    for test_index in test_file_index:
        # Get test file vector
        test_file_vec = bayes.words_2_vec_set(vocab_list, file_list[test_index])

        # Get test file prediction label
        pred_label = bayes.classifyNB(test_file_vec, p_0_vec, p_1_vec, p_spam)

        # Compare with real test file label
        if pred_label != label_list[test_index]:
            counter += 1
            print('Classification error', file_list[test_index])

    rate = round(counter/len(test_file_index), 4)
    print('The error rate is: {}'.format(rate))


def main():
    # with open('./data/email/ham/6.txt', 'rb') as f:
    #     content = f.read().decode('utf-8', errors='ignore')
    #     token_list = text_parse(content)
    #     print(token_list)

    spam_test()


if __name__ == '__main__':
    main()
