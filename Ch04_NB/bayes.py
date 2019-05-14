"""
@Coding: uft-8 *
@Time: 2019-05-13 21:05
@Author: Ryne Chen
@File: bayes.py 
"""


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

    # Return the unique vocabularies list
    return list(vocab_set)


# Function to get document vector of target one according to unique vocabulary list
def words_2_vec_set(vocab_list, input_doc):
    # Init document vector with zeros
    result_vec = [0] * len(vocab_list)

    # Scan target document and set vector value to 1 if it exists in unique vocabulary list
    for word in input_doc:
        if word in vocab_list:
            result_vec[vocab_list.index(word)] = 1
        else:
            print('the word {} is not in my vocabulary.'.format(word))

    # Return target vector
    return result_vec


def main():
    posting_list, class_vec = load_data()
    my_vocab_list = create_vocab_list(posting_list)
    print(my_vocab_list)

    post_vec_1 = words_2_vec_set(my_vocab_list, posting_list[0])
    post_vec_2 = words_2_vec_set(my_vocab_list, posting_list[3])
    print(post_vec_1)
    print(post_vec_2)


if __name__ == '__main__':
    main()
