"""
@Coding: uft-8 *
@Time: 2019-05-15 18:27
@Author: Ryne Chen
@File: NB_SKlearn 
"""
from Ch04_NB import bayes
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Function to parse input text
def text_parse(sentence):
    # Set pattern
    reg_ex = re.compile(r'\W+')

    # Split text by pattern
    tokens = reg_ex.split(sentence)

    # Get token list with specified rules: lower() and len() > 2
    token_list = [token.lower() for token in tokens if len(token) > 2]

    return token_list


# Function to prepare data and label
def pre_data():
    # Init container to save files
    file_list = []
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
            # Set label of spam email as 1
            label_list.append(1)

        with open('./data/email/ham/6.txt', 'rb') as f_ham:
            # Read normal email
            content = f_ham.read().decode('utf-8', errors='ignore')
            # Parse file and get words list
            token_list = text_parse(content)
            # Save normal email file
            file_list.append(token_list)
            # Set label of normal email as 0
            label_list.append(0)

    # Get unique vocabulary list of total words
    vocab_list = bayes.create_vocab_list(file_list)

    file_matrix = []
    # Scan the remaining 40 files
    for file in file_list:
        # Get file vector
        file_vec = bayes.words_2_vec_set(vocab_list, file)
        # Save to file vector matrix
        file_matrix.append(file_vec)

    return file_matrix, label_list


def main():
    # Get data
    X, y = pre_data()

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print('Sample Number：{}，Train Number：{}，Test Number：{}'.format(len(X), len(X_train), len(X_test)))

    # Init model
    nb_model = GaussianNB()

    # Training
    nb_model.fit(X_train, y_train)

    # Predict
    y_pred = nb_model.predict(X_test)

    # Calculate score
    acc = accuracy_score(y_test, y_pred)
    print('Prediction Accuracy: {}'.format(acc))


if __name__ == '__main__':
    main()
