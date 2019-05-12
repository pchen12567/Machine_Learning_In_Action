"""
@Coding: uft-8 *
@Time: 2019-05-12 13:47
@Author: Ryne Chen
@File: KNN_SKlearn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('./data/datingTestSet.txt', '\t', header=None,
                 names=['fly_mile', 'game_time', 'ice_cream', 'label'])

# Init label map
label_map = {'didntLike': 1,
             'smallDoses': 2,
             'largeDoses': 3}

# Change value of label
df['label'] = df['label'].apply(lambda x: label_map[x])

# Split features and label
X = df.iloc[:, 0:3]
y = df.iloc[:, -1]

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Sample Number：{}，Train Number：{}，Test Number：{}'.format(len(X), len(X_train), len(X_test)))

# Init k list
k_neighbors = [3, 5, 7, 9]
acc_list = []

# Training
for k in k_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    print('K = {}, Prediction Accuracy: {}'.format(k, acc))
