"""
@Coding: uft-8 *
@Time: 2019-05-12 14:16
@Author: Ryne Chen
@File: DT_SKlearn.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('./data/lenses.txt', '\t', header=None,
                 names=['age', 'prescript', 'astigmatic', 'tearRare', 'label'])

# Split features and label
X = df.iloc[:, :4]
y = df.iloc[:, -1]

# One_hot encoding features
X = pd.get_dummies(X)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Sample Number：{}，Train Number：{}，Test Number：{}'.format(len(X), len(X_train), len(X_test)))

# Init max depth list
depth_list = [2, 3, 4, 5]
acc_list = []

# Training
for d in depth_list:
    dt_model = DecisionTreeClassifier(max_depth=d)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    print('Depth = {}, Prediction Accuracy: {}'.format(d, acc))
