"""
@Coding: uft-8
@Time: 2019-05-05 14:44
@Author: Ryne Chen
@File: trees.py
"""

from math import log
from collections import defaultdict


# Build function to calculate entropy
def calc_entropy(data):
    # Get total number of data
    num = len(data)

    # Init label counts dict
    label_counts = defaultdict(int)

    # Get total number of each label in data
    for factor_vec in data:
        # Get current label
        current_label = factor_vec[-1]

        # Save to label dict
        label_counts[current_label] += 1

    # Init entropy value
    entropy = 0.0

    # Calculate entropy
    for value in label_counts.values():
        # Calculate probability of each label in data
        prob = float(value) / num

        # Shannon entropy
        entropy -= prob * log(prob, 2)

    return entropy


# Build function to split data by specified factor
def split_data(data, factor_index, factor_value):
    # Init return data list
    ret_data = []

    # Split data by specified factor
    for factor_vec in data:

        # Check if factor value match the specified factor_value
        if factor_vec[factor_index] == factor_value:
            # Get the remaining factors
            remain_factor = factor_vec[: factor_index]

            remain_factor.extend(factor_vec[factor_index + 1:])

            # Save to return data list
            ret_data.append(remain_factor)

    # Return split result
    return ret_data


# Build function to get best split factor position
def best_split_factor(data):
    # Get total number of factors
    factor_num = len(data[0]) - 1

    # Calculate base entropy of total data
    base_entropy = calc_entropy(data)

    # Init information gain
    best_info_gain = 0.0

    # Init best split factor index
    best_factor_index = -1

    # Scan each factor
    for i in range(factor_num):

        # Get each factor values
        factor_values = [sample[i] for sample in data]

        # Get unique values of each factor
        unique_vals = set(factor_values)

        # Init entropy
        new_entropy = 0.0

        for value in unique_vals:
            # Split data by specified factor i and value
            sub_data = split_data(data, i, value)

            # Calculate probability of each split sub data
            prob = len(sub_data) / float(len(data))

            # Calculate entropy of each split sub data
            new_entropy += prob * calc_entropy(sub_data)

        # Calculate information gain
        info_gain = base_entropy - new_entropy

        # Update best information gain
        if info_gain > best_info_gain:
            best_info_gain = info_gain

            # Get index of best factor
            best_factor_index = i

    return best_factor_index


# Build function to get majority label
def majority_cnt(label_list):
    # Init label count dict
    label_count = defaultdict(int)

    for label in label_list:
        label_count[label] += 1

    # sort label count dict by number of each label
    sorted_label = sorted(label_count.items(), key=lambda item: item[1], reverse=True)

    # Return the majority label
    return sorted_label[0][0]


def create_tree(data, factor_names):
    # Get total labels in data
    label_list = [sample[-1] for sample in data]

    # Return label if just one class
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]

    # Return majority factor if complete scan all factors
    if len(data[0]) == 1:
        return majority_cnt(label_list)

    # Get best split factor position
    best_factor_index = best_split_factor(data)

    # Get best split factor name
    best_factor_name = factor_names[best_factor_index]

    # Init tree dict
    my_tree = defaultdict(dict)

    # Delete best split factor name in factor_names
    del (factor_names[best_factor_index])

    # Get all values of best split factor in data
    factor_values = [sample[best_factor_index] for sample in data]

    # Get the unique values
    unique_values = set(factor_values)

    # Scan each unique value of best split factor
    for factor_value in unique_values:
        # Save remaining factor names
        sub_factor_names = factor_names[:]

        # Save tree result
        my_tree[best_factor_name][factor_value] = create_tree(split_data(data, best_factor_index, factor_value),
                                                              sub_factor_names)

    return my_tree


def create_data():
    data = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]

    factor_names = ['no surfacing', 'flippers']

    return data, factor_names


def main():
    my_data, my_factor_names = create_data()
    print('My Data: ', my_data)
    # my_entropy = calc_entropy(my_data)
    # print('My Entropy: ', my_entropy)

    # split_1 = split_data(my_data, 0, 1)
    # print(split_1)
    # split_2 = split_data(my_data, 0, 0)
    # print(split_2)

    # best_factor = best_split_factor(my_data)
    # print('My best split factor position is: ', best_factor)

    my_tree = create_tree(my_data, my_factor_names)
    print(my_tree)


if __name__ == '__main__':
    main()
