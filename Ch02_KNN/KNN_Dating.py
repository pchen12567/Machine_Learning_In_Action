import numpy as np
import matplotlib.pyplot as plt
from Ch02_KNN import KNN


# Parse the input file
def file2matrix(filename):
    with open(filename) as f:
        # Read file by line
        array_lines = f.readlines()

        # Get the length of lines
        number_line = len(array_lines)

        # Init result matrix by 0, shape(length of lines, 3), data type = float
        result_matrix = np.zeros((number_line, 3), dtype=float)

        # Init label vector
        label_vector = []

        # Init label map
        label_map = {'didntLike': 1,
                     'smallDoses': 2,
                     'largeDoses': 3}

        # Init index in matrix
        index = 0

        for line in array_lines:
            # Get info for each line
            info = line.strip().split('\t')

            # Get factors from info, the first 3 ones
            result_matrix[index, :] = info[:3]

            # Get label from info, the last one
            label_vector.append(label_map[info[-1]])

            index += 1

    return result_matrix, label_vector


# Build function to normalization the input data
def auto_norm(data):
    # Get minimum value
    min_val = data.min(0)

    # Get maximum value
    max_val = data.max(0)

    # Get ranges
    range_value = max_val - min_val

    # Get the length of data
    m = data.shape[0]

    # Normalization
    norm_data = data - np.tile(min_val, (m, 1))
    norm_data = norm_data / np.tile(range_value, (m, 1))

    return norm_data, range_value, min_val


#
def dating_classify(file):
    # Parse file to get data and label
    dating_data, dating_label = file2matrix(file)

    # Normalization data
    norm_data, ranges, min_value = auto_norm(dating_data)

    # Set test number
    ratio = 0.1
    m = norm_data.shape[0]
    num_test = int(m * ratio)

    # Init error counter
    count = 0

    for i in range(num_test):
        classify_result = KNN.knn_classify(X=norm_data[i, :], data=norm_data[num_test:m, :],
                                           labels=dating_label[num_test:m], k=3)

        if classify_result != dating_label[i]:
            count += 1

        print('The classifier came back with: {}, the real answer is:{}'.format(classify_result, dating_label[i]))

    print('The total error rate is: {}%'.format((count / num_test) * 100))


# Test
path = './data/datingTestSet.txt'
dating_classify(path)

# path = './data/datingTestSet.txt'
# dating_matrix, dating_label = file2matrix(path)
#
# dating_matrix, ranges, min_val = auto_norm(dating_matrix)
#
# print("Data after normalization: \n", dating_matrix[:10])
# print('Data ranges of each factor: ', ranges)
# print('Min value of each factor: ', min_val)
# print('Labels: ', dating_label[:10])

# Visualization
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dating_matrix[:, 1], dating_matrix[:, 2], 10*np.array(dating_label),  10*np.array(dating_label))
# plt.xlabel('Percentage of Time Spent Playing Video Game')
# plt.ylabel('Liter of Ice Cream Consumed Per Week')
# plt.show()
