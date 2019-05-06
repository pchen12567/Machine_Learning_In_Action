"""
@Coding: uft-8 *
@Time: 2019-05-06 15:56
@Author: Ryne Chen
@File: DT_LensesType 
"""

from Ch03_DT import trees


def predictLensesType(path):
    # Init factor names
    lenses_factors = ['age', 'prescript', 'astigmatic', 'tearRare']

    with open(path) as f:
        # Get sample data
        lenses = [sample.strip().split('\t') for sample in f.readlines()]

        # Build tree
        lenses_tree = trees.create_tree(lenses, lenses_factors)

        return lenses_tree


def main():
    # Set file path
    path = './data/lenses.txt'

    lenses_tree = predictLensesType(path)
    print(lenses_tree)


if __name__ == '__main__':
    main()
