"""
Useful links:
    http://www.otnira.com/2013/03/25/c4-5/
Pseudocode:
    Check for above base cases.
    For each attribute a, find the normalized information gain ratio from splitting on a.
    Let a_best be the attribute with the highest normalized information gain.
    Create a decision Node that splits on a_best.
    Recur on the sublist obtained by splitting on a_best, and add those nodes as children

Based on:
    https://github.com/ddivad195/PyC4.5
"""
import time

import pandas as pd
import numpy as np
import random as rnd
import pickle


class Node(object):
    """
        Simple node class for a decision tree
        """

    def __init__(self, column: int = -1, attribute: str = '', value: str or float = None,
                 results: str = None, right_child=None, left_child=None, tag: str = ""):
        self.column = column
        self.attribute = attribute
        self.value = value
        self.results = results
        self.right_child = right_child
        self.left_child = left_child
        self.tag = tag


class Decision(object):
    """
        Simple decision to classify with the model
    """

    def __init__(self, path=None, data=None):
        if data is None:
            data = []
        if path is None:
            path = []
        self.path: [str] = path
        self.data: [] = data


def partition_data(dataset: pd.DataFrame, test_percentage) -> tuple:
    """
    Splits the data into training and test partitions randomly
    :param dataset: DataFrame to partition
    :param test_percentage: percentage of the DataFrame for training
    :return:
    """
    total_indexes: [] = range(len(dataset))
    test_indexes: list = rnd.sample(total_indexes, int(test_percentage * len(dataset)))
    train_indexes: list = list(set(total_indexes) ^ set(test_indexes))

    return dataset.loc[train_indexes], dataset.loc[test_indexes]


def get_unique_classes(dataset_column) -> dict:
    """
    Counts the occurrences of the labels in the DataFrame
    :param dataset_column: column of the DataFrame to find occurrences for
    :return: results: a dictionary with the class that occurs and the number of times it occurs for the attribute
    """
    results = {}
    for row in dataset_column:
        if row not in results:
            results[row] = 1
        else:
            results[row] += 1

    return results


def get_entropy(dataset: pd.DataFrame, column) -> float:
    """
    Entropy(Decision) = ∑ – p(I) . log2p(I)
    :return:
    """
    results: dict = get_unique_classes(dataset[column])
    entropy: float = 0.0
    for row in results.values():
        p = float(row) / len(dataset[column])
        entropy -= p * np.log2(p)

    return entropy


def find_split_points(dataset: pd.DataFrame, column: int) -> [list]:
    """
    split_point = place in a dataset attribute the value changes
    Sorts the DataFrame for the given attribute, then find split_points
    :param dataset: DataFrame to find split_points for
    :param column: index of attribute to find the split_points for
    :return: split_points: a list of indexes in the dataset where split_points occur
    """
    classification: str = 'label'
    sorted_values = dataset.sort_values([column], ascending=True)  # sort the DataFrame
    sorted_matrix = sorted_values[[column, classification]].values
    split_points: [list] = []  # to store found split_points
    previous = sorted_matrix[0][1]  # previous value to compare
    indexes = sorted_values.index.values
    counter: int = -1

    for row in sorted_matrix:
        if row[1] != previous:
            split_points.append([indexes[counter], sorted_matrix[counter][0]])
            previous = row[1]  # only change previous when a difference is found
        counter += 1

    return split_points


def split_sets(dataset: pd.DataFrame, column: int, split_points: list):
    """
    Split the DataFrame into subsets based on given split_points
    :param dataset: DataFrame to split the subsets for
    :param column: Attribute index of the DataFrame
    :param split_points:
    :return: sets_below: a list of subsets that contain data below the given split_points
             sets_above: a list of subsets that contain data above the given split_points
    """
    sets_below: list = []
    sets_above: list = []

    for i in range(len(split_points)):
        sets_below.append(dataset[dataset[column] <= dataset[column][split_points[i][0]]])
        sets_above.append(dataset[dataset[column] > dataset[column][split_points[i][0]]])

    return sets_below, sets_above


def get_info_gain(dataset: pd.DataFrame, column) -> tuple:
    """
    Gets the information gain of splitting the data on a given split_point
    :param dataset: DataFrame to find info gain for
    :param column: column of the DataFrame to find info gain for
    :return: best_gain: best information gain for the given column
             sets_below: the subset of the data that is below the split_point that gives the best IG
             sets_above: the subset of the data that is above the split_point that gives the best IG
             split_points: the split_point that gives the best IG
    """
    split_points: list = find_split_points(dataset, column)
    sets_below, sets_above = split_sets(dataset, column, split_points)

    instances_above, entropy_above = [], []
    instances_below, entropy_below = [], []

    classification = 'label'

    target_entropy = get_entropy(dataset, classification)

    for below in sets_below:
        entropy_below.append(get_entropy(below, classification))
        instances_below.append(len(below))

    for above in sets_above:
        entropy_above.append(get_entropy(above, classification))
        instances_above.append(len(above))

    total_instances: list = []
    info_gains: list = []

    for i in range(len(instances_below)):
        total_instances.append(instances_below[i] + instances_above[i])
        probability_below: float = instances_below[i] / float(total_instances[i])
        probability_above: float = instances_above[i] / float(total_instances[i])
        info_gains.append(target_entropy - ((entropy_below[i] * probability_below) + (entropy_above[i] * probability_above)))

    best_gain = i = counter = 0

    for gain in info_gains:
        if best_gain < gain:
            best_gain = gain
            counter = i
        i += 1

    return best_gain, sets_below[counter], sets_above[counter], split_points[counter]


def train(dataset, tag: str = "a") -> Node:
    """
    Train a decision tree model from a dataset
    :param dataset: Dataset to train the model from
    :param tag: Name for a given node (helps keeping track in the graphical representation)
    :return:
    """
    classification: str = "label"  # may change for different datasets

    min_gain: float = -1
    best = {}
    columns: list = []
    i: int = 0

    for column in dataset:
        if column != 'label':
            try:
                info_gain, set1, set2, split = get_info_gain(dataset, column)
                columns.append({'info_gain': info_gain, 'left': set1, 'right': set2,
                                'col': i, 'split': split, 'col_name': column})
            except IndexError:
                columns.append({'info_gain': 0, 'left': [], 'right': [], 'col': column, })
        i += 1

    for value in range(len(columns)):
        if columns[value]['info_gain'] > min_gain:
            best = columns[value]
            min_gain = columns[value]['info_gain']

    left = best['left']
    right = best['right']

    if len(left) != 0 and len(right) != 0:
        return Node(column=best['col'], attribute=best['col_name'], value=best['split'][1], results=None,
                    right_child=train(right, tag=tag+'r'), left_child=train(left, tag=tag+'l'), tag=tag)

    else:
        label: list = list(get_unique_classes(dataset[classification]).keys())
        return Node(results=label[0], tag=tag)


def classify(decision: Decision, tree: Node):
    """
    Classifies a given row with a label
    :param decision:
    :param tree:
    :return:
    """

    target_row: list = decision.data
    decision.path.append(tree.tag)
    if tree.results is not None:
        return tree.results
    else:
        value = target_row[tree.column]
        branch = None
        if isinstance(value, int) or isinstance(value, float):
            branch: Node = tree.right_child if value >= tree.value else tree.left_child
        else:
            branch: Node = tree.right_child if value == tree.value else tree.left_child
        return classify(decision, branch)


def print_tree(tree, spacing=''):
    """
    Print the tree in a elegant and readable CLI format
    :param tree: Root node of the tree to be printed
    :param spacing: Initial spacing, leave at ""
    """
    if tree.results is not None:
        print(spacing[:-1] + "  Predict:" + str(tree.results))

    else:
        print(spacing + str(tree.attribute + " >= " + str(tree.value) + " ?"))

        print(spacing + "├─> Right: ┐ ")
        print_tree(tree.right_child, spacing + "│" + " ")

        print(spacing + "└─> Left: ┐ ")
        print_tree(tree.left_child, spacing + "  ")


def test_tree(data, labels, tree):
    """
    Tests the tree and returns how accurate the classifier is
    :param data: test data we want to classify
    :param labels: list of labels that we want to cross reference the classifier
    result with (stripped out of test_data before it is passed in)
    :param tree: decision tree we want to use for testing
    :return: % accuracy for the tree
    """
    values = []
    # Loop over each row in test data frame and get the classification result for each index
    for index, row in data.iterrows():
        values.append([index, classify(Decision(data=row), tree)])

    # Get the indexes from the test dataframe where each label occurs
    indexes = labels.index.values
    correct = incorrect = 0
    # Loop over values list and compare the class that was classified by the tree
    # and the class that was originally in the dataframe
    for l in range(len(values)):
        if values[l][0] == indexes[l] and values[l][1] == labels[indexes[l]]:
            correct += 1  # increment the correctly classified #
        else:
            incorrect += 1  # increment the incorrectly classified #

    return incorrect, correct, float(100 - (incorrect / (incorrect + correct)) * 100)


def main():
    dataset = pd.read_csv("data_set.csv")

    results = []
    tests = 5  # number of times the algorithm will be run (more runs will give a more accurate average accuracy)
    # loop to test the tree. Each loop it:
    # -> generates random data partitions
    # -> generates a decision tree
    # -> classifies the test_data using this decision tree
    # -> gets the accuracy of the decision tree
    # -> gets the average accuracy, over all the iterations
    trees: [Node] = []
    times = []
    print_time = []
    for i in range(tests):
        train_data, test_data = partition_data(dataset, 0.3)  # random partitions
        """for k in range(20):
            print(test_data[k])"""
        initial_time = time.perf_counter()
        tree = train(train_data)  # make tree
        final_time = time.perf_counter()
        times.append(final_time - initial_time)
        trees.append(tree)
        types = test_data['label']  # get labels column from test_data
        del test_data['label']  # deletes labels from test_data so it cannot be used in classification

        incorrect, correct, accuracy = test_tree(test_data, types, tree)  # test the tree
        results.append(accuracy)

        # print information to console
        print("Test " + str(i + 1) + "\n------------")
        print("Tree Generated:" + "\n")
        print()
        initial_time = time.perf_counter()
        print_tree(tree)
        final_time = time.perf_counter()
        print_time.append(final_time - initial_time)
        print("Correctly Classified: " + str(correct) + " / " + str(correct + incorrect))
        print("Accuracy: " + str(accuracy))
        print()

    # get the average accuracy for all the runs
    summation = 0
    for r in range(len(results)):
        summation += results[r]
    average = summation / tests

    print("Average Accuracy after " + str(tests) + " runs")
    print(average)
    print("Best run: " + str(max(results)))
    print("Worst run: " + str(min(results)))
    print("best time train: " + str(min(times)))
    print("Worst time train: " + str(max(times)))
    print("Mean time train: " + str(sum(times) / len(times)))
    print("best time draw: " + str(min(print_time)))
    print("Worst time draw: " + str(max(print_time)))
    print("Mean time draw: " + str(sum(print_time) / len(print_time)))
    save_model(trees[results.index(max(results))], "Model.tree")


def save_model(model, filename: str):
    """
    Use pickle to serialize a decision tree
    :param model: Model to be serialized
    :param filename: path to file to be saved
    """
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename: str):
    """
    Use pickle to deserialize a decision tree
    :param filename: path to saved model
    :return: a decision tree
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


if __name__ == '__main__':
    main()
