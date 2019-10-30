import time

import pandas as pd
import numpy as np
import random as rnd
from C45GUI.Node import Node, Decision
import pickle


targetEntropy = 0


def partition_data(data_frame, test_percentage) -> ():
    """
    Partition dataSet into training and test partitions
    data_frame: dataset we want to partition
    test_percentage: percentage of the dataset we want to use for testing
    """
    tot_index = range(len(data_frame))  # generate array with values 1 to length of dataset
    test_indexes = rnd.sample(tot_index, int(test_percentage * len(data_frame))) \
        # get a random sample of indices from this array and by multiplying by % split we want
    train_indexes = list(set(tot_index) ^ set(test_indexes)) \
        # take away the test indexes from the the train indexes using the intersection between the sets total and train

    # use the indexes generated to build up dataframes
    test_df = data_frame.loc[test_indexes]
    train_df = data_frame.loc[train_indexes]

    return train_df, test_df


def get_unique_classes(data_frame_column) -> dict:
    """
    counts the occurrences of the labels in the dataset
    :param data_frame_column: column of the dataset we want to find occurrences for
    :return: results: a dictionary with the class that occurs and the number of times it occurs in the column
    """
    results = {}
    for row in data_frame_column:
        if row not in results:
            results[row] = 0  # initialise element in dictionary if not already there
        results[row] += 1  # increment value

    return results


def get_entropy(data, column):
    """
    gets the information entropy for a column in the dataset
    :param data: dataset to get the entropy for
    :param column: column currently under investigations
    :return: entropy: entropy value for the given column
    """
    entropy = 0.0
    results = get_unique_classes(data[column])  # get # of classes in data

    for row in results.values():
        p = float(row) / len(data[column])  # calculate probability value for each element in results
        entropy -= p * np.log2(p)  # calculate entropy

    return entropy


def find_split_points(data, column: int):
    """
    Finds the thresholds(split_points) in a dataset for a given column
    This method splits the data where a change in the decision class occurs
    rather than splitting at every row (brute force method)
    This makes it more efficient as it generates less splitpoints for the data
    :param data: dataset to find split_points for
    :param column: column we want to find thresholds for
    :return: split_points: a list of indexes in the dataframe where split_points occur
    """
    sorted_values = data.sort_values([column], ascending=True)
    sorted_matrix = sorted_values[[column, 'label']].as_matrix()
    split_points = []
    previous = sorted_matrix[0][1]  # get target of the first first element in sorted_values matrix
    index = sorted_values.index.values  # get the indexes of each  element in the sorted_values matrix
    counter = 0
    for row in sorted_matrix:
        if row[1] != previous:  # if the label (class) of row 1 = the previous label
            split_points.append([index[counter - 1], sorted_matrix[counter - 1][0]])
        counter += 1
        previous = row[1]

    return split_points  # indexes of the dataframe where splits should occur


def split_sets(data, column, split_points):
    """
    Splits the dataset into subsets based on thresholds
    :param data: dataset to find subsets for
    :param column: column of the dataset we are currently subsetting
    :param split_points: an index of thresholds to use when subsetting the data
    :return: sets_below: a list of dataframes that contain data below the given split_points
             sets_above: a list of dataframes that contain data above the given split_points
    """
    sets_below = []
    sets_above = []
    # split the dataframe into 2 for each splitpoint
    for i in range(len(split_points)):
        df1 = data[data[column] <= data[column][split_points[i][0]]]  # everything below the splitpoint
        df2 = data[data[column] > data[column][split_points[i][0]]]  # everything above it
        # add to the lists
        sets_below.append(df1)
        sets_above.append(df2)

    return sets_below, sets_above


def get_information_gain(data, column) -> ():
    """
    Gets the information gain of splitting the data on a given split_point in the data
    :param data: Data to train the model
    :param column:
    :return: best_gain: best information gain for the given column
             sets_below: the subset of the data that is below the split_point that gives the best IG
             sets_above: the subset of the data that is above the split_point that gives the best IG
             split_points: the split_point that gives the best IG
    """
    split_points = find_split_points(data, column)  # get split_points for this column
    sets_below, sets_above = split_sets(data, column,
                                        split_points)  # split the data into sets based on these split_points
    # lists to store the # of instances in each subset that are above and below each given threshold and their entropies
    instances_above = []
    instances_below = []
    entropy_above = []
    entropy_below = []
    target_entropy = get_entropy(data, 'label')  # get target entropy for the dataset
    # get entropy for sets above and below each of the thresholds
    for sample in sets_below:
        entropy_below.append(get_entropy(sample, 'label'))
        instances_below.append(len(sample))
    for sample in sets_above:
        entropy_above.append(get_entropy(sample, 'label'))
        instances_above.append(len(sample))

    total_instances = []
    info_gains = []
    # work out the Information Gain for each threshold
    for i in range(len(instances_below)):
        total_instances.append(instances_below[i] + instances_above[i])
        prob_a = (instances_above[i] / float(total_instances[i]))
        prob_b = (instances_below[i] / float(total_instances[i]))
        info_gains.append(target_entropy - ((entropy_below[i] * prob_b) + (entropy_above[i] * prob_a)))

    # work out the highest information gain for this column of the dataset
    best_gain = i = counter = 0
    for gain in info_gains:
        if best_gain < gain:
            best_gain = gain
            counter = i  # variable to hold the index in the list where the best gain occurs
        i += 1

    return best_gain, sets_below[counter], sets_above[counter], split_points[counter]


def train(data, tag="a") -> Node:
    """
    Build the tree
    :param data: Data set to train the model
    :param tag: Name of the node (just for the graphical representation)
    :return: Decision tree for the given dataset
    """
    optimal_gain = -1
    best = {}
    columns = []
    i = 0

    for column in data:  # loop over each attribute
        if column != 'label':
            try:
                ig, set1, set2, split = get_information_gain(data, column)  # get information gain for each column
                # column holds information that is used when creating a tree node.
                # the values in each of the columns will be used below when creating nodes for the tree
                columns.append({"ig": ig, "left": set1, "right": set2, 'col': i, 'split': split,
                                'colName': column})  # append attributes to list to generate tree node

            # above code will work until the set1 and set2 values that would
            # be returned bu the informationGain function will be 0
            # in that case, an indexError will be thrown as we cannot access element 0 of the sets lists
            # so if we catch that exception we know this data should be
            # used as a leaf node and can format the tree information accordingly
            except IndexError:
                columns.append({"ig": 0, "left": [], "right": [], 'col': column, })
        i += 1  # counter to get int value for row(used for tree node)

    # loops through each column and pulls out the one with the best information gain for the given data
    for val in range(len(columns)):

        if columns[val]['ig'] > optimal_gain:
            best = columns[val]
            optimal_gain = columns[val]['ig']

    # get data for left branch and data for right branch
    left = best['left']
    right = best['right']
    # check if we have data for the left and right branches of the tree
    # if they are = 0 it is the stop condition for recursion, and the else block will generate a leaf node for the tree
    if len(best['left']) != 0 and len(best['right']) != 0:
        return Node(column=best['col'], attribute=best['colName'], value=best['split'][1], results=None,
                    right_child=train(right, tag=tag+"r"), left_child=train(left, tag=tag+"l"), tag=tag)

    else:
        label = list(get_unique_classes(data['label']).keys())  # get label for the leaf node
        return Node(results=label[0], tag=tag)


def classify(desicion: Decision, tree):
    """
    Traverse the tree to find the leaf node that the target_row will be classified as
    :param desicion: A decision set for classifying
    :param tree: decision tree
    :return:
    """
    target_row = desicion.data
    desicion.path.append(tree.tag)
    if tree.results is not None:
        return tree.results

    else:
        # gets the attribute from the target row that we are looking at
        val = target_row[tree.column]
        branch = None
        if isinstance(val, int) or isinstance(val, float):
            # checks the value of the tree against the value of the attribute from the target row
            # go down right side
            if val >= tree.value:
                branch = tree.right_child
            # go down left side
            else:
                branch = tree.left_child
        # recur over the tree again, using either the left or right branch to determine where to go next
        return classify(desicion, branch)


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

    return incorrect, correct, np.round(100 - (incorrect / (incorrect + correct)) * 100)


def main():
    # Read data_set from file into a dataframe
    # Any dataset can be used, as long as the last column is the result
    # And the columns have headings, with the last column called 'label'
    data_set = pd.read_csv('data_set.csv')

    results = []
    tests = 1  # number of times the algorithm will be run (more runs will give a more accurate average accuracy)
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
        train_data, test_data = partition_data(data_set, 0.3)  # random partitions
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
