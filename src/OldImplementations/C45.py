"""
    sflorezs1 and jdbuenol implementation of the C45 algorithm for decision trees
"""
import collections
from ctypes import windll
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import math
import numpy as np
import copy
import time


class Data(object):
    """

    Features of every data

    """

    def __init__(self, classifier):
        self.rows = []
        self.attributes = []
        self.attribute_types = []
        self.classifier = classifier
        self.class_col_index = None


class Split(object):
    """

    Is a question

    """

    def __init__(self, attribute_index: int, attribute, attribute_value):
        self.attribute_index = attribute_index
        self.attribute = attribute
        self.attribute_value = attribute_value


class Node(object):
    """
    Basic Node class

    """

    def __init__(self, is_leaf: bool, classification, split, parent, left, right, height: int):
        self.is_leaf = is_leaf
        self.classification = classification
        self.split = split
        self.parent = parent
        self.left = left
        self.right = right
        self.height = height

    def to_json(self):  # New special method.
        """ Convert to JSON format string representation. """
        return '{"is_leaf": "', self.is_leaf, '", "classification": "', self.classification, '", "split": "', \
               self.split, '", "parent": "', self.parent, '", "left": "', self.left, '", "right": "', self.right, \
               '", "height": "', self.height, '"} '


def serialize(root: Node):
    queue: [Node] = [root]
    for node in queue:
        if not node:
            continue
        queue += [node.left, node.right]

    out = ""
    token = "\n"
    for i, item in enumerate(queue):
        if item is not None and not item.is_leaf:
            out += (token if i > 0 else "") + str(item.split.attribute + " >= " + str(item.split.attribute_value))
        elif item is not None and item.is_leaf:
            out += token + item.classification
        elif item is None:
            out += token + '#'
        else:
            out += token + "Error here"
    return out


def deserialize(series: str) -> Node:
    def parse_list() -> [str]:
        series_list: list = series.split("\n")
        return series_list

    def build_node(val: str) -> Node or None:
        if val.__contains__(">="):
            val = val.split(" >= ")
            return Node(is_leaf=False, classification=None,
                        split=Split(attribute=val[0], attribute_value=float(val[1]), attribute_index=-1),
                        parent=None, left=None, right=None, height=0)
        elif val.__contains__("yes") or val.__contains__("no"):
            return Node(is_leaf=True, classification=val,
                        split=None,
                        parent=None, left=None, right=None, height=0)
        else:
            return None

    series_listed: list = parse_list()

    root: Node = build_node(series_listed[0])
    root.height = 0

    queue, i = collections.deque([root]), 1

    while queue:
        node = queue.popleft()
        if node:
            node.left, node.right = map(
                build_node, (series_listed[i], series_listed[i + 1]))
            queue.append(node.left)
            queue.append(node.right)
            i += 2

    def fix_height(tree: Node):
        if tree is None:
            return True
        tree.height = get_height(tree)
        fix_height(tree.left)
        fix_height(tree.right)

    def get_height(some_node: Node):
        if some_node is None:
            return 0
        l_height = get_height(some_node.left)
        r_height = get_height(some_node.right)

        if l_height > r_height:
            return l_height + 1
        else:
            return r_height + 1

    fix_height(root)

    return root


def preprocessing(dataset: Data):
    """
    Convert all attributes that are numeric into float
    :param dataset: Given dataset
    """
    for sample in dataset.rows:
        for x in range(len(dataset.rows[0])):
            if dataset.attribute_types[x] == 'true':
                sample[x] = float(sample[x])


def compute_decision_tree(dataset: Data, parent: Node, classifier):
    """
    Everything may be wrong here
    :param dataset:
    :param parent:
    :param classifier:
    :return:
    """
    node = Node(True, None, None, parent, None, None, 0)
    if parent is None:
        node.height = 0
    else:
        node.height = node.parent.height + 1

    ones = count_positives(dataset.rows, dataset.attributes, classifier)

    if len(dataset.rows) == ones:
        node.classification = "yes"
        node.is_leaf = True
    elif ones == 0:
        node.is_leaf = True
        node.classification = "no"
        return node
    else:
        node.is_leaf = False

    splitting_attribute = None

    maximum_info_gain = 0

    split_value = None
    # minimum_info_gain = 0.00320226097416328
    minimum_info_gain = 0
    entropy = calculate_entropy(dataset, classifier)

    for attribute_index in range(len(dataset.rows[0])):

        if dataset.attributes[attribute_index] != classifier:
            local_maximum_gain: float = 0
            local_split_value = -1
            attribute_value_list = [sample[attribute_index] for sample in dataset.rows]
            attribute_value_list = list(set(attribute_value_list))

            if len(attribute_value_list) > 100:
                attribute_value_list = sorted(attribute_value_list)
                total = len(attribute_value_list)
                ten_percentile = total // 10
                new_list = []
                for x in range(1, 10):
                    new_list.append(attribute_value_list[x * ten_percentile])
                attribute_value_list = new_list

            for value in attribute_value_list:
                current_gain = calculate_information_gain(attribute_index, dataset, value, entropy)

                if current_gain > local_maximum_gain:
                    local_maximum_gain = current_gain
                    local_split_value = value

            if local_split_value > maximum_info_gain:
                maximum_info_gain = local_maximum_gain
                split_value = local_split_value
                splitting_attribute = attribute_index

    if maximum_info_gain <= minimum_info_gain or node.height > 20:
        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)
        return node

    node.split = Split(splitting_attribute, dataset.attributes[splitting_attribute], split_value)

    left_dataset = Data(classifier)
    right_dataset = Data(classifier)

    left_dataset.attributes = dataset.attributes
    right_dataset.attributes = dataset.attributes

    left_dataset.attribute_types = dataset.attribute_types
    left_dataset.attribute_types = dataset.attribute_types

    for row in dataset.rows:
        if splitting_attribute is not None and row[splitting_attribute] >= split_value:
            left_dataset.rows.append(row)
        elif splitting_attribute is not None:
            right_dataset.rows.append(row)

    node.left = compute_decision_tree(left_dataset, node, classifier)
    node.right = compute_decision_tree(right_dataset, node, classifier)

    return node


def classify_leaf(dataset: Data, classifier):
    """

    :param dataset:
    :param classifier:
    :return:
    """
    ones = count_positives(dataset.rows, dataset.attributes, classifier)
    total = len(dataset.rows)
    zeroes = total - ones
    if ones >= zeroes:
        return "yes"
    else:
        return "no"


def get_classification(sample, node: Node, class_col_index):
    if node.is_leaf:
        return node.classification
    else:
        if sample[node.split.attribute_index] >= node.split.attribute_value:
            return get_classification(sample, node.left, class_col_index)
        else:
            return get_classification(sample, node.right, class_col_index)


def calculate_entropy(dataset: Data, classifier):
    ones = count_positives(dataset.rows, dataset.attributes, classifier)

    total_rows = len(dataset.rows)

    entropy = 0.0

    p = ones / total_rows
    if p != 0:
        entropy += p * np.log2(p)

    p = (total_rows - ones) / total_rows
    if p != 0:
        entropy += p * np.log2(p)

    entropy = -entropy
    return entropy


def calculate_information_gain(attribute_index, dataset: Data, val, entropy: float):
    classifier = dataset.attributes[attribute_index]
    attribute_entropy = 0
    total_rows = len(dataset.rows)
    gain_upper_dataset = Data(classifier)
    gain_lower_dataset = Data(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attribute_types = dataset.attribute_types
    gain_lower_dataset.attribute_types = dataset.attribute_types

    for row in dataset.rows:
        if row[attribute_index] >= val:
            gain_upper_dataset.rows.append(row)
        else:
            gain_lower_dataset.rows.append(row)

    if len(gain_upper_dataset.rows) == 0 or len(gain_lower_dataset.rows) == 0:
        return -1

    attribute_entropy += calculate_entropy(gain_upper_dataset, classifier) * len(gain_upper_dataset.rows) / total_rows
    attribute_entropy += calculate_entropy(gain_lower_dataset, classifier) * len(gain_lower_dataset.rows) / total_rows

    return entropy - attribute_entropy


# bad method!!, not anymore :3
def count_positives(instances, attributes, classifier):
    count = 0
    class_col_index = None

    for index in range(len(attributes)):
        if attributes[index] == classifier:
            class_col_index = index
        else:
            class_col_index = len(attributes) - 1
    for instance in instances:
        if instance[class_col_index] == "yes":
            count += 1
    return count


def validate_tree(node, dataset: Data):
    total = len(dataset.rows)
    correct = 0
    for row in dataset.rows:
        correct += validate_row(node, row)
    return correct / total


def validate_row(node: Node, row):
    if node.is_leaf:
        projected = node.classification
        actual = int(row[-1])
        if projected == actual:
            return 1
        else:
            return 0


def prune_tree(root, node, validate_set, best_score):
    if node.is_leaf:
        classification = node.classification
        node.parent.is_leaf = True
        node.parent.classification = node.classification
        if node.height < 20:
            new_score = validate_tree(root, validate_set)
        else:
            new_score = 0

        if new_score >= best_score:
            return new_score
        else:
            node.parent.is_leaf = False
            node.parent.classification = None
            return best_score

    else:
        new_score = prune_tree(root, node.left, validate_set, best_score)
        if node.left:
            return new_score
        new_score = prune_tree(root, node.right, validate_set, new_score)
        if node.is_leaf:
            return new_score

        return new_score


def run_decision_tree():
    """

    :return:
    """
    dataset = Data("")
    training_set = Data("")
    test_set = Data("")

    # Load data set
    # with open("data.csv") as f:
    #    dataset.rows = [tuple(line) for line in csv.reader(f, delimiter=",")]
    # print "Number of records: %d" % len(dataset.rows)

    try:
        # File opener dialog
        windll.shcore.SetProcessDpiAwareness(1)
        Tk().withdraw()
        f = open(askopenfilename())
        original_file = f.read()
        row_splitted_data = original_file.splitlines()
        dataset.rows = [row.split(",") for row in row_splitted_data]

        dataset.attributes = dataset.rows.pop(0)
        print(dataset.attributes)

        dataset.attribute_types = ['true', 'true', 'true', 'true', 'true', 'true', 'false']

        classifier = dataset.attributes[-1]
        dataset.classifier = classifier

        for attribute_index in range(len(dataset.attributes)):
            if dataset.attributes[attribute_index] == dataset.classifier:
                dataset.class_col_index = attribute_index
            else:
                dataset.class_col_index = range(len(dataset.attributes))[-1]

        print("Classifier Id: ", dataset.class_col_index)

        preprocessing(dataset)

        training_set = copy.deepcopy(dataset)
        training_set.rows = []
        test_set = copy.deepcopy(dataset)
        test_set.rows = []
        validate_set = copy.deepcopy(dataset)
        validate_set.rows = []

        runs = 10

        accuracy = []
        start = time.perf_counter()

        trees: list = []

        for run in range(7):
            print("Folding: ", run)
            training_set.rows = []
            for i, x in enumerate(dataset.rows):
                if i % runs != run and i % runs != run + 1 and i % runs != run + 2:
                    training_set.rows.append(x)
            test_set.rows = []
            for i, x in enumerate(dataset.rows):
                if i % runs == run or i % runs == run + 1 or i % runs == run + 2:
                    test_set.rows.append(x)

            print("Number of training records: ", len(training_set.rows))
            print("Number of test records: ", len(test_set.rows))
            root = compute_decision_tree(training_set, None, classifier)
            trees.append(root)

            results: list = []
            for instance in test_set.rows:
                result = get_classification(instance, root, test_set.class_col_index)
                results.append(str(result) == str(instance[-1]))

            acc: float = float(results.count(True)) / float(len(results))
            print("accuracy: %.4f " % acc)

        mean_accuracy = math.fsum(accuracy) / 10
        print("Accuracy  %f " % mean_accuracy)
        print("Took %f secs" % (time.perf_counter() - start))
        # Writing results to a file (DO NOT CHANGE)
        f = open("result.txt", "w")
        # x = json.dumps(trees)
        # f.write(x)
        f.write("accuracy: %.4f" % mean_accuracy)
        f.close()

        tree_to_file(trees[0], "some_tree.tree")

        """for index, root in enumerate(trees):
            print("Tree #", index)
            print_tree(root)"""

        print_tree(trees[0])
    except FileNotFoundError:
        print("File not found")


def print_tree(node: Node, spacing: str = ""):
    if node and node.is_leaf:
        print(spacing[:-1] + "  Predict:", node.classification)
        return
    elif node:

        print(spacing + str(node.split.attribute + " >= " + str(node.split.attribute_value) + " ?"))

        print(spacing + "├─> True: ┐ ")
        print_tree(node.left, spacing + "│" + " ")

        print(spacing + "└─> False: ┐ ")
        print_tree(node.right, spacing + "  ")
    else:
        return


def tree_to_file(root: Node, filename: str):
    tree = serialize(root)
    file = open(filename, "w")
    file.write(tree)
    file.close()


def file_to_tree(path_to_file: str):
    file = open(path_to_file, "r")
    file.readline()  # Skip line
    root: Node = deserialize(file.read())
    return root


if __name__ == "__main__":
    run_decision_tree()
