from MyCode.Helpers import *
from MyCode.Node import Node


class Leaf:

    def __init__(self, rows):
        self.predictions = class_counts(rows)


def build_tree(rows: list):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Node(question, true_branch, false_branch)


"""
def build_tree_from_list(tree: list):
    
"""


def print_tree(node: Node, spacing: str = ""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.data))

    print(spacing + "--> True:")
    print_tree(node.left, spacing + " ")

    print(spacing + "--> False:")
    print_tree(node.right, spacing + " ")
