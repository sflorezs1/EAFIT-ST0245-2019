from Helpers import *


class Leaf:

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class DecisionNode:

    def __init__(self, question: Question, true_branch, false_branch):
        self.question: Question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return DecisionNode(question, true_branch, false_branch)


def print_tree(node: DecisionNode, spacing: str = ""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.question))

    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + " ")

    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + " ")
