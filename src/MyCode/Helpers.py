import collections

from InputOutput import data_to_training_set

header, data_set = data_to_training_set("data.csv")


class Node(object):

    def __init__(self, d, left=None, right=None):
        self.data = d
        self.left = left
        self.right = right

    def insert(self, d):
        if self.data == d:
            return False
        elif d < self.data:
            if self.left:
                return self.left.insert(d)
            else:
                self.left = Node(d)
                return True
        else:
            if self.right:
                return self.right.insert(d)
            else:
                self.right = Node(d)
                return True

    def find(self, d):
        if self.data == d:
            return True
        elif d < self.data and self.left:
            return self.left.find(d)
        else:
            return self.right.find(d)
        return False

    def pre_order(self, l: list):
        """
        :param l: the list of data objects so far in the traversal
        :return:
        """
        l.append(self.data)
        if self.left:
            self.left.pre_order(l)
        if self.right:
            self.right.pre_order(l)
        return l

    def in_order(self, l: list):
        """
        :param l: the list of data objects so far in the traversal
        :return:
        """
        if self.left:
            self.left.pre_order(l)
        l.append(self.data)
        if self.right:
            self.right.pre_order(l)
        return l

    def post_order(self, l: list):
        """
        :param l: the list of data objects so far in the traversal
        :return:
        """
        if self.left:
            self.left.pre_order(l)
        if self.right:
            self.right.pre_order(l)
        l.append(self.data)
        return l

    def ask(self, sample: list) -> bool:
        if isinstance(self.data, Question):
            test = sample[self.data.column]
            head = header[self.data.column]
            vs = float(self.data.value)
            if sample[self.data.column] >= float(self.data.value):
                return self.left.ask(sample)
            else:
                return self.right.ask(sample)
        else:
            if isinstance(self.data, dict):
                data = self.data.popitem()[0]
                return data.__contains__('yes')
            return self.data.__contains__('yes')


def unique_vals(rows: list, col: list) -> set:
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def is_numeric(value: str) -> bool:
    """Test if a value is numeric."""
    try:
        value = float(value)
    except ValueError:
        value = value
    return isinstance(value, int) or isinstance(value, float)


def class_counts(rows: list) -> dict:
    """Counts the number of each type of example in a dataset."""
    counts: dict = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example) -> bool:
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self) -> str:
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


def partition(rows: list, question: Question) -> (list, list):
    true_rows: list = []
    false_rows: list = []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows) -> float:
    counts = class_counts(rows)
    impurity: float = 1
    for lbl in counts:
        prob_of_lbl: float = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty) -> float:
    p: float = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows) -> (float, Question):

    best_gain: float = 0
    best_question: Question = None
    current_uncertainty: float = gini(rows)
    n_features: int = len(rows[0]) - 1

    for col in range(n_features):

        values = set([row[col] for row in rows])

        for val in values:

            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf(Node):

    def __init__(self, rows: list):
        self.predictions = class_counts(rows)
        super().__init__(self.predictions)


def build_tree(rows: list):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Node(question, true_branch, false_branch)


def print_tree(node: Node, spacing: str = ""):
    if isinstance(node, Leaf):
        print(spacing[:-1] + "  └Predict:", node.predictions)
        return

    if node.left is None or node.right is None:
        print(spacing[:-1] + "  └Predict:", node.data)
        return

    print(spacing + str(node.data))

    print(spacing + "├-> True:")
    print_tree(node.left, spacing + "|" + " ")

    print(spacing + "└-> False:")
    print_tree(node.right, spacing + "  ")


def serialize(root: Node):
    queue = [root]
    for node in queue:
        if not node:
            continue
        queue += [node.left, node.right]

    return str(header) + '\n' + ';;'.join(map(lambda item: str(item.data) if item else '#', queue))


def tree_to_file(root: Node):
    tree: list = serialize(root)


def deserialize(serie: str) -> Node:

    def parse_list() -> (list, list):
        headers = serie.split("\n")[0].strip("[]").split(", ")
        headers = [n.strip("\'") for n in headers]
        serie_list = serie.split("\n")[1].split(";;")
        return headers, serie_list

    def build_node(val: str):
        if val.__contains__("{") or val.__contains__("}"):
            val = val.strip("{}").replace(' :', '').split(" ")
            thing: Node = Node(val[0].replace(":", "").strip("\'"))
            return thing
        elif val.__contains__("#"):
            return None
        else:
            val = val.split(" ")
            item = Node(Question(h.index(val[1]), val[3][:-1]))
            return item

    h, pre_order = parse_list()

    root: Node = build_node(pre_order[0])

    queue, i = collections.deque([root]), 1

    while queue and i < len(pre_order):
        node = queue.popleft()
        if node:
            node.left, node.right = map(
                build_node, (pre_order[i], pre_order[i + 1]))
            queue.append(node.left)
            queue.append(node.right)
            i += 2
    return root

