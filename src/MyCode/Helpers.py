import InputOutput

header, data_set = InputOutput.data_to_training_set("data.csv")


def unique_vals(rows, col) -> set:
    return set([row[col] for row in rows])


def is_numeric(value) -> bool:
    return isinstance(value, int) or isinstance(value, float)


def class_counts(rows) -> dict:
    counts = {}
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


def partition(rows, question) -> (list, list):
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
        impurity -= prob_of_lbl**2
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

