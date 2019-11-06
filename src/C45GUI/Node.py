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

    def __init__(self, path=None, data=None):
        if data is None:
            data = []
        if path is None:
            path = []
        self.path: [str] = path
        self.data: [] = data
