from MyCode.Helpers import Question


class Decision:

    def __init__(self, question: Question):
        self.question = question


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
