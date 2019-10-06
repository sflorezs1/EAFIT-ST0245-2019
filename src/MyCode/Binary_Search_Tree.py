from MyCode.Helpers import *


class BinarySearchTree(object):

    def __init__(self, root: Node = None):
        self.root = root

    def insert(self, d):
        """
        :param d: Data to insert
        :return: True if successfully inserted, false if exists
        """
        if self.root:
            return self.root.insert(d)
        else:
            self.root = Node(d)

    def find(self, d):
        """
        :param d: Data to find
        :return: True if d is found in the tree, false otherwise
        """
        if self.root:
            return self.root.find(d)
        else:
            return False

    def remove(self, d):
        """
        :param d: Data to remove
        :return: True if node successfully removed, False otherwise
        """
        # Case 1: Empty Tree
        if self.root is None:
            return False

        # Case 2: Deleting root node
        if self.root.data == d:
            # Case 2.1: Root node has no children
            if self.root.left is None and self.root.right is None:
                self.root = None
                return True
            # Case 2.2: Root node has left child
            if self.root.left and self.root.right is None:
                self.root = self.root.left
                return True
            # Case 2.3: Root node has right child
            if self.root.left is None and self.root.right:
                self.root = self.root.right
                return True
            # Case 2.4: Root node has two children
            else:
                move_node: Node = self.root.right
                move_node_parent: Node = None
                while move_node.left:
                    move_node_parent = move_node
                    move_node = move_node.left
                if move_node.data < move_node_parent.data:
                    move_node_parent.left = None
                else:
                    move_node_parent.right = None
                return True
        # Find node to remove
        parent: Node = None
        node: Node = self.root
        while node and node.data != d:
            parent = node
            if d < node.data:
                node = node.left
            elif d > node.data:
                node = node.right
        # Case 3: Node not found
        if node is None or node.data != d:
            return False
        # Case 4: Node has no children
        elif node.left is None and node.right is None:
            if d < parent.data:
                parent.left = None
            else:
                parent.right = None
        # Case 5: Node has left child only
        if node.left and node.right is None:
            if d < parent.data:
                parent.left = node.left
            else:
                parent.right = node.left
            return True
        # Case 6: Node has right child only
        if node.left is None and node.right:
            if d < parent.data:
                parent.left = node.right
            else:
                parent.right = node.right
            return True
        # Case 7: Node has both children
        else:
            move_node: Node = node.right
            move_node_parent: Node = node
            while move_node.left:
                move_node_parent = move_node
                move_node = move_node.left
            node.data = move_node.data
            if move_node.right:
                if move_node.data < move_node_parent.data:
                    move_node_parent.left = move_node.right
                else:
                    move_node_parent.right = move_node.right
            else:
                if move_node.data < move_node_parent.data:
                    move_node_parent.left = None
                else:
                    move_node_parent.right = None
        return True

    def pre_order(self):
        """
        :return: list of data elements resulting from pre-order tree traversal
        """
        if self.root:
            return self.root.pre_order([])
        else:
            return []

    def post_order(self):
        """
        :return: list of data elements resulting from post-order tree traversal
        """
        if self.root:
            return self.root.post_order([])
        else:
            return []

    def in_order(self):
        """
        :return: list of data elements resulting from in-order tree traversal
        """
        if self.root:
            return self.root.in_order([])
        else:
            return []

    def ask(self, sample: list):
        """
        :param sample:
        :return: boolean value for the asked data
        """
        if self.root:
            return self.root.ask(sample)
        else:
            return False
