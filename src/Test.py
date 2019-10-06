from MyCode.Helpers import data_set, build_tree, print_tree, serialize, deserialize
from MyCode.Binary_Search_Tree import BinarySearchTree
my_tree = build_tree(data_set)

print(serialize(my_tree))
print_tree(my_tree)
print("================================STOOOP========================================================")
print_tree(deserialize(str(serialize(my_tree))))
print("Will this plant " + str([6.69,22.25,48.19,3496.0,24.0,99.0]) + " die? ")
print(str(BinarySearchTree(my_tree).ask([6.69,22.25,48.19,3496.0,24.0,99.0])))
