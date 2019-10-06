from InputOutput import serialize
from Tree import build_tree, print_tree
from MyCode.Helpers import data_set

my_tree = build_tree(data_set)
print_tree(my_tree)

print(serialize(my_tree))
