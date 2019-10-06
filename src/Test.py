from InputOutput import serialize
from MyCode.Helpers import data_set, build_tree, print_tree

my_tree = build_tree(data_set)
print_tree(my_tree)

print(serialize(my_tree))
