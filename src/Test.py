from MyCode.Helpers import data_set, build_tree, print_tree, serialize, deserialize

my_tree = build_tree(data_set)

print(serialize(my_tree))

print_tree(deserialize(str(serialize(my_tree))))
