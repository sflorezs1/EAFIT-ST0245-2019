from MyCode.Helpers import data_set, build_tree, print_tree, serialize, deserialize
from MyCode.Binary_Search_Tree import BinarySearchTree
my_tree = build_tree(data_set)

print_tree(my_tree)

file = open('data.csv')
file.readline()  # Skip header

accuracy = 0

for line in file:
    l = line.split(",")[:-1]
    l = [float(n) for n in l]
    e = True if line.split(",")[-1] == 'yes' else False
    r = BinarySearchTree(my_tree).ask(l)
    print("Will this plant " + str(l) + " die? EXPECTED: " + str(e))
    print(str(r))
    if e == r:
        accuracy += 1

print("Accuracy = " + str(accuracy * 100 / 672) + "%")
