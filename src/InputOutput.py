import ast

from Coffee_Plant import CoffeePlant
from MyCode import Node


def line_to_plant(line: str) -> CoffeePlant:
    attributes: list = line.split(",")
    ph: float = float(attributes[0])
    soil_temperature: float = float(attributes[1])
    soil_moisture: float = float(attributes[2])
    illuminance: float = float(attributes[3])
    env_temperature: float = float(attributes[4])
    env_humidity: float = float(attributes[5])
    label: bool = True if attributes[6][0:3] == "yes" else False
    return CoffeePlant(ph, soil_temperature, soil_moisture, illuminance, env_temperature, env_humidity, label)


def gather_data(path: str) -> list:
    try:
        file = open(path, "r")
        plants: list = []
        file.readline()  # to skip header of the file
        for line in file:
            plants.append(line_to_plant(line))
    except FileNotFoundError:
        print("Wrong path, '" + path + "' file, does not exist")
        return None
    finally:
        file.close()
        return plants


def data_to_training_set(path: str) -> (list, list):
    try:
        file = open(path, "r")
        headers: list = file.readline().rstrip("\n").split(",")
        data_set: list = [[0] * len(headers)] * (sum(1 for line in open(path)) - 1)
        for row in range(len(data_set)):
            data_set[row] = file.readline().rstrip('\n').split(",")
    except FileNotFoundError:
        print("Wrong path, " + path + " file, does not exist")
        return None, None
    finally:
        file.close()
        print(headers)
        return headers, data_set[:-1]


def serialize(root: Node) -> list:
    return root.pre_order([])


def tree_to_file(root: Node):
    tree: list = serialize(root)


def deserialize(serie: list) -> Node:

    def parse_list() -> list:
        serie_list: list = ast.literal_eval(serie)
        serie_list = [n.strip() for n in serie_list]
        return serie_list

    def build_node(val: str):


    pre_order = parse_list()

    root: Node = Node(pre_order[0])


    return None



