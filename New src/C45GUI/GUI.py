from ctypes import windll
from tkinter import *
from tkinter.filedialog import askopenfilename
import time
import pandas as pd
from C45GUI import pyC45reimplementation, Node
from tkinter import messagebox
import graphviz
import os

from C45GUI.Node import Decision

os.environ["PATH"] += os.pathsep + 'Graphviz2.38\\bin'  # Add the Graphviz binaries to the PATH
                                                        # (This may not work outside Windows)


class GUI(object):
    log: str = ''

    def __init__(self, width, height):
        windll.shcore.SetProcessDpiAwareness(1)  # This helps when the window is upscaled, only works in Windows
        self.graph = graphviz.Digraph(comment="Decision Tree Model")
        self.window = Tk()
        self.width = width
        self.height = height
        self.canvas = Canvas(self.window, width=self.width, height=self.height, bg="white")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas_image = None
        self.arrows = None
        self.tree = None

    def gui_init(self):

        def show_move():
            """
            Show the arrows
            :return:
            """
            self.arrows = PhotoImage(file="Resources/Images/DirArr.png")
            self.canvas.create_image(self.arrows.width()/2, self.arrows.height(),
                                     anchor=CENTER, image=self.arrows, tags="arrows")
            self.canvas.pack(fill=BOTH, expand=True)

        def resize(event):
            """
            Give the canvas a minimum size and readjusts it
            :param event:
            :return:
            """
            event.width = event.width if event.width >= 800 else 800
            event.height = event.height if event.height >= 600 else 600
            w, h = event.width - 100, event.height - 100
            self.canvas.config(width=w, height=h)

        def move(event):
            """
            Allows to move the image with the arrows
            :param event:
            :return:
            """
            if event.keysym == "Right":
                self.canvas.move("image", -50, 0)
            elif event.keysym == "Left":
                self.canvas.move("image", 50, 0)
            elif event.keysym == "Down":
                self.canvas.move("image", 0, -50)
            elif event.keysym == "Up":
                self.canvas.move("image", 0, 50)

        self.canvas.bind("<Configure>", resize)
        self.canvas_image = PhotoImage(file="Resources/Images/Logo.png")
        self.canvas.create_image(self.canvas_image.width()/2, self.canvas_image.height()/2,
                                 anchor=CENTER, image=self.canvas_image, tags="image")
        self.canvas.pack(fill=BOTH, expand=True)
        self.window.title('Rust Prediction App')
        frame_1 = Frame(self.window)
        frame_1.pack()
        welcome_text: str = "Welcome to our Rust Predicting App!\n"
        welcome = Label(frame_1, text=welcome_text, font=("Times New Roman", 14))
        welcome.pack()
        decision_text: str = "Do you want to train with a new dataset " \
                             "(.csv file)? Or use an existent model (.tree file)?"
        decision = Label(frame_1, text=decision_text, font=("Times New Roman", 13))
        decision.pack()

        def get_data_input():
            """
            Get the input for a new classification in the model
            :return:
            """
            get_input_text: str = "Enter the data for the plant.\n"
            frame_2 = Frame(self.canvas)
            get_input = Label(frame_2, text=get_input_text)
            get_input.grid(row=0)
            # for all features of the coffee plant
            Label(frame_2, text="Insert the data, ").grid(row=1, column=0)
            Label(frame_2, text="then press 'Calculate' to execute.").grid(row=2, column=0)
            Label(frame_2, text="ph: ").grid(row=1, column=1)
            Label(frame_2, text="soil_temperature: ").grid(row=1, column=2)
            Label(frame_2, text="soil_moisture: ").grid(row=1, column=3)
            Label(frame_2, text="illuminance: ").grid(row=1, column=4)
            Label(frame_2, text="env_temperature: ").grid(row=1, column=5)
            Label(frame_2, text="env_humidity: ").grid(row=1, column=6)
            Label(frame_2).grid(row=3)
            frame_2.pack(fill=BOTH)

            get_ph = Entry(frame_2, width=14)
            get_ph.grid(row=2, column=1)
            get_soil_temperature = Entry(frame_2, width=14)
            get_soil_temperature.grid(row=2, column=2)
            get_soil_moisture = Entry(frame_2, width=14)
            get_soil_moisture.grid(row=2, column=3)
            get_illuminance = Entry(frame_2, width=14)
            get_illuminance.grid(row=2, column=4)
            get_env_temperature = Entry(frame_2, width=14)
            get_env_temperature.grid(row=2, column=5)
            get_env_humidity = Entry(frame_2, width=14)
            get_env_humidity.grid(row=2, column=6)

            def calculate():
                """
                Classify a plant with the given user input
                :return:
                """
                plant: Decision = Decision()

                def gather_data():
                    """
                    Gather the data from the text entries and convert it to floating point numbers
                    :return:
                    """
                    plant.data.append(get_ph.get())
                    plant.data.append(get_soil_temperature.get())
                    plant.data.append(get_soil_moisture.get())
                    plant.data.append(get_illuminance.get())
                    plant.data.append(get_env_temperature.get())
                    plant.data.append(get_env_humidity.get())
                    try:
                        plant.data = [float(i) for i in plant.data]
                    except ValueError:
                        answer = messagebox.askretrycancel("Question", "Incorrect input, try again?")
                        if answer:
                            frame_2.destroy()
                            get_data_input()
                        else:
                            frame_2.destroy()
                            self.log += "Incorrect input @[" + str(time.perf_counter()) + "]"
                            print("Log: Incorrect input @[" + str(time.perf_counter()) + "]")
                            self.gui_init()

                gather_data()

                pyC45reimplementation.classify(plant, self.tree)  # classify the given data

                self.graph = graphviz.Digraph(comment="Decision Tree Model")
                self.translate_to_DOT(self.tree, decision=plant)  # generate a DOT with a path

                self.render_image()

                self.canvas.delete(self.canvas_image)  # remove old image from canvas

                self.canvas_image = PhotoImage(file="Digraph.gv.png")
                self.canvas.create_image(self.canvas_image.width() / 3, self.canvas_image.height() / 1.5, ancho=CENTER,
                                         image=self.canvas_image, tags="image")
                if self.arrows not in self.canvas.children:
                    show_move()
                self.canvas.pack(fill=BOTH, expand=True)

            calculate_button = Button(frame_2, text="Calculate", command=calculate)
            calculate_button.grid(row=0, column=1)

        def clicked_existent():
            """
            Shows menu for the selection of a .tree file
            :return:
            """
            try:
                frame_1.destroy()  # Destroy the asking file frame
                filename = askopenfilename()

                tree: Node = pyC45reimplementation.load_model(filename)  # load trained model
                self.tree = tree  # save reference to the tree
                self.graph = graphviz.Digraph(comment="Decision Tree Model")
                self.translate_to_DOT(self.tree)
                self.render_image()
                self.canvas.delete(self.canvas_image)  # remove old image from canvas
                self.canvas_image = PhotoImage(file="Digraph.gv.png")  # Save the image reference
                self.canvas.create_image(self.canvas_image.width()/3, self.canvas_image.height()/1.5, ancho=CENTER,
                                         image=self.canvas_image, tags="image")
                if self.arrows is None:
                    show_move()
                self.canvas.pack(fill=BOTH, expand=True)
                self.window.bind('<KeyPress>', move)
                print(self.graph.source)
                pyC45reimplementation.print_tree(self.tree)  # Print to console the tree (for debug)

                get_data_input()
            except FileNotFoundError:
                answer = messagebox.askretrycancel("Question", "File not specified, do you want to try that again?")
                if answer:
                    clicked_existent()
                else:
                    self.log += "File not selected @[" + str(time.perf_counter()) + "]"
                    print("Log: File not selected @[" + str(time.perf_counter()) + "]")
                    self.gui_init()
            except ValueError or IndexError:
                answer = messagebox.askretrycancel("Question", "Wrong file format, do you want to try again?")
                if answer:
                    clicked_train()
                else:
                    self.log += "File not selected @[" + str(time.perf_counter()) + "]"
                    print("Log: File not selected @[" + str(time.perf_counter()) + "]")
                    self.gui_init()

        def clicked_train():
            """
            Shows menu for the selection of a .csv file containing a dataset
            :return:
            """
            try:
                frame_1.destroy()  # Destroy the asking file frame
                filename = askopenfilename()
                tree: Node = pyC45reimplementation.train(pd.read_csv(filename))  # train a new model for the dataset
                self.tree = tree
                self.graph = graphviz.Digraph(comment="Decision Tree Model")
                self.translate_to_DOT(self.tree)
                pyC45reimplementation.print_tree(self.tree)  # Print to console the tree (for debug)
                self.render_image()
                self.canvas.delete(self.canvas_image)  # remove old image from canvas
                self.canvas_image = PhotoImage(file="Digraph.gv.png")  # Save the image reference
                self.canvas.create_image(self.canvas_image.width() / 3, self.canvas_image.height() / 1.5, ancho=CENTER,
                                         image=self.canvas_image, tags="image")
                if self.arrows not in self.canvas.children:
                    show_move()
                self.canvas.pack(fill=BOTH, expand=True)

                self.window.bind('<KeyPress>', move)  # allow the canvas to move with the arrow keys
                # print(self.graph.source)  # print the DOT notation for the tree
                get_data_input()  # init interface for the classifying
            except FileNotFoundError:
                # in the case a file was not specified
                answer = messagebox.askretrycancel("Question", "Do you want to try that again?")
                if answer:
                    clicked_train()
                else:
                    self.log += "File not selected @[" + str(time.perf_counter()) + "]"
                    print("Log: File not selected @[" + str(time.perf_counter()) + "]")
                    self.gui_init()
            except ValueError:
                # in the case a wrongly formatted file was specified
                answer = messagebox.askretrycancel("Question", "Wrong file format, do you want to try again?")
                if answer:
                    clicked_train()
                else:
                    self.log += "File not selected @[" + str(time.perf_counter()) + "]"
                    print("Log: File not selected @[" + str(time.perf_counter()) + "]")
                    self.gui_init()

        existent_button = Button(frame_1, text="Existent Model", font=("Times New Roman", 12), command=clicked_existent)
        existent_button.pack()
        train_button = Button(frame_1, text="New Model", font=("Times New Roman", 12), command=clicked_train)
        train_button.pack()
        self.window.mainloop()

    def translate_to_DOT(self, tree: Node, tag: str = "a", decision: Decision = None):
        """
        Translate a Node to DOT notation
        :param tree: Root node of the tree
        :param tag: Name of the node (helps keeping track in DOT)
        :param decision: Decision path
        :return:
        """
        if hasattr(tree, "tag"):
            # Check if the node has a name
            # This helps with compatibility with older versions of the program
            node_name: str = tree.tag
        else:
            tree.tag = tag
            node_name: str = tag
        node_name_right: str = node_name + "r"
        node_name_left: str = node_name + "l"
        if decision is None:
            # Check if a Decision has been made, helps with the drawing of the tree with a decision path
            if tree.results is not None:
                # Check if the node is a leaf
                label: str = ""
                label += tree.results
                """
                    Change the color and shape of the node depending on its contents
                    "yes" : cyan diamond
                    "no" : red square
                """
                self.graph.node(node_name, label, shape=("square" if label.__contains__("no") else "diamond"),
                                color=("crimson" if label.__contains__("no") else "dodgerblue2"),
                                style="filled", fillcolor=("red" if label.__contains__("no") else "cyan"))
            else:
                this_info: str = tree.attribute + " >= " + str(tree.value) + "?"  # info to be displayed in the node
                self.graph.node(node_name, this_info, shape="box",
                                color="blueviolet", style="filled", fillcolor="gray")  # current node
                self.translate_to_DOT(tree.left_child, tag=tag + "l")  # left child / false path
                self.graph.edge(node_name, node_name_left, label="False")  # link left node to its parent
                self.translate_to_DOT(tree.right_child, tag=tag + "r")  # right child / true path
                self.graph.edge(node_name, node_name_right, label="True")  # link node to its parent
        else:
            if tree.results is not None:
                # Check if the node is a leaf
                label: str = ""
                label += tree.results
                """
                    Change the color and shape of the node depending on its contents
                    "yes" and not in Decision path : cyan diamond
                    "yes" or "no" and in Decision path : green star
                    "no" and not in Decision path : red square
                """
                self.graph.node(node_name, label, shape=("star" if node_name in decision.path else
                                                         "square"if label.__contains__("no") else "diamond"),
                                color=("darkgreen" if node_name in decision.path else
                                       "crimson" if label.__contains__("no") else "dodgerblue2"),
                                style="filled", fillcolor=("green" if node_name in decision.path else
                                                           "red" if label.__contains__("no") else "cyan"))
            else:
                this_info: str = tree.attribute + " >= " + str(tree.value) + "?"  # info to be displayed in the node
                self.graph.node(node_name, this_info, shape="box", color="blueviolet", style="filled",
                                fillcolor=("green" if node_name in decision.path else "gray"))  # current node
                self.translate_to_DOT(tree.left_child, decision=decision, tag=tag + "l")  # left child / false path
                self.graph.edge(node_name, node_name_left, label="False")  # link left node to its parent
                self.translate_to_DOT(tree.right_child, decision=decision, tag=tag + "r")  # right child / true path
                self.graph.edge(node_name, node_name_right, label="True")  # link node to its parent

    def render_image(self):
        """
        Use graphviz to generate translate DOT to png
        :return:
        """
        self.graph.format = "png"
        self.graph.render()


if __name__ == '__main__':
    gui = GUI(800, 600)
    gui.window.minsize(800, 600)
    gui.gui_init()
