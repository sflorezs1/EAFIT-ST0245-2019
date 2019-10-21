from ctypes import windll
from tkinter import *
from tkinter.filedialog import askopenfilename
import time
import pandas as pd
import pyC45reimplementation
import Node
from tkinter import messagebox
import graphviz
import os
from PIL import Image, ImageTk
from rsvg import *
import cairo

os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'


class GUI(object):

    log: str = ''

    def __init__(self, width, height):
        windll.shcore.SetProcessDpiAwareness(1)
        self.graph = graphviz.Digraph(comment="Decision Tree Model")
        self.window = Tk()
        self.width = width
        self.height = height
        self.canvas = Canvas(self.window, width=self.width, height=self.height, bg="white")
        self.canvas.pack()
        self.tree = None

    def gui_init(self):
        self.window.title('Rust detecting Decision Tree Program')
        frame_1 = Frame(self.window)
        frame_1.pack()
        welcome_text: str = "Welcome to the program for testing if a coffee plant has rust.\n"
        welcome = Label(frame_1, text=welcome_text, font=("Times New Roman", 14))
        welcome.pack()
        decision_text: str = "For starters, do you want to train a new model (.csv file)?" \
                             " or use a existent one (.tree file)?"
        decision = Label(frame_1, text=decision_text, font=("Times New Roman", 13))
        decision.pack()

        def clicked_existent():
            try:
                frame_1.destroy()
                filename = askopenfilename()

                root: Node = pyC45reimplementation.load_model(filename)
                self.tree = root
                self.translate_to_DOT(root)
                self.render_image()
                print(self.graph.source)
                pyC45reimplementation.print_tree(root)
                pyC45reimplementation.print_tree(root)
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
            try:
                frame_1.destroy()
                filename = askopenfilename()
                tree: Node = pyC45reimplementation.train(pd.read_csv(filename))
                self.tree = tree
                self.translate_to_DOT(tree)
                pyC45reimplementation.print_tree(tree)
            except FileNotFoundError:
                answer = messagebox.askretrycancel("Question", "Do you want to try that again?")
                if answer:
                    clicked_train()
                else:
                    self.log += "File not selected @[" + str(time.perf_counter()) + "]"
                    print("Log: File not selected @[" + str(time.perf_counter()) + "]")
                    self.gui_init()
            except ValueError:
                answer = messagebox.askretrycancel("Question", "Wrong file format, do you want to try again?")
                if answer:
                    clicked_train()
                else:
                    self.log += "File not selected @[" + str(time.perf_counter()) + "]"
                    print("Log: File not selected @[" + str(time.perf_counter()) + "]")
                    self.gui_init()

        existent_button = Button(frame_1, text="existent model", command=clicked_existent)
        existent_button.pack()
        train_button = Button(frame_1, text="train new model", command=clicked_train)
        train_button.pack()
        self.window.mainloop()

    def translate_to_DOT(self, tree: Node, node_name: str = "a"):
        if tree.results is not None:
            label: str = ""
            label += tree.results
            self.graph.node(node_name, label, shape=("square" if label.__contains__("no") else "diamond"),
                            color=("crimson" if label.__contains__("no") else "dodgerblue2"),
                            style="filled", fillcolor=("red" if label.__contains__("no") else "cyan"))
        else:
            this_info: str = tree.attribute + " : " + str(tree.value) + "?"
            self.graph.node(node_name, this_info, shape="box", color="blueviolet", style="filled", fillcolor="gray")
            node_name_left: str = node_name + "l"
            self.translate_to_DOT(tree.left_child, node_name_left)
            self.graph.edge(node_name, node_name_left, label="True")
            node_name_right: str = node_name + "r"
            self.translate_to_DOT(tree.right_child, node_name_right)
            self.graph.edge(node_name, node_name_right, label="False")

    def render_image(self):
        self.graph.format = "svg"
        self.graph.render()

    def svgPhotoImage(self, file_path_name):
        """Returns a ImageTk.PhotoImage object represeting the svg file"""
        # Based on pygame.org/wiki/CairoPygame and http://bit.ly/1hnpYZY
        svg = cairosvg.
        width, height = svg.get_dimension_data()[:2]
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height))
        context = cairo.Context(surface)
        # context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)
        svg.render_cairo(context)
        tk_image = ImageTk.PhotoImage('RGBA')
        image = Image.frombuffer('RGBA', (width, height), surface.get_data(), 'raw', 'BGRA', 0, 1)
        tk_image.paste(image)
        return tk_image


if __name__ == '__main__':
    gui = GUI(800, 600)
    gui.gui_init()
