from ctypes import windll
from tkinter import *
from tkinter.filedialog import askopenfilename
import time
import pandas as pd
import turtle
import pyC45reimplementation
import Node
from tkinter import messagebox


class GUI(object):

    log: str = ''

    def __init__(self, width, height):
        windll.shcore.SetProcessDpiAwareness(1)
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
                self.drawtree(root)
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
                self.drawtree(tree)
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

        base_a = (0, 0)
        base_b = (30, 20)
        self.draw_line(base_a[0], base_b[0], base_a[1], base_b[1])
        self.window.mainloop()

    def draw_line(self, x1, y1, x2, y2):
        self.canvas.create_line(x1, y1, x2, y2, fill="black")
        self.canvas.pack()

    def draw_tree(self, root: Node):
        return 0

    def drawtree(self, root):
        def height(root):
            return 1 + max(height(root.left_child), height(root.right_child)) if root else -1

        def jumpto(x, y):
            t.penup()
            t.goto(x, y)
            t.pendown()

        def draw(node, x, y, dx):
            if node:
                t.goto(x, y)
                jumpto(x, y - 20)
                t.write(node.value, align='center', font=('Arial', 12, 'normal'))
                draw(node.left_child, x - dx, y - 60, dx / 2)
                jumpto(x, y - 20)
                draw(node.right_child, x + dx, y - 60, dx / 2)

        t = turtle.Turtle()
        t.speed(0)
        turtle.delay(0)
        h = height(root)
        jumpto(0, 30 * h)
        draw(root, 0, 30 * h, 40 * h)
        t.hideturtle()
        turtle.mainloop()


if __name__ == '__main__':
    gui = GUI(800, 600)
    gui.gui_init()

