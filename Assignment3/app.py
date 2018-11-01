import tkinter as tk
import random
import time
import math

class Application:
    def __init__(self, start_state, size=(800, 800), center=True):
        self.start_state = start_state.board
        self.width, self.height = size
        self.center = center

        self.root = tk.Tk()
        self.root.title('Application')
        self.resize()

        self.create_widget()
        self.canvasman.redraw()
        self.root.update_idletasks()
        self.root.update()

    def resize(self):
        if self.center:
            x_pos = int(self.root.winfo_screenwidth() / 2  - self.width / 2)
            y_pos = int(self.root.winfo_screenheight() / 2 - self.height / 2)

            self.root.geometry(f'{self.width}x{self.height}+{x_pos}+{y_pos}')
        else:
            self.root.geometry(f'{self.width}x{self.height}')

    def create_widget(self):
        self.canvasman = CanvasManager(root=self.root, start_state=self.start_state)

    def update(self, state):
        self.canvasman.update(state.board)
        self.canvasman.redraw()
        self.root.update_idletasks()
        self.root.update()

class CanvasManager:
    def __init__(self, root, start_state):
        self.start_state = start_state
        self.root = root
        self.size = (800, 800)
        self.padding = 100
        self.n = len(self.start_state)
        self.circle_radius = (self.size[0] - 2 * self.padding) / (4 * self.n)
        self.canvas = tk.Canvas(master=self.root, bg='#DDDDDD')
        self.circle_centers = [[None for i in range(self.n)] for j in range(self.n)]
        self.calculate_circle_centers()
        self.circles = [[None for i in range(self.n)] for j in range(self.n)]
        self.create_circles()
        self.arcs = {}
        self.create_arcs()
        self.create_labels()
        self.color = 'red'
        self.selected = []

    def create_labels(self):
        window_width, window_height = self.size
        font = ('Helvetica', 20)
        player_one_name = 'Bot 1'
        player_two_name = 'Bot 2'
        p1_pos = (self.padding * 2, self.padding)
        p2_pos = (window_width - self.padding * 2, self.padding)
        label_one = self.canvas.create_text(p1_pos, text=player_one_name, font=font)
        label_two = self.canvas.create_text(p2_pos, text=player_two_name, font=font)
        
        r = 25  # Color label radius
        offset = 50  # Color label offset from text label
        p1_start = (p1_pos[0] - r, p1_pos[1] - r + offset)
        p1_end = (p1_pos[0] + r, p1_pos[1] + r + offset)
        p2_start = (p2_pos[0] - r, p2_pos[1] - r + offset)
        p2_end = (p2_pos[0] + r, p2_pos[1] + r + offset)

        p1_options = {
            'fill': 'red',
            'outline': '',
        }
        p2_options = {
            'fill': 'black',
            'outline': '',
        }

        color_label_one = self.canvas.create_oval(*p1_start, *p1_end, **p1_options)
        color_label_two = self.canvas.create_oval(*p2_start, *p2_end, **p2_options)


    def calculate_circle_centers(self):
        window_width, window_height = self.size

        first_center = (window_width / 2, self.padding)
        second_center = (window_width - self.padding, window_height / 2)
        third_center = (self.padding, window_height / 2)
        
        first_vector_x = (second_center[0] - first_center[0]) / (self.n - 1)
        first_vector_y = (second_center[1] - first_center[1]) / (self.n - 1)

        second_vector_x = (third_center[0] - first_center[0]) / (self.n - 1)
        second_vector_y = (third_center[1] - first_center[1]) / (self.n - 1)

        for i in range(self.n):
            for j in range(self.n):
                new_center_x = first_center[0] + first_vector_x * j + second_vector_x * i
                new_center_y = first_center[1] + first_vector_y * j + second_vector_y * i
                new_center = (new_center_x, new_center_y)
                self.circle_centers[i][j] = new_center

    def redraw(self):
        self.canvas.pack(expand=True, fill=tk.BOTH)

    def update(self, state):
        for i, row in enumerate(state):
            for j, cell in enumerate(row):
                circle = self.circles[i][j]
                if cell == 1:
                    color = 'red'
                elif cell == 2:
                    color = 'black'
                else:
                    color = 'white'
                self.canvas.itemconfig(circle.graphic, fill=color)

    def create_circles(self):
        for i, row in enumerate(self.circle_centers):
            for j, center in enumerate(row):
                self.circles[i][j] = Circle(canvas=self.canvas, center=self.circle_centers[i][j], radius=self.circle_radius)

    def create_arcs(self):
        directions = ((0, 1), (1, 0), (1, -1))
        for x in range(self.n):
            for y in range(self.n):
                for direction in directions:
                    new_x = x + direction[0]
                    new_y = y + direction[1]
                    if new_x < 0 or new_x >= self.n or new_y < 0 or new_y >= self.n:
                        continue
                    
                    start_center = self.circle_centers[x][y] 
                    end_center = self.circle_centers[new_x][new_y]

                    arc = Arc(canvas=self.canvas, start_center=start_center, end_center=end_center, circle_radius=self.circle_radius)
                    self.arcs[((x, y), (new_x, new_y))] = arc
                    self.arcs[((new_x, new_y), (x, y))] = arc 


class Circle:
    def __init__(self, canvas, cell=(0, 0), center=(0, 0), radius=25):
        self.canvas = canvas
        self.cell = cell
        self.center = center
        self.radius = radius
        self.graphic = None
        self.options = {
            'fill': 'white',
            'outline': '',
        }

        self.create()
        
    def create(self):
        center_x, center_y = self.center
        x0 = center_x - self.radius
        y0 = center_y - self.radius
        x1 = center_x + self.radius
        y1 = center_y + self.radius
        self.pos = (x0, y0, x1, y1)
        self.graphic = self.canvas.create_oval(*self.pos, **self.options)

class Arc:
    def __init__(self, canvas, start_center, end_center, circle_radius):
        self.canvas = canvas
        self.start_center = start_center
        self.end_center = end_center
        self.circle_radius = circle_radius
        self.graphic = None
        self.options = {
            'fill': 'black',
            'width': 3,
        }
        
        self.create()

    def create(self):
        x0, y0 = self.start_center
        x1, y1 = self.end_center
        
        vector = (x1 - x0, y1 - y0)
        vector_length = math.sqrt(vector[0]**2 + vector[1]**2)
        unit_vector = (vector[0] / vector_length, vector[1] / vector_length)

        start_x = x0 + self.circle_radius * unit_vector[0]
        start_y = y0 + self.circle_radius * unit_vector[1]

        end_x = x1 - self.circle_radius * unit_vector[0]
        end_y = y1 - self.circle_radius * unit_vector[1]

        self.pos = (start_x, start_y, end_x, end_y)

        self.graphic = self.canvas.create_line(*self.pos, **self.options)
    

# app = Application(center=True)
# app.start()
