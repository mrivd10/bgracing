import matplotlib.pyplot as plt
import numpy as np
import random

# Define the RRT algorithm
class Node:
    def init(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def init(self, start, goal, boundaries, max_iter=500, step_size=0.5):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.boundaries = boundaries
        self.max_iter = max_iter
        self.step_size = step_size
        self.tree = [self.start]

    

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)*2 + (node1.y - node2.y)*2)

    def get_random_point(self):
        x_min, x_max, y_min, y_max = self.boundaries
        return random.uniform(x_min, x_max), random.uniform(y_min, y_max)

    def nearest_node(self, random_point):
        return min(self.tree, key=lambda node: self.distance(node, Node(*random_point)))

    def steer(self, from_node, to_point):
        angle = np.arctan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        new_x = from_node.x + self.step_size * np.cos(angle)
        new_y = from_node.y + self.step_size * np.sin(angle)
        return Node(new_x, new_y)

    def is_valid(self, node):
        x_min, x_max, y_min, y_max = self.boundaries
        return x_min <= node.x <= x_max and y_min <= node.y <= y_max

    def build(self):
        for _ in range(self.max_iter):
            rand_point = self.get_random_point()
            nearest = self.nearest_node(rand_point)
            new_node = self.steer(nearest, rand_point)

            if self.is_valid(new_node):
                new_node.parent = nearest
                self.tree.append(new_node)

                if self.distance(new_node, self.goal) < self.step_size:
                    self.goal.parent = new_node
                    self.tree.append(self.goal)
                    break

    def get_path(self):
        path = []
        current = self.goal
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]

# Load cone positions
import pandas as pd

file_path = 'BrandsHatchLayout.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Separate left and right cones
left_cones = data[data['side'] == 'left']
right_cones = data[data['side'] == 'right']

# Define track boundaries (bounding box around cones)
x_min, x_max = min(data['x']), max(data['x'])
y_min, y_max = min(data['y']), max(data['y'])
boundaries = (x_min, x_max, y_min, y_max)

# RRT parameters
start = (np.mean(left_cones['x']), np.mean(left_cones['y']))  # Start near the left boundary
goal = (np.mean(right_cones['x']), np.mean(right_cones['y']))  # Goal near the right boundary

# Build the RRT and get the path
rrt = RRT.init(start, goal, boundaries, max_iter=1000, step_size=0.5)
rrt.build()
path = rrt.get_path()

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(left_cones['x'], left_cones['y'], c='red', label='Left Boundary')
plt.scatter(right_cones['x'], right_cones['y'], c='blue', label='Right Boundary')
plt.plot([p[0] for p in path], [p[1] for p in path], 'k-', linewidth=2, label='Optimal Racing Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('RRT-based Optimal Racing Line')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
