import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


# Step 1: Clean the data and separate by side
def process_csv(file_path):
    data = pd.read_csv(file_path).dropna()
    left_cones = data[data["side"] == "left"]
    right_cones = data[data["side"] == "right"]
    return left_cones, right_cones


# Step 2: Generate a smooth path for each side
def generate_smooth_path(x, y):
    tck, u = splprep([x, y], s=0)
    u_fine = np.linspace(0, 1, 500)
    x_fine, y_fine = splev(u_fine, tck)
    return x_fine, y_fine


# Step 3: Calculate the middle path
def calculate_middle_path(left_x, left_y, right_x, right_y):
    mid_x = (left_x + right_x) / 2
    mid_y = (left_y + right_y) / 2
    return mid_x, mid_y


# Step 4: Visualize the cones and the calculated path
def visualize_path(left_x, left_y, right_x, right_y, mid_x, mid_y):
    plt.figure(figsize=(12, 8))
    plt.scatter(left_x, left_y, color="blue", label="Left Cones")
    plt.scatter(right_x, right_y, color="green", label="Right Cones")
    plt.plot(mid_x, mid_y, color="red", label="Middle Path")
    plt.title("Path Planning with Side Consideration")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()


# Main Execution
left_cones, right_cones = process_csv("./BrandsHatchLayout.csv")

# Generate smooth paths for left and right cones
left_x_smooth, left_y_smooth = generate_smooth_path(
    left_cones["x"].values, left_cones["y"].values
)
right_x_smooth, right_y_smooth = generate_smooth_path(
    right_cones["x"].values, right_cones["y"].values
)

# Calculate the middle path
mid_x, mid_y = calculate_middle_path(
    left_x_smooth, left_y_smooth, right_x_smooth, right_y_smooth
)

# Visualize the results
visualize_path(
    left_cones["x"], left_cones["y"], right_cones["x"], right_cones["y"], mid_x, mid_y
)
