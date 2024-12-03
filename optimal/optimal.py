import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'BrandsHatchLayout.csv'
data = pd.read_csv(file_path).dropna()

left_cones = data[data['side'] == 'left']
right_cones = data[data['side'] == 'right']

def generate_centerline(left_x, left_y, right_x, right_y):
    """
    Generate the centerline of the track by averaging the positions of left and right cones.
    
    Parameters:
    - left_x, left_y: Coordinates of the left cones.
    - right_x, right_y: Coordinates of the right cones.
    
    Returns:
    - center_x, center_y: Coordinates of the centerline.
    """
    center_x = (left_x + right_x) / 2
    center_y = (left_y + right_y) / 2
    return center_x, center_y

def calculate_curvature(x, y):
    """
    Calculate the curvature of the track centerline using the formula:
    k = |x''y' - y''x'| / (x'^2 + y'^2)^(3/2)
    
    Parameters:
    - x, y: Coordinates of the centerline.
    
    Returns:
    - curvature: Array of curvature values along the centerline.
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
    curvature[np.isnan(curvature)] = 0
    return curvature

def find_fastest_racing_line(center_x, center_y, curvature, track_width, max_velocity, max_lateral_acceleration):
    """
    Use dynamic programming to find the fastest racing line.
    
    Parameters:
    - center_x, center_y: Coordinates of the track centerline.
    - curvature: Curvature values of the centerline.
    - track_width: Discretized width of the track.
    - max_velocity: Maximum velocity of the vehicle (m/s).
    - max_lateral_acceleration: Maximum lateral acceleration (m/s²).
    
    Returns:
    - optimal_path: Optimal racing line as a list of indices.
    """
    track_length = len(center_x)
    dp_cost = np.full((track_length, track_width), np.inf)
    dp_path = np.zeros((track_length, track_width), dtype=int)  
    
    dp_cost[-1, :] = 0

    for i in range(track_length - 2, -1, -1):
        for j in range(track_width):
            for k in range(track_width):
                distance = np.sqrt((center_x[i+1] - center_x[i])**2 + ((j - k) / track_width)**2)
                velocity = min(max_velocity, np.sqrt(max_lateral_acceleration / curvature[i]) if curvature[i] > 0 else max_velocity)
                time = distance / velocity
                
                cost = dp_cost[i+1, k] + time
                if cost < dp_cost[i, j]:
                    dp_cost[i, j] = cost
                    dp_path[i, j] = k

    optimal_path = []
    current = np.argmin(dp_cost[0, :]) 
    for i in range(track_length):
        optimal_path.append(current)
        current = dp_path[i, current]
    
    return optimal_path

def visualize_racing_line(center_x, center_y, left_x, left_y, right_x, right_y, optimal_path, track_width):
    """
    Visualize the cones, centerline, and fastest racing line.
    
    Parameters:
    - center_x, center_y: Coordinates of the track centerline.
    - left_x, left_y: Coordinates of the left cones.
    - right_x, right_y: Coordinates of the right cones.
    - optimal_path: Optimal racing line as a list of indices.
    - track_width: Discretized width of the track.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(left_x, left_y, label="Left Cones", color="blue")
    plt.plot(right_x, right_y, label="Right Cones", color="green")
    plt.plot(center_x, center_y, label="Centerline", color="gray", linestyle="--")
    
    racing_line_x = center_x
    racing_line_y = [center_y[i] + (path - track_width // 2) * (right_y[i] - left_y[i]) / track_width for i, path in enumerate(optimal_path)]
    plt.plot(racing_line_x, racing_line_y, label="Fastest Racing Line", color="red")
    
    plt.title("Fastest Racing Line - Dynamic Programming")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()

# Main
left_x, left_y = left_cones['x'].values, left_cones['y'].values
right_x, right_y = right_cones['x'].values, right_cones['y'].values

center_x, center_y = generate_centerline(left_x, left_y, right_x, right_y)

curvature = calculate_curvature(center_x, center_y)

# settings:
track_width = 10  # resolution
max_velocity = 50  # Max Speed(m/s)
max_lateral_acceleration = 10  # Maxi acceleration(m/s²)
optimal_path = find_fastest_racing_line(center_x, center_y, curvature, track_width, max_velocity, max_lateral_acceleration)

visualize_racing_line(center_x, center_y, left_x, left_y, right_x, right_y, optimal_path, track_width)
