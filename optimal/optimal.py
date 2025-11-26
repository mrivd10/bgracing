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
    curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2) ** 1.5
    curvature[~np.isfinite(curvature)] = 0
    return curvature

def find_fastest_racing_line(
    center_x,
    center_y,
    curvature,
    left_x,
    left_y,
    right_x,
    right_y,
    track_width,
    max_velocity,
    max_lateral_acceleration,
):
    """
    Use dynamic programming to find the fastest racing line.

    Parameters:
    - center_x, center_y: Coordinates of the track centerline.
    - left_x, left_y: Coordinates of the left cones.
    - right_x, right_y: Coordinates of the right cones.
    - curvature: Curvature values of the centerline.
    - track_width: Discretized width of the track.
    - max_velocity: Maximum velocity of the vehicle (m/s).
    - max_lateral_acceleration: Maximum lateral acceleration (m/s²).
    
    Returns:
    - optimal_path: Optimal racing line as a list of indices.
    """
    # Notes:
    # - track_width represents the discretization resolution across the lane; a higher
    #   value yields finer lateral steps at the cost of computation time.
    # - offset_fraction converts a discrete step into a symmetric lateral fraction, so
    #   j = track_width // 2 follows the centerline while the extremes trace the edges.
    # - allowable_velocity uses a curvature floor to avoid division-by-zero and ensures
    #   the dynamic-programming cost stays physically meaningful.
    track_length = len(center_x)
    dp_cost = np.full((track_length, track_width), np.inf)
    dp_path = np.zeros((track_length, track_width), dtype=int)

    def offset_fraction(index: int) -> float:
        return (index - track_width // 2) / track_width

    dp_cost[-1, :] = 0

    for i in range(track_length - 2, -1, -1):
        for j in range(track_width):
            current_offset = offset_fraction(j)
            current_x = center_x[i] + current_offset * (right_x[i] - left_x[i])
            current_y = center_y[i] + current_offset * (right_y[i] - left_y[i])

            curvature_safe = max(curvature[i], 1e-6)
            allowable_velocity = min(
                max_velocity,
                np.sqrt(max_lateral_acceleration / curvature_safe),
            )

            for k in range(track_width):
                next_offset = offset_fraction(k)
                next_x = center_x[i + 1] + next_offset * (right_x[i + 1] - left_x[i + 1])
                next_y = center_y[i + 1] + next_offset * (right_y[i + 1] - left_y[i + 1])

                distance = np.hypot(next_x - current_x, next_y - current_y)
                time = distance / allowable_velocity

                cost = dp_cost[i + 1, k] + time
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
    
    def offset_fraction(index: int) -> float:
        return (index - track_width // 2) / track_width

    racing_line_x = [
        center_x[i] + offset_fraction(path) * (right_x[i] - left_x[i])
        for i, path in enumerate(optimal_path)
    ]
    racing_line_y = [
        center_y[i] + offset_fraction(path) * (right_y[i] - left_y[i])
        for i, path in enumerate(optimal_path)
    ]
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
optimal_path = find_fastest_racing_line(
    center_x,
    center_y,
    curvature,
    left_x,
    left_y,
    right_x,
    right_y,
    track_width,
    max_velocity,
    max_lateral_acceleration,
)

visualize_racing_line(center_x, center_y, left_x, left_y, right_x, right_y, optimal_path, track_width)
