import string
import random
import matplotlib.pyplot as plt

def generate_column_labels(n):
    labels = []
    for i in range(1, n + 1):
        label = ""
        while i > 0:
            i -= 1
            label = chr(65 + (i % 26)) + label
            i //= 26
        labels.append(label)
    return labels

def plot_environment(row, col):
    """
    Plot the environment with the given number of rows and columns,
    and save the plot to a .png image file.

    Parameters:
    row (int): The number of rows in the grid.
    col (int): The number of columns in the grid.
    """
    # Define grid dimensions
    grid_size = (row, col)

    # Initialize the grid with empty cells
    display_grid = [[" " for _ in range(grid_size[1])] for _ in range(grid_size[0])]

    # Generate unique random positions for all elements
    total_positions = row * col
    num_obstacles = 12
    all_positions = random.sample(range(total_positions), 2 + 1 + num_obstacles)
    
    r1_position = divmod(all_positions[0], col)
    r2_position = divmod(all_positions[1], col)
    target_position = divmod(all_positions[2], col)
    obstacle_positions = [divmod(pos, col) for pos in all_positions[3:]]

    # Place elements on the grid
    display_grid[r1_position[0]][r1_position[1]] = "R1"
    display_grid[r2_position[0]][r2_position[1]] = "R2"
    display_grid[target_position[0]][target_position[1]] = "T"

    for obs in obstacle_positions:
        display_grid[obs[0]][obs[1]] = "*"

    # Plotting the environment
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_aspect('equal')

    # Draw gridlines with borders around all cells
    for x in range(grid_size[1] + 1):
        ax.axvline(x, color='black', lw=1)  # Thicker lines for better visibility
    for y in range(grid_size[0] + 1):
        ax.axhline(y, color='black', lw=1)

    # Draw the grid elements
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            ax.text(j + 0.5, grid_size[0] - i - 0.5, display_grid[i][j], va='center', ha='center', fontsize=12)

    # Generate column labels dynamically
    columns = generate_column_labels(grid_size[1])
    rows = [str(i + 1) for i in range(grid_size[0])]

    # Draw column labels at the topmost part of the grid
    for idx, label in enumerate(columns[:grid_size[1]]):
        ax.text(idx + 0.5, -0.7, label, va='center', ha='center', fontsize=12, fontweight='bold')

    # Draw row labels
    for idx, label in enumerate(rows[:grid_size[0]]):
        ax.text(-0.7, idx + 0.5, label, va='center', ha='center', fontsize=12, fontweight='bold')

    # Configure plot appearance
    plt.gca().invert_yaxis()
    plt.axis('off')

    # Save the plot to a file
    plt.savefig('output.png', bbox_inches='tight')
    plt.show()

# Example usage
plot_environment(100, 100)