import random
import numpy as np

def generate_map(size, num_obstacles, num_robots):
    # Initialize the map with free cells
    grid = np.zeros((size, size), dtype=int)

    # Place obstacles randomly
    for _ in range(num_obstacles):
        while True:
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if grid[x, y] == 0:
                grid[x, y] = 1
                break

    # Place robots randomly with roles
    robots = []
    roles = ['leader', 'follower']
    for i in range(num_robots):
        while True:
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if grid[x, y] == 0:
                grid[x, y] = 2
                robots.append({'position': (x, y), 'role': roles[i % len(roles)]})
                break

    # Place the target randomly
    while True:
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        if grid[x, y] == 0:
            grid[x, y] = 3
            target = (x, y)
            break

    return grid, robots, target

def print_map(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))

if __name__ == "__main__":
    size = 10  # Size of the map
    num_obstacles = 15  # Number of obstacles
    num_robots = 2  # Number of robots

    grid, robots, target = generate_map(size, num_obstacles, num_robots)
    print("Generated Map:")
    print_map(grid)
    print("\nRobots:", robots)
    print("Target:", target)