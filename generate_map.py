import random
import numpy as np
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define environment constants
FREE: int = 0
OBSTACLE_SOFT: int = 1
OBSTACLE_HARD: int = 2
AGENT: int = 3
TARGET: int = 4

def generate_map(rowSize: int, colSize: int, num_soft_obstacles: int, num_hard_obstacles: int, num_robots: int, tetherDist: int, num_leaders: int = 1, num_target: int = 1):
    # Initialize the map with free cells
    grid = np.zeros((rowSize, colSize), dtype=int)

    total_cells = rowSize * colSize
    total_obstacles = num_soft_obstacles + num_hard_obstacles
    if (total_obstacles + num_robots + num_target) > total_cells:
        raise ValueError("Total number of obstacles, robots, and target exceeds the grid size.")

    # Place soft obstacles randomly
    for _ in range(num_soft_obstacles):
        while True:
            x, y = random.randint(0, rowSize-1), random.randint(0, colSize-1)
            if grid[x, y] == FREE:
                grid[x, y] = OBSTACLE_SOFT
                break

    # Place hard obstacles randomly
    for _ in range(num_hard_obstacles):
        while True:
            x, y = random.randint(0, rowSize-1), random.randint(0, colSize-1)
            if grid[x, y] == FREE:
                grid[x, y] = OBSTACLE_HARD
                break

    # Place robots randomly with roles
    robots = []
    roles = ['leader', 'follower']
    for i in range(num_robots):
        while True:
            x, y = random.randint(0, rowSize-1), random.randint(0, colSize-1)
            if grid[x, y] == FREE:
                if i == 0 or all(max(abs(x - robot['position'][0]), abs(y - robot['position'][1])) <= tetherDist for robot in robots):
                    grid[x, y] = AGENT
                    robots.append({
                        'position': (x, y), 
                        'role': roles[0 if i<num_leaders else 1]
                    })
                    break

    # Place the targets randomly
    targets = []
    for i in range(num_target):
        while True:
            x, y = random.randint(0, rowSize-1), random.randint(0, colSize-1)
            if grid[x, y] == FREE:
                grid[x, y] = TARGET
                targets.append((x, y))
                break

    return grid, robots, targets

def print_map(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))

def generate_sample_data(number: int):
    os.makedirs("sample_data", exist_ok=True)
    
    for currSampleId in range(number):
        size = random.randint(5, 10)
        num_soft_obstacles = random.randint(5, 10)
        num_hard_obstacles = random.randint(1, 5)
        num_robots = random.randint(1, 3)
        grid, robots, target = generate_map(
            rowSize=size, 
            colSize=size, 
            num_soft_obstacles=num_soft_obstacles, 
            num_hard_obstacles=num_hard_obstacles, 
            num_robots=num_robots,
            tetherDist=2
        )
        try:
            np.save(f"sample_data/sample_grid_{currSampleId+1}.npy", grid)
            joblib.dump(robots, f"sample_data/sample_robots_{currSampleId+1}.pkl")
            joblib.dump(target, f"sample_data/sample_target_{currSampleId+1}.pkl")
        except:
            print("Error in saving sample data")
            return False
    return True

if __name__ == "__main__":
    size = 10  # Size of the map
    num_soft_obstacles = 10  # Number of soft obstacles
    num_hard_obstacles = 5  # Number of hard obstacles
    num_robots = 2  # Number of robots
    tetherDist = 2  # Tether distance

    grid, robots, target = generate_map(
        rowSize=size, 
        colSize=size, 
        num_soft_obstacles=num_soft_obstacles, 
        num_hard_obstacles=num_hard_obstacles, 
        num_robots=num_robots,
        tetherDist=tetherDist
    )
    print("Generated Map:")
    print_map(grid)
    print("\nRobots:", robots)
    print("Target:", target)

    generate_sample_data(20) # Generate 20 sample data