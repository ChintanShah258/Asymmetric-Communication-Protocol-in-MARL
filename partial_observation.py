import numpy as np

def get_partial_observation(grid: np.ndarray, agent_position: tuple[int, int], observation_radius: int) -> np.ndarray:
    """
    Get a partial observation of the grid for an agent.

    Parameters:
    - grid (np.ndarray): The full grid environment.
    - agent_position (tuple[int, int]): The (x, y) position of the agent on the grid.
    - observation_radius (int): The radius of the agent's observation.

    Returns:
    - np.ndarray: A partial observation of the grid centered around the agent.
    """
    x, y = agent_position
    grid_size_x, grid_size_y = grid.shape

    # Define the bounds of the partial observation
    x_min = max(0, x - observation_radius)
    x_max = min(grid_size_x, x + observation_radius + 1)
    y_min = max(0, y - observation_radius)
    y_max = min(grid_size_y, y + observation_radius + 1)

    # Extract the partial observation
    partial_observation = grid[x_min:x_max, y_min:y_max]

    # Pad the observation if the agent is near the edge of the grid
    pad_top = max(0, observation_radius - x)
    pad_bottom = max(0, (x + observation_radius + 1) - grid_size_x)
    pad_left = max(0, observation_radius - y)
    pad_right = max(0, (y + observation_radius + 1) - grid_size_y)

    partial_observation = np.pad(
        partial_observation,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=-1  # Use -1 to represent out-of-bounds areas
    )

    # Replace the agent's position with 0 in the partial observation
    # Denote a possible place to move - i.e., stay 
    agent_local_x = observation_radius
    agent_local_y = observation_radius
    partial_observation[agent_local_x, agent_local_y] = 0

    return partial_observation

# Example Usage
if __name__ == "__main__":
    # Example grid (10x10)
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    # Agent's position and observation radius
    agent_position = (3, 3)
    observation_radius = 2

    # Get partial observation
    partial_obs = get_partial_observation(grid, agent_position, observation_radius)

    print("Full Grid:")
    print(grid)
    print("\nAgent's Position:", agent_position)
    print("Partial Observation:")
    print(partial_obs)