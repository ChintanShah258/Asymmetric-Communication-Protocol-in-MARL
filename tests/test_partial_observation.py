import unittest
import numpy as np
import os
import sys
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT)
from partial_observation import get_partial_observation
from generate_map import generate_map

class TestPartialObservation(unittest.TestCase):
    def setUp(self):
        self.grid, self.agents, self.targets = generate_map(
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )

    def test_partial_observation_center(self):
        agent_position = (5, 5)
        observation_radius = 2
        partial_obs = get_partial_observation(self.grid, agent_position, observation_radius)

        # Verify the shape of the partial observation
        self.assertEqual(partial_obs.shape, (2 * observation_radius + 1, 2 * observation_radius + 1))

    def test_partial_observation_edge(self):
        # Agent near the edge of the grid
        agent_position = (1, 1)
        observation_radius = 2
        partial_obs = get_partial_observation(self.grid, agent_position, observation_radius)

        # Verify the shape of the partial observation
        self.assertEqual(partial_obs.shape, (2 * observation_radius + 1, 2 * observation_radius + 1))

        # Verify that out-of-bounds areas are padded with -1
        self.assertTrue(np.all(partial_obs[0, :] == -1))  # Top padding
        self.assertTrue(np.all(partial_obs[:, 0] == -1))  # Left padding

    def test_partial_observation_corner(self):
        # Agent in the top-left corner of the grid
        agent_position = (0, 0)
        observation_radius = 2
        partial_obs = get_partial_observation(self.grid, agent_position, observation_radius)

        # Verify the shape of the partial observation
        self.assertEqual(partial_obs.shape, (2 * observation_radius + 1, 2 * observation_radius + 1))

        # Verify that out-of-bounds areas are padded with -1
        self.assertTrue(np.all(partial_obs[0:2, :] == -1))  # Top padding
        self.assertTrue(np.all(partial_obs[:, 0:2] == -1))  # Left padding

    def test_partial_observation_out_of_bounds(self):
        # Agent out of bounds (negative position)
        agent_position = (-1, -1)
        observation_radius = 2
        partial_obs = get_partial_observation(self.grid, agent_position, observation_radius)
        # Verify the shape of the partial observation
        self.assertEqual(partial_obs.shape, (2 * observation_radius + 1, 2 * observation_radius + 1))

if __name__ == '__main__':
    unittest.main()
