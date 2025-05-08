import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_map import generate_map

class TestGenerateMap(unittest.TestCase):
    size = 5
    num_soft_obstacles = 5
    num_hard_obstacles = 3
    num_robots = 2
    tetherDist = 2

    def test_generate_map_size(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=self.num_soft_obstacles, 
            num_hard_obstacles=self.num_hard_obstacles, 
            num_robots=self.num_robots,
            tetherDist=self.tetherDist
        )
        self.assertEqual(grid.shape, (self.size, self.size))

    def test_generate_map_obstacles(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=self.num_soft_obstacles, 
            num_hard_obstacles=self.num_hard_obstacles, 
            num_robots=self.num_robots,
            tetherDist=self.tetherDist
        )
        self.assertEqual(np.sum(grid == 1), self.num_soft_obstacles)
        self.assertEqual(np.sum(grid == 2), self.num_hard_obstacles)

    def test_generate_map_robots(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=self.num_soft_obstacles, 
            num_hard_obstacles=self.num_hard_obstacles, 
            num_robots=self.num_robots,
            tetherDist=self.tetherDist
        )
        self.assertEqual(len(robots), self.num_robots)
        for robot in robots:
            self.assertIn(robot['role'], ['leader', 'follower'])

    def test_generate_map_target(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=self.num_soft_obstacles, 
            num_hard_obstacles=self.num_hard_obstacles, 
            num_robots=self.num_robots,
            tetherDist=self.tetherDist
        )
        for t in targets:
            self.assertTrue(np.array_equal(grid[t], 4))

    def test_generate_map_no_obstacles(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=0, 
            num_hard_obstacles=0, 
            num_robots=self.num_robots,
            tetherDist=self.tetherDist
        )
        self.assertEqual(np.sum(grid == 1), 0)
        self.assertEqual(np.sum(grid == 2), 0)

    def test_generate_map_no_robots(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=self.num_soft_obstacles, 
            num_hard_obstacles=self.num_hard_obstacles, 
            num_robots=0,
            tetherDist=self.tetherDist
        )
        self.assertEqual(len(robots), 0)
    
    def test_generate_map_more_leaders(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=self.num_soft_obstacles, 
            num_hard_obstacles=self.num_hard_obstacles, 
            num_robots=5,
            tetherDist=self.tetherDist,
            num_leaders=3
        )
        self.assertEqual(len([robot for robot in robots if robot['role'] == 'leader']), 3)
        self.assertEqual(len([robot for robot in robots if robot['role'] == 'follower']), 2)

    def test_generate_map_multiple_targets(self):
        grid, robots, targets = generate_map(
            rowSize=self.size,
            colSize=self.size, 
            num_soft_obstacles=self.num_soft_obstacles, 
            num_hard_obstacles=self.num_hard_obstacles, 
            num_robots=self.num_robots,
            tetherDist=self.tetherDist,
            num_target=2
        )
        self.assertEqual(len(targets), 2)
        for target in targets:
            self.assertTrue(np.array_equal(grid[target], 4))

if __name__ == '__main__':
    unittest.main()