import unittest
import sys
import os
import numpy as np
SIMPLEGRID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv
ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
sys.path.append(ROOT_PATH)
from constants import ACTION_SPACE

class TestSimpleGridEnv(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, 'env'):
            self.env.close()

    def test_samplevalidStatexy(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        valid_state:tuple = self.env.sample_valid_state_xy()
        self.assertEqual(type(valid_state), tuple)
        self.assertEqual(len(valid_state), 2)
    
    def test_parseObstacleMap_listFormat(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_obstacle_map = self.env.parse_obstacle_map(
            obstacle_map=obstacle_map
        )
        self.assertEqual(type(parsed_obstacle_map), np.ndarray)
        self.assertEqual(len(parsed_obstacle_map), 4)

    def test_parseObstacleMap_strFormat(self):
        # Define a simple map
        obstacle_map = "0000\n0101\n0001\n1000"
        agent_map = "0030\n0000\n0000\n0000"
        target_map = "0000\n0000\n0000\n0004"

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_obstacle_map = self.env.parse_obstacle_map(
            obstacle_map=obstacle_map
        )
        self.assertEqual(type(parsed_obstacle_map), np.ndarray)
        self.assertEqual(len(parsed_obstacle_map), 4)

    def test_parseAgentMap_listFormat(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_agent_map = self.env.parse_agent_map(
            agent_map=agent_map
        )
        self.assertEqual(type(parsed_agent_map), list)
        self.assertEqual(len(parsed_agent_map), 1)

    def test_parseAgentMap_strFormat(self):
        # Define a simple map
        obstacle_map = "0000\n0101\n0001\n1000"
        agent_map = "0030\n0000\n0000\n0000"
        target_map = "0000\n0000\n0000\n0004"

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_agent_map = self.env.parse_agent_map(
            agent_map=agent_map
        )
        self.assertEqual(type(parsed_agent_map), list)
        self.assertEqual(len(parsed_agent_map), 1)

    def test_parseTargetMap_listFormat(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_target_map = self.env.parse_target_map(
            target_map=target_map
        )
        self.assertEqual(type(parsed_target_map), np.ndarray)
        self.assertEqual(len(parsed_target_map), 4)

    def test_parseTargetMap_strFormat(self):
        # Define a simple map
        obstacle_map = "0000\n0101\n0001\n1000"
        agent_map = "0030\n0000\n0000\n0000"
        target_map = "0000\n0000\n0000\n0004"

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_target_map = self.env.parse_target_map(
            target_map=target_map
        )
        self.assertEqual(type(parsed_target_map), np.ndarray)
        self.assertEqual(len(parsed_target_map), 4)

    def test_initialize_with_predefined_maps(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (4, 4))
        self.assertEqual(len(self.env.agents), 1)

    def test_initialize_without_predefined_maps(self):
        # Initialize the environment without predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (4, 4))
        self.assertEqual(len(self.env.agents), 1)

    def test_reset_with_predefined_maps(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        initial_state = self.env.reset()
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (1, 2))
        self.assertEqual(len(self.env.agents), 1)
        self.assertEqual(type(initial_state), dict)
        self.assertIn('observation', initial_state)

    def test_reset_without_predefined_maps(self):
        # Initialize the environment without predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        initial_state = self.env.reset()
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (1, 2))
        self.assertEqual(len(self.env.agents), 1)
        self.assertEqual(type(initial_state), dict)
        self.assertIn('observation', initial_state)

    def test_parse_state_option_int(self):
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset(seed=42)
        state = self.env.parse_state_option("start_loc", {"start_loc": 5})
        self.assertEqual(state, (1, 1))  # Assuming a 4x4 grid

    def test_parse_state_option_tuple(self):
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset(seed=42)
        state = self.env.parse_state_option("goal_loc", {"goal_loc": (2, 2)})
        self.assertEqual(state, (2, 2))

    def test_parse_state_option_random(self):
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset(seed=42)
        state = self.env.parse_state_option("start_loc", {})
        self.assertEqual(type(state), tuple)
        self.assertEqual(len(state), 2)

    def test_render_mode_rgb_array(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with render_mode="rgb_array"
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="rgb_array",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.env.reset()
        img = self.env.render()
        self.assertEqual(type(img), np.ndarray)
        self.assertEqual(img.shape[2], 4)  # Check if the image has 4 channels (RGBA)
        print(f"Image shape: {img.shape}")
        print(f"Image: {img}")

    def test_render_mode_ansi(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with render_mode="ansi"
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="ansi",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.env.reset()
        ansi_output = self.env.render()
        self.assertEqual(type(ansi_output), str)
        self.assertIn(",", ansi_output)  # Check if the output contains comma-separated values
        print(f"ANSI Output: {ansi_output}")

    def test_training_mode_no_reset(self):
        # Initialize the environment
        self.env = SimpleGridEnv(
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset()

        self.env.agents[0]['position'] = (0, 0)  # force top row position
        initial_position = self.env.agents[0]['position']
        actions = {0: ACTION_SPACE.UP.value} # obviously invalid!
        obs, reward, done, truncated, info = self.env.step(actions, isTraining=True)

        self.assertEqual(self.env.agents[0]['position'], initial_position)
        self.assertFalse(done)
        self.assertIn('agent_positions', info)
        self.assertEqual(info['agent_positions'][0], initial_position)

    def test_compute_distance(self):
        self.env = SimpleGridEnv(
            obstacle_map=None,
            agent_map=None,
            target_map=None,
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=0,
            num_hard_obstacles=0,
            num_robots=0,
            tetherDist=2,
            num_leaders=0,
            num_target=0
        )
        # Same point
        self.assertEqual(self.env.compute_distance((1, 1), (1, 1)), 0)

        # Horizontal
        self.assertEqual(self.env.compute_distance((1, 1), (1, 3)), 2)

        # Vertical
        self.assertEqual(self.env.compute_distance((3, 1), (1, 1)), 2)

        # Diagonal
        self.assertEqual(self.env.compute_distance((1, 1), (2, 2)), 1)

        # Farther diagonal
        self.assertEqual(self.env.compute_distance((0, 0), (3, 3)), 4)

    def test_out_of_tether_penalty(self):
        self.env = SimpleGridEnv(
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=0,
            num_hard_obstacles=0,
            num_robots=2,  # critical: need more than 1 agent to check tether
            tetherDist=1,
            num_leaders=1,
            num_target=1
        )
        self.env.reset()

        # Force agent positions
        self.env.agents[0]['position'] = (0, 0)
        self.env.agents[1]['position'] = (3, 3)  # far away that exceeds tetherDist=1

        initial_positions = {agent_id: agent['position'] for agent_id, agent in enumerate(self.env.agents)}

        # Move agent 0 further away
        actions = {0: (1, 1), 1: (0, 0)}  # arbitrary move, but will still violate tether

        obs, reward, done, truncated, info = self.env.step(actions, isTraining=True)

        # Ensure agent 0 is reset to its original position (because it violated tether)
        self.assertEqual(self.env.agents[0]['position'], initial_positions[0])
        # Ensure agent 1 is reset to its original position
        self.assertEqual(self.env.agents[1]['position'], initial_positions[1])
        # Confirm training mode does not trigger done
        self.assertFalse(done)
        # Confirm that agent positions are reported in info
        self.assertIn('agent_positions', info)
        self.assertEqual(info['agent_positions'], initial_positions)
        # Check the out of tether count in log
        self.assertIn('out_of_tether_count', info)
        print(f"Out of tether count: {info['out_of_tether_count']}")


if __name__ == '__main__':
    unittest.main()