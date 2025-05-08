import unittest
import numpy as np
import os 
import sys
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT)
SIMPLEGRID_PATH = os.path.join(ROOT, 'gym-simplegrid', 'gym_simplegrid', 'envs')
sys.path.append(SIMPLEGRID_PATH)
from agent import Agent
from simple_grid import SimpleGridEnv

class TestAgent(unittest.TestCase):

    def setUp(self):
        Agent._id_counter = 0

        self.env = SimpleGridEnv(
            render_mode="rgb_array", # numpy array representation
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.leader_agent = Agent(role="leader")
        self.follower_agent = Agent(role="follower")

    def test_agent_initialization(self):
        self.assertEqual(self.leader_agent.role, "leader")
        self.assertEqual(self.follower_agent.role, "follower")
        self.assertEqual(self.leader_agent.agent_id, 0)
        self.assertEqual(self.follower_agent.agent_id, 1)

    def test_agent_act(self): 
        # Leader
        observation = np.zeros((3, 3))
        action = self.leader_agent.act(observation)
        self.assertIsInstance(action, tuple)

        # Follower
        observation = np.zeros((4, 5))
        action = self.follower_agent.act(observation, self.leader_agent.message)
        self.assertIsInstance(action, tuple)

    def test_leader_speak(self):
        message = self.leader_agent.speak()
        self.assertIsInstance(message, np.ndarray)
        self.assertEqual(message.shape, (1, 32))

    def test_follower_listen(self):
        encoded_message = self.leader_agent.speak()
        self.assertIsInstance(encoded_message, np.ndarray)
        self.assertEqual(encoded_message.shape, (1, 32))
        self.follower_agent.listen(encoded_message)
        self.assertIsInstance(self.follower_agent.message, np.ndarray)
        self.assertEqual(self.follower_agent.message.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()