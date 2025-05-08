import tensorflow as tf
import numpy as np
import datetime
import os 
import sys
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT)
SIMPLEGRID_PATH = os.path.join(ROOT, 'gym-simplegrid', 'gym_simplegrid', 'envs')
sys.path.append(SIMPLEGRID_PATH)

from agent import Agent
from simple_grid import SimpleGridEnv
from partial_observation import get_partial_observation

class Proxy:
    def __init__(self):
        self.env = SimpleGridEnv(
            render_mode="human",
            rowSize=5,
            colSize=5,
            num_soft_obstacles=5,
            num_hard_obstacles=2,
            num_robots=2,
            tetherDist=1,
            num_leaders=1,
            num_target=1
        )
        Agent._id_counter = 0
        self.agents = []

    def initialize_agents(self):
        self.agents = []
        for idx, agent_info in enumerate(self.env.agents):
            agent = Agent(role=agent_info.get('role', 'follower'))
            agent.agent_id = idx
            agent.position = agent_info['position']
            agent.message = np.zeros((1, 32))  # Initialize with dummy message
            self.agents.append(agent)

    def run_simulations(self, rounds: int):
        self.env.reset()
        self.initialize_agents()
        self.env.render_initial_frame()  # Render only once initially

        for round in range(rounds):
            print(f"--- Round {round + 1} ---")

            actions = {}
            for agent in self.agents:
                observation = get_partial_observation(
                    grid=self.env.obstacles,
                    agent_position=agent.position,
                    observation_radius=2
                )

                if agent.role == "follower" and agent.message is None:
                    agent.message = np.zeros((1, 32))

                action = agent.act(observation=observation, message=agent.message)
                actions[agent.agent_id] = action

                if agent.role == "leader":
                    for follower in self.agents:
                        if follower.role == "follower":
                            follower.message = agent.message

            print(f"Actions: {actions}")
            obs, reward, done, truncated, info = self.env.step(actions)

            self.env.update_agent_patch()  # Only update the existing agent patches
            self.env.ax.set_title(f"Step: {self.env.n_iter}, Reward: {reward}")
            self.env.fig.canvas.draw()
            self.env.fig.canvas.flush_events()

            if not os.path.exists("Simulations"):
                os.makedirs("Simulations")
            plt.savefig(os.path.join("Simulations", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_step_{self.env.n_iter}.png"), dpi=300)

            if done:
                print("Episode finished early. Resetting environment.")
                self.env.reset()
                self.initialize_agents()
                self.env.render_initial_frame()  # Reuse same figure

        print(f"Cumulative reward: {self.env.cumulative_reward}")
        self.env.close()
        plt.close('all')

if __name__ == "__main__":
    proxy = Proxy()
    proxy.run_simulations(10)
    print("Simulation completed successfully.")