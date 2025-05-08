# Implementation is based on gym-simplegrid:
# https://github.com/damat-le/gym-simplegrid

from __future__ import annotations
import logging
import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT)
from generate_map import generate_map
from constants import ACTION_SPACE, REWARDS
import random
import numpy as np

class SimpleGridEnv(Env):
    """
    Simple Grid Environment

    The environment is a grid with obstacles (walls) and agents. The agents can move in one of the four cardinal directions. If they try to move over an obstacle or out of the grid bounds, they stay in place. Each agent has a unique color and a goal state of the same color. The environment is episodic, i.e. the episode ends when the agents reaches its goal.

    To initialise the grid, the user must decide where to put the walls on the grid. This can be done by either selecting an existing map or by passing a custom map. To load an existing map, the name of the map must be passed to the `obstacle_map` argument. Available pre-existing map names are "4x4" and "8x8". Conversely, if to load custom map, the user must provide a map correctly formatted. The map must be passed as a list of strings, where each string denotes a row of the grid and it is composed by a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. An example of a 4x4 map is the following:
    ["0000", 
     "0101", 
     "0001", 
     "1000"]

    Assume the environment is a grid of size (nrow, ncol). A state s of the environment is an elemente of gym.spaces.Discete(nrow*ncol), i.e. an integer between 0 and nrow * ncol - 1. Assume nrow=ncol=5 and s=10, to compute the (x,y) coordinates of s on the grid the following formula are used: x = s // ncol  and y = s % ncol.
     
    The user can also decide the starting and goal positions of the agent. This can be done by through the `options` dictionary in the `reset` method. The user can specify the starting and goal positions by adding the key-value pairs(`starts_xy`, v1) and `goals_xy`, v2), where v1 and v2 are both of type int (s) or tuple (x,y) and represent the agent starting and goal positions respectively. 
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], 'render_fps': 8}
    FREE: int = 0
    OBSTACLE_SOFT: int = 1
    OBSTACLE_HARD: int = 2
    AGENT: int = 3
    TARGET: int = 4
    MOVES = ACTION_SPACE

    obstacle_map: np.ndarray
    agents: np.ndarray | list[dict]
    targets: np.ndarray

    cumulative_reward: float

    leaders = []
    followers = []

    env_configurations = {
        "rowSize": 0,
        "colSize": 0,
        "num_soft_obstacles": 0,
        "num_hard_obstacles": 0,
        "num_robots": 0,
        "tetherDist": 0,
        "num_leaders": 0,
        "num_target": 0
    }

    OUT_OF_TETHER_COUNT = 0

    def __init__(
        self, 
        render_mode: str | None,
        rowSize: int,
        colSize: int,
        num_soft_obstacles: int,
        num_hard_obstacles: int,
        num_robots: int,
        tetherDist: int,
        num_leaders: int,
        num_target: int,
        # allow supplying maps
        obstacle_map: str | list[str] = None,
        agent_map: str | list[str] = None,
        target_map: str | list[str] = None,
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        obstacle_map: str | list[str]
            Map to be loaded. If a string is passed, the map is loaded from a set of pre-existing maps. The names of the available pre-existing maps are "4x4" and "8x8". If a list of strings is passed, the map provided by the user is parsed and loaded. The map must be a list of strings, where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. 
        agent_map: str | list[str]
            Map to be loaded for agents.
        target_map: str | list[str]
            Map to be loaded for targets.
        render_mode: str | None
            Mode to render the environment.
        rowSize: int
            Number of rows in the grid.
        colSize: int
            Number of columns in the grid.
        num_soft_obstacles: int
            Number of soft obstacles in the grid.
        num_hard_obstacles: int
            Number of hard obstacles in the grid.
        num_robots: int
            Number of robots in the grid.
        tetherDist: int
            Tether distance for robots.
        num_leaders: int
            Number of leader robots.
        num_target: int
            Number of targets in the grid.
        """

        if (obstacle_map is None) and (agent_map is None) and (target_map is None):
            # Record the env configurations
            self.env_configurations = {
                "rowSize": rowSize,
                "colSize": colSize,
                "num_soft_obstacles": num_soft_obstacles,
                "num_hard_obstacles": num_hard_obstacles,
                "num_robots": num_robots,
                "tetherDist": tetherDist,
                "num_leaders": num_leaders,
                "num_target": num_target
            }
            # Generate the initial map
            self.obstacles, self.robots, self.targets = generate_map(
                rowSize=rowSize,
                colSize=colSize,
                num_soft_obstacles=num_soft_obstacles,
                num_hard_obstacles=num_hard_obstacles,
                num_robots=num_robots,
                tetherDist=tetherDist,
                num_leaders=num_leaders,
                num_target=num_target
            )
            self.targets = np.zeros((rowSize, colSize), dtype=int)
            for target in self.robots:
                self.targets[target['position'][0], target['position'][1]] = self.TARGET
            self.agents = self.robots  # Initialize agents from robots
        else: 
            # Load the map
            self.obstacles = self.parse_obstacle_map(obstacle_map)
            self.agents = self.parse_agent_map(agent_map)
            self.targets = self.parse_target_map(target_map)
            self.targets = np.zeros((self.obstacles.shape[0], self.obstacles.shape[1]), dtype=int)
            for target in self.agents:
                self.targets[target['position'][0], target['position'][1]] = self.TARGET
            self.targets = np.array(self.targets)  # Ensure targets is a numpy array
            # Ensure agents is a list of dicts if generated, otherwise a numpy array
            if isinstance(self.agents, np.ndarray):
                self.agents = [{'position': tuple(pos)} for pos in np.argwhere(self.agents == self.AGENT)]
            # Record the env configurations
            self.env_configurations = {
                "rowSize": self.obstacles.shape[0],
                "colSize": self.obstacles.shape[1],
                "num_soft_obstacles": np.sum(self.obstacles == self.OBSTACLE_SOFT),
                "num_hard_obstacles": np.sum(self.obstacles == self.OBSTACLE_HARD),
                "num_robots": len(self.agents),
                "tetherDist": tetherDist,
                "num_leaders": len([agent for agent in self.agents if agent.get('role') == 'leader']),
                "num_target": np.sum(self.targets == self.TARGET)
            }

        # Separate agents into leaders and followers
        for agent in self.agents:
            if agent.get('role') == 'leader':
                self.leaders.append(agent)
            else:
                self.followers.append(agent)

        # Convert maps to numpy arrays
        self.obstacles = np.array(self.obstacles)
        self.targets = np.array(self.targets)

        # Ensure agents is a list of dicts if generated, otherwise a numpy array
        if isinstance(self.agents, np.ndarray):
            self.agents = [{'position': tuple(pos)} for pos in np.argwhere(self.agents == self.AGENT)]

        # Get the number of rows and columns of the grid
        self.nrow, self.ncol = self.obstacles.shape

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.observation_space = spaces.Discrete(n=self.nrow*self.ncol)

        self.cumulative_reward = 0

        # Rendering configuration
        self.fig = None

        self.render_mode = render_mode
        self.fps = self.metadata['render_fps']

    def reset(
            self, 
            seed: int | None = None, 
            options: dict = dict()
        ) -> dict:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        
        Returns
        -------
        dict
            The observation and info of the environment.
        """
        print("Starting reset process")  # Debug print

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        else: 
            seed=random.randint(0, 1000) if seed is None else seed

        self.cumulative_reward = 0
        self.OUT_OF_TETHER_COUNT = 0

        # Re-generating a new map within an episode
        self.obstacles, self.robots, self.targets = generate_map(
            rowSize=self.env_configurations["rowSize"], 
            colSize=self.env_configurations["colSize"],
            num_soft_obstacles=self.env_configurations["num_soft_obstacles"], 
            num_hard_obstacles=self.env_configurations["num_hard_obstacles"],
            num_robots=self.env_configurations["num_robots"], 
            tetherDist=self.env_configurations["tetherDist"], 
            num_leaders=self.env_configurations["num_leaders"], 
            num_target=self.env_configurations["num_target"]
        )

        # Separate agents into leaders and followers
        self.leaders = [agent for agent in self.robots if agent.get('role') == 'leader']
        self.followers = [agent for agent in self.robots if agent.get('role') == 'follower']

        # Update self.agents from self.robots
        self.agents = self.robots

        # Get valid start and goal positions
        self.start_xy = self.sample_valid_state_xy()
        self.goal_xy = self.sample_valid_state_xy()
        while self.goal_xy == self.start_xy:
            self.goal_xy = self.sample_valid_state_xy()
        
        # Initialize internal state
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.agent_action = None
        self.n_iter = 0

        # Check integrity
        self.integrity_checks()

        # render the environment according to mode: ansi (string), rgb_array (numpy.ndarray), human (None + matplotlib visualization)
        render = self.render()
        self.render_initial_frame()

        return {'observation': self.get_obs(), 'environment': render, **self.get_info()}
    
    def step(self, actions: dict[int, tuple[int, int]], isTraining: bool = False):
        """
        Take a step in the environment.

        Parameters
        ----------
        actions: dict[int, tuple[int, int]]
            The actions to be taken by the agents: a dictionary where keys are agent indices and values are coordinates (dx, dy).
        
        Returns
        -------
        observation : Any
            The observation of the current state.
        reward : float
            The reward obtained after taking the action.
        done : bool
            Whether the episode has ended.
        truncated : bool
            Whether the episode was truncated (always False in this implementation).
        info : dict
            Additional information about the environment.
        """
        self.agent_actions = actions

        total_reward = 0
        reset_required = False
        invalid_agents = set()

        original_positions = {agent_id: agent['position'] for agent_id, agent in enumerate(self.agents)}

        tether_violation_occurred = False
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]

            # Get the current position of the agent
            row, col = agent['position']

            # Get the delta of the action
            dx, dy = action

            # Compute the target position of the agent
            target_row = row + dx
            target_col = col + dy

            # Check if the move is valid
            if self.is_in_bounds(target_row, target_col):
                if self.is_free(target_row, target_col):
                    # Check if the move is within the tethered distance
                    if all(self.compute_distance((target_row, target_col), other_agent['position']) <= self.env_configurations["tetherDist"] for other_agent in self.agents):
                        agent['position'] = (target_row, target_col)
                        self.done = self.on_goal()
                        if self.done:
                            self.cumulative_reward += REWARDS.TARGET.value
                            print("Target reached")
                            reset_required = True
                            break
                    else:
                        # Reset the game if the move is out of tethered distance
                        print("Out of tethered distance")
                        print(f"Agent {agent_id} out of tethered distance")
                        tether_violation_occurred = True
                        invalid_agents.add(agent_id)
                        reset_required = True
                else:
                    if (target_row, target_col) in [other_agent['position'] for other_agent in self.agents if other_agent != agent]:
                        # Reset the game if the agent crashes with another agent
                        print(f"Agent {agent_id} crashed with another agent")
                        invalid_agents.add(agent_id)
                        reset_required = True
                    elif self.obstacles[target_row, target_col] == self.OBSTACLE_HARD:
                        # Reset the game if the agent bumps into a hard obstacle
                        print("Hard obstacle encountered")
                        print(f"Agent {agent_id} encountered a hard obstacle")
                        invalid_agents.add(agent_id)
                        reset_required = True
            else:
                # Reset the game if an agent attempts to go out of bound
                print("Out of bounds")
                print(f"Agent {agent_id} moved out of bounds")
                invalid_agents.add(agent_id)
                reset_required = True
        
        # Increment tether count BEFORE rewards
        if tether_violation_occurred:
            self.OUT_OF_TETHER_COUNT += 1

        # Calculate Reward
        total_reward = 0
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            row, col = agent['position']
            dx, dy = action
            target_row = row + dx
            target_col = col + dy

            reward = self.get_reward(
                x=target_row,
                y=target_col,
                out_of_tether_count=self.OUT_OF_TETHER_COUNT
            )
            total_reward += reward
        
        self.cumulative_reward += total_reward

        # Reverse invalid moves
        for agent_id in invalid_agents:
            self.agents[agent_id]['position'] = original_positions[agent_id]
        
        self.n_iter += 1

        if reset_required:
            if isTraining:
                # we allow it to proceed in training mode
                pass
            else: # only resets in execution mode
                self.reset()

        # if self.render_mode == "human":
        self.render()

        return self.get_obs(), total_reward, self.done, False, {
            **self.get_info(),
            'agent_positions': {agent_id: agent['position'] for agent_id, agent in enumerate(self.agents)},
            'out_of_tether_count': self.OUT_OF_TETHER_COUNT
        }

    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Initialise the grid.

        The grid is described by a map, i.e. a list of strings where each string denotes a row of the grid and is a sequence of 0s, 1s, and 2s, where 0 denotes a free cell, 1 denotes a soft obstacle, and 2 denotes a hard obstacle.

        The grid can be initialised by passing a map name or a custom map.
        If a map name is passed, the map is loaded from a set of pre-existing maps. If a custom map is passed, the map provided by the user is parsed and loaded.

        Examples
        --------
        >>> my_map = ["001", "020", "011"]
        >>> SimpleGridEnv.parse_obstacle_map(my_map)
        array([[0, 0, 1],
               [0, 2, 0],
               [0, 1, 1]])
        """
        if isinstance(obstacle_map, list) or isinstance(obstacle_map, np.ndarray):
            map_str = np.asarray(obstacle_map, dtype='c') # convert to char
            map_int = np.asarray(map_str, dtype=int) # convert to int
        elif isinstance(obstacle_map, str):
            map_list = obstacle_map.splitlines()  # convert to list of strings
            map_str = np.asarray([list(row) for row in map_list], dtype='c')  # convert to char array
            map_int = np.asarray(map_str, dtype=int)  # convert to int array
        else:
            raise ValueError("You must provide a valid obstacle map.")
        return map_int

    def parse_agent_map(self, agent_map) -> np.ndarray:
        """
        Parse the agent map.

        The agent map is described by a list of strings where each string denotes a row of the grid and is a sequence of 0s and 3s, where 0 denotes a free cell and 3 denotes an agent cell.

        Examples
        --------
        >>> my_map = ["003", "030", "033"]
        >>> SimpleGridEnv.parse_agent_map(my_map)
        array([[0, 0, 3],
               [0, 3, 0],
               [0, 3, 3]])
        """
        agents = []
        if isinstance(agent_map, list):
            map_str = np.asarray(agent_map, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
        elif isinstance(agent_map, str):
            map_list = agent_map.splitlines()  # convert to list of strings
            map_str = np.asarray([list(row) for row in map_list], dtype='c')  # convert to char array
            map_int = np.asarray(map_str, dtype=int)  # convert to int array
        else:
            raise ValueError("You must provide a valid agent map.")

        # Extract agent positions
        agent_positions = np.argwhere(map_int == self.AGENT)
        for pos in agent_positions:
            agents.append({'position': tuple(pos)})
        
        return agents

    def parse_target_map(self, target_map) -> np.ndarray:
        """
        Parse the target map.

        The target map is described by a list of strings where each string denotes a row of the grid and is a sequence of 0s and 4s, where 0 denotes a free cell and 4 denotes a target cell.

        Examples
        --------
        >>> my_map = ["004", "040", "044"]
        >>> SimpleGridEnv.parse_target_map(my_map)
        array([[0, 0, 4],
               [0, 4, 0],
               [0, 4, 4]])
        """
        if isinstance(target_map, list):
            map_str = np.asarray(target_map, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
        elif isinstance(target_map, str):
            map_list = target_map.splitlines()  # convert to list of strings
            map_str = np.asarray([list(row) for row in map_list], dtype='c')  # convert to char array
            map_int = np.asarray(map_str, dtype=int)  # convert to int array
        else:
            raise ValueError("You must provide a valid target map.")
        
        return map_int

    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        Parse the value of an option of type state from the dictionary of options usually passed to the reset method. Such value denotes a position on the map and it must be an int or a tuple.

        Parameters
        ----------
        state_name: str
            Name of the state to be parsed.
        options: dict
            Dictionary of options.
        
        Returns
        -------
        tuple
            The position on the map.
        """
        try:
            state = options[state_name]
            if isinstance(state, int):
                return self.to_xy(state)
            elif isinstance(state, tuple):
                return state
            else:
                raise TypeError(f'Allowed types for `{state_name}` are int or tuple.')
        except KeyError:
            state = self.sample_valid_state_xy()
            logger = logging.getLogger()
            logger.info(f'Key `{state_name}` not found in `options`. Random sampling a valid value for it:')
            logger.info(f'...`{state_name}` has value: {state}')
            return state

    def sample_valid_state_xy(self) -> tuple:
        """
        Samples a valid (x, y) position within the grid environment.
        This method repeatedly samples a state from the observation space and converts it to (x, y) coordinates
        until a free position is found.

        Returns
        -------
            tuple: A tuple representing a valid (x, y) position within the grid.
        """
        free_positions = np.argwhere(self.obstacles == self.FREE)  # Get all free positions
        if len(free_positions) == 0:
            raise ValueError("No valid free cells available in the grid.")  # Ensure at least one free space exists

        idx = np.random.choice(len(free_positions))  # Randomly select from free positions
        return tuple(free_positions[idx])  # Convert to (x, y)
    
    def integrity_checks(self) -> None:
        """
        Perform integrity checks to ensure the environment is correctly set up.
        """
        # Check that start and goal positions do not overlap with obstacles
        assert self.obstacles[self.start_xy] == self.FREE, \
            f"Start position {self.start_xy} overlaps with an obstacle."
        assert self.obstacles[self.goal_xy] == self.FREE, \
            f"Goal position {self.goal_xy} overlaps with an obstacle."
        
        # Check that start and goal positions are within the grid bounds
        assert self.is_in_bounds(*self.start_xy), \
            f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(*self.goal_xy), \
            f"Goal position {self.goal_xy} is out of bounds."
        
        # Check that start and goal positions do not overlap with each other
        assert self.start_xy != self.goal_xy, \
            f"Start position {self.start_xy} overlaps with the goal position {self.goal_xy}."
        
        # check that goals do not overlap with walls
        assert self.obstacles[self.start_xy] == self.FREE, \
            f"Start position {self.start_xy} overlaps with a wall."
        assert self.obstacles[self.goal_xy] == self.FREE, \
            f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(*self.start_xy), \
            f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(*self.goal_xy), \
            f"Goal position {self.goal_xy} is out of bounds."
        
    def to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on any of the possible targets.
        """
        agent_x, agent_y = self.agent_xy
        self.targets = np.array(self.targets)
        if agent_x >= self.targets.shape[0] or agent_y >= self.targets.shape[1]:
            return False
        return self.targets[agent_x, agent_y] == self.TARGET

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        return self.obstacles[row, col] == self.FREE
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def get_reward(self, x: int, y: int, out_of_tether_count: int = 0) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x, y):
            return REWARDS.WALL.value
        elif not self.is_free(x, y):
            if (x, y) in [agent['position'] for agent in self.agents if agent['position'] != self.agent_xy]:
                return REWARDS.CRASH.value
            elif self.obstacles[x, y] == self.OBSTACLE_SOFT:
                return REWARDS.SOFT_OBSTACLE.value
            elif self.obstacles[x, y] == self.OBSTACLE_HARD:
                return REWARDS.HARD_OBSTACLE.value
        elif (x, y) == self.goal_xy:
            return REWARDS.TARGET.value
        elif not all(self.compute_distance((x, y), other_agent['position']) <= self.env_configurations["tetherDist"] for other_agent in self.agents):
            return REWARDS.OUT_OF_TETHER.value * out_of_tether_count
        else:
            # if stay
            rewards: int = 0
            if (x, y) == self.agent_xy:
                rewards += REWARDS.STAY.value
            # step
            rewards += REWARDS.STEP.value
            return rewards
        return 0.0  # Ensure a float value is always returned

    def get_obs(self) -> dict:
        return {agent_id: self.to_s(*agent['position']) for agent_id, agent in enumerate(self.agents)}
    
    def get_info(self) -> dict:
        return {
            'agent_positions': {agent_id: agent['position'] for agent_id, agent in enumerate(self.agents)},
            'n_iter': self.n_iter,
            'cumulative_reward': self.cumulative_reward
        }

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            return None
        
        elif self.render_mode == "ansi":
            s = f"{self.n_iter},{self.agent_xy[0]},{self.agent_xy[1]},{self.reward},{self.done},{self.agent_action}\n"
            #print(s)
            return s

        elif self.render_mode == "rgb_array":
            self.render_frame()
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
        elif self.render_mode == "human":
            self.render_frame()
            plt.pause(1/self.fps)
            return None
        
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def render_frame(self):
        if self.fig is None:
            self.render_initial_frame()
            self.fig.canvas.mpl_connect('close_event', self.close)
        else:
            self.update_agent_patch()
        self.ax.set_title(f"Step: {self.n_iter}, Reward: {self.reward}")
    
    def create_agent_patch(self, agent):
        """
        Create a Circle patch for the agent.

        @NOTE: If agent position is (x,y) then, to properly render it, we have to pass (y,x) as center to the Circle patch.
        """
        x, y = agent['position']
        color = 'blue' if agent.get('role') == 'leader' else 'orange'
        return mpl.patches.Circle(
            (y + .5, x + .5), 
            0.3, 
            facecolor=color, 
            fill=True, 
            edgecolor='black', 
            linewidth=1.5,
            zorder=100,
        )

    def update_agent_patch(self):
        """
        Update the position of the agent patches.
        """
        for agent_patch, agent in zip(self.agent_patches, self.agents):
            x, y = agent['position']
            agent_patch.center = (y + .5, x + .5)
        return None
    
    def render_initial_frame(self):
        """
        Render the initial frame.

        @NOTE: 0: free cell (white), 1: soft obstacle (light gray), 2: hard obstacle (black), 3: start (red), 4: goal (green)
        """
        data = self.obstacles.copy()
        print(f"Data: {data}")
        print(f"Agents: {self.agents}")
        for agent in self.agents:
            x, y = agent['position']
            if data[x, y] == self.FREE:  # Only set if it's a free cell
                data[x, y] = self.AGENT  # Mark agent starting positions in red
        for target in self.targets:
            x, y = target
            data[x, y] = self.TARGET  # Mark target positions in green

        colors = ['white', 'lightgray', 'black', 'red', 'green']
        bounds = [i-0.1 for i in range(6)]

        # create discrete colormap
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.ion()
        fig, ax = plt.subplots(tight_layout=True)
        self.fig = fig
        self.ax = ax

        ax.grid(axis='both', color='k', linewidth=1.3)
        ax.set_xticks(np.arange(0, data.shape[1], 1))
        ax.set_yticks(np.arange(0, data.shape[0], 1))
        ax.tick_params(
            bottom=False, 
            top=False, 
            left=False, 
            right=False, 
            labelbottom=False, 
            labelleft=False
        ) 

        # draw the grid
        ax.imshow(
            data, 
            cmap=cmap, 
            norm=norm,
            extent=[0, data.shape[1], data.shape[0], 0],
            interpolation='none'
        )

        # Create white holes on agent starting positions
        for agent in self.agents:
            wp = self.create_white_patch(*agent['position'])
            ax.add_patch(wp)

        # Create agent patches
        self.agent_patches = []
        for agent in self.agents:
            agent_patch = self.create_agent_patch(agent)
            self.agent_patches.append(agent_patch)
            ax.add_patch(agent_patch)

        plt.show()  # Ensure the plot is displayed

        return None

    def create_white_patch(self, x, y):
        """
        Render a white patch in the given position.
        """
        return mpl.patches.Circle(
            (y+.5, x+.5), 
            0.4, 
            color='white', 
            fill=True, 
            zorder=99,
        )

    def close(self, *args):
        """
        Close the environment.
        """
        if self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception as e:
                pass
            self.fig = None

    def compute_distance(self, point1: tuple[int, int], point2: tuple[int, int]) -> float:
        """
        Compute the Euclidean distance between two points in the grid.

        Parameters
        ----------
        point1: tuple[int, int]
            The first point.
        point2: tuple[int, int]
            The second point.

        Returns
        -------
        float
            The distance between the two points.
        """
        # Round down for diagonal distances
        return np.floor(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))

if __name__ == "__main__":
    import time

    obstacle_map = ["0000", "0101", "0001", "1000"]
    agent_map = ["0030", "0000", "0000", "0000"]
    target_map = ["0000", "0000", "0000", "0004"]

    env = SimpleGridEnv(
        #obstacle_map=obstacle_map,
        #agent_map=agent_map,
        #target_map=target_map,
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
    env.reset()
    env.render()
    for _ in range(10):
        actions = {agent_id: random.choice([move.value for move in env.MOVES]) for agent_id in range(len(env.agents))}
        print(f"Actions: {actions}")
        time.sleep(3)
        env.step(actions)
        if (env.done):
            break
    print(f"cumulative reward: {env.cumulative_reward}")
    env.close()