import numpy as np
import sys
import os
SIMPLEGRID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from constants import ACTION_SPACE, LEADER_MESSAGE_SIZE
from simple_grid import SimpleGridEnv
import tensorflow as tf

class Agent:
    _id_counter = 0  # Class-level counter for auto-assigning agent IDs

    agent_id: int # Unique identifier of the agent
    role: str # Role of the agent
    position: tuple[int, int] # Position of the agent
    action_space: list[tuple[int, int]] # Action space of the agent
    env: SimpleGridEnv # Environment of the agent
    message: np.ndarray | tf.Tensor = None # Encoded message from the encoder-decoder

    # Algorithms
    encoder: tf.keras.Model
    decoder: tf.keras.Model
    leader_policy: tf.keras.Model
    follower_policy: tf.keras.Model

    def __init__(self, role: str):
        """
        Generalized Agent class for interacting with the SimpleGridEnv.
        
        Parameters:
        - role: String indicator of the agent's role (leader or follower)
        """
        self.role = role
        self.action_space = [action.value for action in ACTION_SPACE]

        # Auto Assigning an Agent ID
        self.agent_id = Agent._id_counter
        
        # Increment the counter for the next agent
        Agent._id_counter += 1

        # Load and Compile models from Parameters
        self.encoder = tf.keras.models.load_model(os.path.join("training", "models", "best_encoder_model.h5"))

        self.decoder = tf.keras.models.load_model(os.path.join("training", "models", "best_decoder_model.h5"))

        self.leader_policy = tf.keras.models.load_model(os.path.join("training", "models", "best_leader_model.h5"))

        self.follower_policy = tf.keras.models.load_model(os.path.join("training", "models", "best_follower_model.h5"))

    def act(self, observation: np.ndarray, message: np.ndarray | tf.Tensor = None) -> tuple[int, int]:
        """
        Args:
            observation (np.ndarray): partial observation of the environment

        Returns:
            tuple[int, int]: an action to be taken by the agent
        """
        # Flatten the grid observation to a 1D array
        observation = observation.flatten()

        # Ensure the observation has exactly 8 elements
        if observation.size < LEADER_MESSAGE_SIZE:
            # Pad with zeros if the observation has fewer than 8 elements
            observation = np.pad(observation, (0, LEADER_MESSAGE_SIZE - observation.size), mode='constant')
        elif observation.size > LEADER_MESSAGE_SIZE:
            # Truncate if the observation has more than 8 elements
            observation = observation[:LEADER_MESSAGE_SIZE]

        # Reshape observation to include batch dimension
        observation = observation.reshape(1, -1)
        
        if (self.role == "leader"):
            self.message = self.speak()
            # Leader decide based on observation
            predictions = self.leader_policy.predict(observation)
        else:
            if (message is None):
                # Empty message
                self.message = np.zeros((1, 32))
            else: 
                self.listen(message)
            
            # follower decide based on message
            # Ensure combined_input matches the expected input shape of the follower policy
            combined_input = np.concatenate((observation, self.message), axis=1)
            combined_input = combined_input[:, :LEADER_MESSAGE_SIZE]  # Adjust to match the expected shape
            predictions = self.follower_policy.predict(combined_input)
        
        # Take action with best probability
        if (type(predictions) == tf.Tensor):
            predictions = predictions.numpy()
        action = int(np.argmax(predictions))
        print(f"Action: {action}; Movement: {list(ACTION_SPACE)[action].name}")
        print(f"Movement Value: {list(ACTION_SPACE)[action].value}")
        return list(ACTION_SPACE)[action].value
    
    def listen(self, message: np.ndarray) -> None:
        """
        Args:
            message (dict): message from the leader agent
        """
        if (self.role == "follower"):
            self.message = self.decoder.predict(message)
        else:
            raise ValueError("Leader cannot listen")
    
    def speak(self) -> np.ndarray:
        """
        Returns:
            dict: message to be sent to the follower agent
        """
        if (self.role == "leader"):
            # should return encoder's output
            dummy_input = tf.random.normal((1, LEADER_MESSAGE_SIZE))
            encoded_message = self.encoder.predict(dummy_input)
            return encoded_message
        else:
            ValueError("Follower cannot speak")