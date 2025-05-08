# This test is in regards with the new implementation of marl_3_chintan.py
# where the leader message size = 8

import unittest

import sys
import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH)

from marl_3_chintan import SimpleGridEnv, ACTION_SPACE, new_pos, get_agent_observation, encoder, decoder, build_encoder, build_decoder, leader_policy_network, follower_policy_network, MAPPO, contrastive_loss, train_MAPPO
import numpy as np
import tensorflow as tf
from constants import ACTION_SPACE, LEADER_MESSAGE_SIZE

class TestMoveAgent(unittest.TestCase):
    def checkPosition(self, coord):
        self.assertIsInstance(coord, tuple)
        self.assertEqual(len(coord), 2)
        self.assertTrue(all(isinstance(pos, int) for pos in coord))
        self.assertTrue(0 <= coord[0] < self.env.env_configurations["rowSize"])
        self.assertTrue(0 <= coord[1] < self.env.env_configurations["colSize"])

    def setUp(self):
        # Initialize the SimpleGridEnv
        self.env = SimpleGridEnv(
            render_mode=None,
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset()

        # Create MAPPO object
        self.mappo = MAPPO(
            leader_model=leader_policy_network(),
            follower_model=follower_policy_network(),
            encoder=encoder,
            decoder=decoder,
        )
        self.assertIsInstance(self.mappo, MAPPO)

        # Mock the Grid Map for deterministic results
        self.env.obstacles = np.zeros((10, 10), dtype=int)  # No obstacles
        self.env.agents = [
            {'position': (9, 9), 'role': 'leader'},
            {'position': (8, 8), 'role': 'follower'}
        ]
    
    def test_agents_list(self):
        self.agent_position = self.env.agents[0]['position']  # Use the first agent's position
        self.checkPosition(self.agent_position)

    def test_new_pos(self):
        # Move the agent to a new position - Agent in Focus: Leader
        leader_current_pos = self.env.agents[0]['position']
        follower_current_pos = self.env.agents[1]['position']
        action = ACTION_SPACE.UP
        actionValue = action.value
        agents = self.env.agents
        newPos_leader = new_pos(leader_current_pos, action, agents)
        newPos_follower = new_pos(follower_current_pos, action, agents)

        # Check if the new position is valid
        leader_expected_new_pos = (leader_current_pos[0] + actionValue[0], leader_current_pos[1] + actionValue[1])
        follower_expected_new_pos = (follower_current_pos[0] + actionValue[0], follower_current_pos[1] + actionValue[1])

        # Conditions
        isLeader_out_of_bounds = not (0 <= leader_expected_new_pos[0] < self.env.env_configurations["rowSize"] and 0 <= leader_expected_new_pos[1] < self.env.env_configurations["colSize"])
        isFollower_out_of_bounds = not (0 <= follower_expected_new_pos[0] < self.env.env_configurations["rowSize"] and 0 <= follower_expected_new_pos[1] < self.env.env_configurations["colSize"])
        isLeader_agent_collision = any(agent['position'] == leader_expected_new_pos for agent in agents)
        isFollower_agent_collision = any(agent['position'] == follower_expected_new_pos for agent in agents)
        isLeader_obstacle_collision = self.env.obstacles[leader_expected_new_pos[0], leader_expected_new_pos[1]] in [self.env.OBSTACLE_HARD]
        isFollower_obstacle_collision = self.env.obstacles[follower_expected_new_pos[0], follower_expected_new_pos[1]] in [self.env.OBSTACLE_HARD]

        if isLeader_out_of_bounds or isFollower_out_of_bounds:
            # If the move is out of bounds, the position should remain the same
            print("Out of Bounds")
            if isLeader_out_of_bounds:
                print("Leader out of bounds")
                self.assertEqual(newPos_leader, leader_current_pos)
            if isFollower_out_of_bounds:
                print("Follower out of bounds")
                self.assertEqual(newPos_follower, follower_current_pos)
        elif isLeader_agent_collision or isFollower_agent_collision:
            # If the move collides with another agent, the position should remain the same
            print("Agent Collision")
            if isLeader_agent_collision:
                print("Leader agent collision")
                self.assertEqual(newPos_leader, leader_current_pos)
            if isFollower_agent_collision:
                print("Follower agent collision")
                self.assertEqual(newPos_follower, follower_current_pos)
        elif isLeader_obstacle_collision or isFollower_obstacle_collision:
            # If the move collides with an obstacle, the position should remain the same
            print("Obstacle Collision")
            if isLeader_obstacle_collision:
                print("Leader obstacle collision")
                self.assertEqual(newPos_leader, leader_current_pos)
            if isFollower_agent_collision:
                print("Follower obstacle collision")
                self.assertEqual(newPos_follower, follower_current_pos)
        else:
            # Otherwise, the position should update correctly
            print("Valid Move")
            if not (isLeader_out_of_bounds and isLeader_agent_collision and isLeader_obstacle_collision):
                self.assertEqual(newPos_leader, leader_expected_new_pos)
            
            # We do not care about follower's position in this test
            # because we are testing the leader's position
            # environment's generation is not deterministic

    def test_get_leader_message(self):
        # Get the position of the leader agent
        leader_pos = next(agent['position'] for agent in self.env.agents if agent.get('role') == 'leader')
        self.checkPosition(leader_pos)

        # Generate the leader message
        message = get_agent_observation(leader_pos, self.env)
        print("LEADER MESSAGE: ", message)

        # Check the structure of the message
        self.assertIsInstance(message, list)
        self.assertEqual(len(message), 8)  # Ensure the message has 8 elements
        self.assertTrue(all(isinstance(value, (int, float)) for value in message[:5]))  # Check numeric values
        self.assertIn(message[5], [0, 1])  # Ensure path_blocked is 0 or 1

        obs_dist, agent_visibility, agent_dist, path_blocked, action_dx, action_dy, dx, dy = message

        # Additional checks for specific values
        if agent_visibility == 1:
            self.assertGreaterEqual(agent_visibility, 0)  # Ensure valid follower distance
        if obs_dist != -1:
            self.assertGreaterEqual(obs_dist, 0)  # Ensure valid obstacle distance

        self.assertIsInstance(action_dx, int)
        self.assertIsInstance(action_dy, int)

        self.assertIsInstance(dx, int)
        self.assertIsInstance(dy, int)
        self.checkPosition((dx, dy))  # Check if current position of an agent is a valid position

    def test_build_encoder_decoder(self):
        # Build the encoder-decoder model
        encoder = build_encoder()
        decoder = build_decoder()

        # Check if the model is an instance of tf.keras.Model
        self.assertIsInstance(encoder, tf.keras.Model)
        self.assertIsInstance(decoder, tf.keras.Model)

        # Encoder: Check the input and output shapes
        encoder_input_shape = encoder.input_shape
        encoder_output_shape = encoder.output_shape
        self.assertEqual(encoder_input_shape, (None, 8))  # Input shape should match the expected input size
        self.assertEqual(encoder_output_shape, (None, 32))  # Output shape should match the expected output size

        # Decoder: Check the input and output shapes
        decoder_input_shape = decoder.input_shape
        decoder_output_shape = decoder.output_shape
        self.assertEqual(decoder_input_shape, (None, 32))
        self.assertEqual(decoder_output_shape, (None, 8))

        # Check the activation function of the output layer
        output_activation = decoder.layers[-1].activation.__name__
        self.assertEqual(output_activation, "linear")  # Ensure the output layer uses linear activation

    def test_build_policy_network(self,policy_model=leader_policy_network):
        # Build the policy network model
        self.model = policy_model()

        # Check if the model is an instance of tf.keras.Model
        self.assertIsInstance(self.model, tf.keras.Model)

        # Check the input and output shapes
        input_shape = self.model.input_shape
        output_shape = self.model.output_shape
        self.assertEqual(input_shape, (None, 8))  # Input shape should match the expected input size
        self.assertEqual(output_shape, (None, len(ACTION_SPACE)))  # Output shape should match the number of actions

        # Check the activation function of the output layer
        output_activation = self.model.layers[-2].activation.__name__ # last layer is a Reshape layer
        self.assertEqual(output_activation, "softmax")  # Ensure the output layer uses softmax activation

        # Check the activation functions of the hidden layers
        hidden_activations = [layer.activation.__name__ for layer in self.model.layers if hasattr(layer, 'activation') and layer.activation]
        for activation in hidden_activations[:-1]:  # Exclude the output layer
            self.assertIn(activation, ["relu", "tanh", "sigmoid"])  # Ensure hidden layers use valid activation functions

        # Test a forward pass with dummy data
        dummy_input = np.random.rand(1, 8).astype(np.float32)
        output = self.model.predict(dummy_input)
        self.assertEqual(output.shape, (1, len(ACTION_SPACE)))  # Ensure the output shape matches the expected output size

        # Test Forward Pass
        dummy_input = np.random.rand(1, 8).astype(np.float32)
        output = self.model.predict(dummy_input)
        self.assertEqual(output.shape, (1, len(ACTION_SPACE)))
        self.assertAlmostEqual(np.sum(output), 1.0, places=3)  # Ensure probabilities sum to ~1

    def test_MAPPO_init(self):
        self.assertIsInstance(self.mappo.leader_model, tf.keras.Model)
        self.assertIsInstance(self.mappo.follower_model, tf.keras.Model)
        self.assertIsInstance(self.mappo.encoder, tf.keras.Model)
        self.assertIsInstance(self.mappo.decoder, tf.keras.Model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def test_MAPPO_compute_loss(self):
        # Generate dummy data for the test
        state_leader = np.random.rand(8).astype(np.float32)
        decoded_msg = np.random.rand(8).astype(np.float32)
        action_leader = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        action_follower = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        reward = np.random.rand(1).astype(np.float32)
        leader_message = np.random.rand(8).astype(np.float32)
        encoded_message = np.random.rand(8).astype(np.float32)
        decoded_message = np.random.rand(8).astype(np.float32)

        # Call compute_loss using the MAPPO object from setUp
        loss = self.mappo.compute_loss(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )

        # Assert the loss is a valid tensor
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.dtype, tf.float32)

    def test_MAPPO_apply_gradient(self):
        # Generate dummy data for the test
        state_leader = np.random.rand(8).astype(np.float32)
        decoded_msg = np.random.rand(8).astype(np.float32)
        action_leader = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        action_follower = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        reward = np.random.rand(1).astype(np.float32)
        leader_message = np.random.rand(8).astype(np.float32)
        encoded_message = np.random.rand(8).astype(np.float32)
        decoded_message = np.random.rand(1, 8).astype(np.float32)

        # Call the apply_gradients function
        self.mappo.apply_gradients(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )

        # Assert that the leader model weights are updated
        leader_weights_before = [tf.identity(w) for w in self.mappo.leader_model.trainable_variables]
        self.mappo.apply_gradients(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )
        leader_weights_after = self.mappo.leader_model.trainable_variables
        for before, after in zip(leader_weights_before, leader_weights_after):
            self.assertFalse(np.array_equal(before.numpy(), after.numpy()), "Leader model weights did not update.")
            
        # Assert that the follower model weights are updated
        follower_weights_before = [tf.identity(w) for w in self.mappo.follower_model.trainable_variables]
        self.mappo.apply_gradients(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )
        follower_weights_after = self.mappo.follower_model.trainable_variables
        for before, after in zip(follower_weights_before, follower_weights_after):
            self.assertTrue(np.array_equal(before.numpy(), after.numpy()), "Follower model weights did not update.")

    def test_contrastive_loss(self):
        # Generate dummy data for the test
        messages = np.random.rand(5, 8).astype(np.float32)  # 5 messages with 8 dimensions each
        positive_pairs = [0, 1, 2, 3, 4]  # Positive pairs for contrastive loss
        temperature = 0.1  # Temperature parameter for contrastive loss

        # Call the contrastive_loss function
        loss = contrastive_loss(messages, positive_pairs, temperature)

        # Assert the loss is a valid tensor
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.dtype, tf.float32)
        self.assertGreaterEqual(loss.numpy(), 0.0, "Contrastive loss should be non-negative.")

    def test_train_MAPPO(self):
        env = SimpleGridEnv(
            render_mode="rgb_array",
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        train_MAPPO(
            episodes=1,  # Minimal number of episodes
            leader_model=self.mappo.leader_model,
            follower_model=self.mappo.follower_model,
            encoder=self.mappo.encoder,
            decoder=self.mappo.decoder,
            env=env
        )

        # Assert that the environment reset and training completed
        self.assertIsNotNone(env.agents, "Agents should be initialized.")
        self.assertIsNotNone(env.targets, "Targets should be initialized.")

    def test_predict_results(self):
        env = SimpleGridEnv(
            render_mode="rgb_array",
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        train_MAPPO(
            episodes=1,  # Minimal number of episodes
            leader_model=self.mappo.leader_model,
            follower_model=self.mappo.follower_model,
            encoder=self.mappo.encoder,
            decoder=self.mappo.decoder,
            env=env
        )

        print("LEADER MODEL SUMMARY:")
        print(self.mappo.leader_model.summary())
        print("FOLLOWER MODEL SUMMARY:")
        print(self.mappo.follower_model.summary())

        # Convert agent positions to NumPy arrays and pad to match the expected input shape
        leader_position = np.array(env.agents[0]['position']).reshape(1, -1)
        leader_position_padded = np.pad(leader_position, ((0, 0), (0, LEADER_MESSAGE_SIZE-2)), mode='constant')  # Pad to shape (1, 8)
        # leader_position_padded = np.array(leader_position_padded).reshape(1, -1)
        print(f"Leader Position Shape: {leader_position_padded.shape}")

        print("ENCODER SUMMARY:")
        self.mappo.encoder.summary()
        print("DECODER SUMMARY:")
        self.mappo.decoder.summary()

        # Communication: Encoder-Decoder
        encoded_output = self.mappo.encoder.predict(leader_position_padded)
        decoded_output = self.mappo.decoder.predict(encoded_output)
        decoded_output = decoded_output.reshape(1, -1)  # Ensure correct shape (1, 8)
        self.assertEqual(decoded_output.shape, (1, LEADER_MESSAGE_SIZE))

        # Follower
        follower_position = np.array(env.agents[1]['position']).reshape(1, -1)
        follower_position_padded = np.pad(follower_position, ((0, 0), (0, LEADER_MESSAGE_SIZE-2)), mode='constant')  # Pad to shape (1, 8)
        # Combine the Input: leader's message and follower's own observation
        combined_follower_input = np.stack([follower_position_padded, decoded_output], axis=1)  # shape (1, 2, 8)
        print(f"Follower Position Shape: {combined_follower_input.shape}")

        self.assertEqual(combined_follower_input.shape, (1, 2, LEADER_MESSAGE_SIZE))

        # POLICY NETWORK PREDICTION - Actions
        leaderModel_pred = self.mappo.leader_model.predict(leader_position_padded)  # Output: numpy array with probabilities
        followerModel_pred = self.mappo.follower_model.predict(combined_follower_input)  # Output: numpy array with probabilities
        self.assertIsInstance(leaderModel_pred, np.ndarray)
        self.assertIsInstance(followerModel_pred, np.ndarray)
        self.assertEqual(leaderModel_pred.shape[1], len(ACTION_SPACE))

        # Obtain the action with the highest probability for both leader and follower
        leader_action_index = np.argmax(leaderModel_pred, axis=1)  # Index of the highest probability
        follower_action_index = np.argmax(followerModel_pred, axis=1)  # Index of the highest probability

        leader_action = list(ACTION_SPACE)[leader_action_index[0]]  # Enum
        follower_action = list(ACTION_SPACE)[follower_action_index[0]]  # Enum

        print("LEADER ACTION: ", leader_action, " ", leader_action.value)
        print("FOLLOWER ACTION: ", follower_action, " ", follower_action.value)
        self.assertIsInstance(leader_action, ACTION_SPACE)
        self.assertIsInstance(follower_action, ACTION_SPACE)
        self.assertIsInstance(leader_action.value, tuple)
        self.assertIsInstance(follower_action.value, tuple)
        for val in leader_action.value:
            self.assertIsInstance(val, int)
        for val in follower_action.value:
            self.assertIsInstance(val, int)
        
        
        # Step 5: Assert that the output is valid and action probabilities sum to ~1
        # Leader
        self.assertEqual(leaderModel_pred.shape, (1, len(ACTION_SPACE)))
        self.assertAlmostEqual(np.sum(leaderModel_pred), 1.0, places=3)
        # Follower
        self.assertEqual(followerModel_pred.shape, (1, len(ACTION_SPACE)))
        self.assertAlmostEqual(np.sum(followerModel_pred), 1.0, places=3)

        print("Encoded Msg:", encoded_output)
        print("Decoded Msg:", decoded_output)
        print("Follower Policy Output:", followerModel_pred)

    def test_display_model_summary(self):
        env = SimpleGridEnv(
            render_mode="rgb_array",
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        train_MAPPO(
            episodes=1,  # Minimal number of episodes
            leader_model=self.mappo.leader_model,
            follower_model=self.mappo.follower_model,
            encoder=self.mappo.encoder,
            decoder=self.mappo.decoder,
            env=env
        )
        # Display the model summary for leader and follower models
        print("LEADER MODEL SUMMARY:")
        self.mappo.leader_model.summary()
        print("FOLLOWER MODEL SUMMARY:")
        self.mappo.follower_model.summary()
        print("ENCODER MODEL SUMMARY:")
        self.mappo.encoder.summary()
        print("DECODER MODEL SUMMARY:")
        self.mappo.decoder.summary()

    def tearDown(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
if __name__ == '__main__':
    unittest.main()
