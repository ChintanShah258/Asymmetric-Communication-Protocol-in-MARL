import unittest

import os
import sys
TRAINING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)

import tensorflow as tf
from constants import ACTION_SPACE

class TestMoveAgent(unittest.TestCase):
    def setUp(self):
        # Load the Models from files
        self.leader_policy_model = tf.keras.models.load_model(os.path.join(TRAINING_PATH, "models", "best_leader_model.h5"))
        self.follower_policy_model = tf.keras.models.load_model(os.path.join(TRAINING_PATH, "models", "best_follower_model.h5"))
        self.encoder_decoder_model = tf.keras.models.load_model(os.path.join(TRAINING_PATH, "models", "best_policy_network.h5"))
    
    def test_display_model_summary(self):
        print("Leader Policy Model Summary:")
        self.leader_policy_model.summary()

        print("Follower Policy Model Summary:")
        self.follower_policy_model.summary()

        print("Encoder Decoder Model Summary:")
        self.encoder_decoder_model.summary()

    def test_policy_network_prediction(self):
        dummy_input = tf.random.normal((1, 8))
        prediction = self.leader_policy_model(dummy_input)
        print("Leader's Policy Network Prediction:")
        print(prediction.numpy())
        self.assertIsInstance(prediction, tf.Tensor)
        self.assertEqual(prediction.shape, (1, len(ACTION_SPACE))) # array length = total number of actions

        prediction = self.follower_policy_model(dummy_input)
        print("Follower's Policy Network Prediction:")
        print(prediction.numpy())
        self.assertIsInstance(prediction, tf.Tensor)
        self.assertEqual(prediction.shape, (1, len(ACTION_SPACE))) # array length = total number of actions
    
    def test_encoder_decoder_prediction(self):
        dummy_input = tf.random.normal((1, 8))
        prediction = self.encoder_decoder_model(dummy_input)
        print("Encoder Decoder Model Prediction:")
        print(prediction.numpy())
        self.assertIsInstance(prediction, tf.Tensor)
        self.assertEqual(prediction.shape, (1, 8)) # array of 8 values


if __name__ == '__main__':
    unittest.main()
