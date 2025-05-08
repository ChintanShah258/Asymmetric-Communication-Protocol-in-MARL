import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
SIMPLEGRID_PATH = os.path.abspath(os.path.join(ROOT_PATH, 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv
from marl_5 import train_MAPPO, encoder, decoder, critic_model, leader_policy, follower_policy
import time
import tensorflow as tf
import psutil
import shutil

# Hyperparemeter Tuning
import random

# Optimize 
#policy = tf.keras.mixed_precision.Policy('mixed_float16')
#tf.keras.mixed_precision.set_global_policy(policy)

# Check CPU Memory Usage (Referenced from: ChatGPT)
def log_memory_usage():
    process = psutil.Process()  # current process
    # RSS is the "Resident Set Size," the non-swapped physical memory used
    mem_info = process.memory_info().rss / (1024 * 1024)  # convert to MB
    print(f"Current process memory usage: {mem_info:.2f} MB")

def main(alg:str = "MAPPO"): # main pipeline goes here
    # Remove the evaluation CSV if it exists
    CSVPATH = 'logs/evaluation.csv'
    if os.path.exists(CSVPATH):
        os.remove(CSVPATH)
        print(f"Removed existing evaluation CSV file: {CSVPATH}")
    
    # Tracking GPU
    tf.debugging.set_log_device_placement(True)

    startTime = time.time()

    # Initialize the Environment
    env = SimpleGridEnv(
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

    if not (os.path.exists('models')):
        os.mkdir('models')

    # Load and Save the pretrained RAW models
    leader_policy.save('models/leader_policy_RAW.h5')
    follower_policy.save('models/follower_policy_RAW.h5')
    encoder.save('models/encoder_RAW.h5')
    decoder.save('models/decoder_RAW.h5')
    critic_model.save('models/critic_model_RAW.h5')

    # Compile the models
    leader_policy.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    follower_policy.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    encoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    decoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    critic_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # Define Hyperparameter Grid
    HYPERPARAMETER_COUNT = 1
    learning_rates = [round(random.uniform(0.0001, 0.01), 6) for _ in range(HYPERPARAMETER_COUNT)]
    episodes_list = [50] # Now we only consider a static number of episodes for simplicity
    contrastive_weights = [round(random.uniform(0.1, 1.0), 2) for _ in range(HYPERPARAMETER_COUNT)]
    reconstruction_weights = [round(random.uniform(0.1, 0.5), 2) for _ in range(HYPERPARAMETER_COUNT)]
    entropy_weights = [round(random.uniform(0.01, 0.1), 3) for _ in range(HYPERPARAMETER_COUNT)]
    max_steps_list = [100]

    best_score = -float('inf')
    best_params = None

    algoStartTime = time.time()

    training_count = highest_reward_training = 1 # count of trainings done
    # Perform Hyperparameter Tuning with Grid Search
    for lr in learning_rates:
        for episodes in episodes_list:
            startTimeInEpisode = time.time()

            for contrastive_weight in contrastive_weights:
                for reconstruction_weight in reconstruction_weights:
                    for entropy_weight in entropy_weights:
                        for max_steps in max_steps_list:
                            # Construct the Grid
                            params = {
                                "lr": lr, 
                                "episodes": episodes,
                                "contrastive_weight": contrastive_weight,
                                "reconstruction_weight": reconstruction_weight,
                                "entropy_weight": entropy_weight,
                                "max_steps": max_steps
                            }
                            print(f"Training with parameters: {params}")

                            tf.profiler.experimental.start('logs') # start GPU memory count
                            # Train the models
                            match alg:
                                case "MAPPO":
                                    train_MAPPO(
                                        episodes=episodes,
                                        leader_model=leader_policy,
                                        follower_model=follower_policy,
                                        encoder=encoder,
                                        decoder=decoder,
                                        critic_model=critic_model,
                                        env=env,
                                        hyperparams=params
                                    )
                                case _:
                                    raise ValueError("Algorithm not supported.")
                            log_memory_usage(); # log CPU RAM usage
                            tf.profiler.experimental.stop() # end GPU memory count

                            # Evaluate the models (e.g., based on cumulative reward or success rate)
                            # Retrieve cumulative reward from the environment's info
                            curr_cumulative_reward = env.get_info().get('cumulative_reward', 0)

                            print(f"Cumulative Reward in training {training_count}: {curr_cumulative_reward}")

                            # Save the best model
                            if curr_cumulative_reward > best_score:
                                best_score = curr_cumulative_reward
                                highest_reward_training = training_count
                                best_params = {
                                    "lr": lr, 
                                    "episodes": episodes,
                                    "contrastive_weight": contrastive_weight,
                                    "reconstruction_weight": reconstruction_weight,
                                    "entropy_weight": entropy_weight,
                                    "max_steps": max_steps,
                                    "cumulative_reward": curr_cumulative_reward,
                                    "training_count": training_count,
                                    "training_time": time.time() - startTimeInEpisode
                                }

                                # Save the best models
                                leader_policy.save('models/best_leader_model.h5')
                                follower_policy.save('models/best_follower_model.h5')
                                encoder.save('models/best_encoder_model.h5')
                                decoder.save('models/best_decoder_model.h5')
                                critic_model.save('models/best_critic_model.h5')
                                print(f"Best models are saved.")

            endTimeInEpisode = time.time()
            print(f"Time taken for this episode: {endTimeInEpisode - startTimeInEpisode} seconds")

    endTime = time.time()
    print(f"Time taken for hyperparameter tuning and training with the best ones: {endTime - algoStartTime} seconds")
    print(f"Time taken for the entire pipeline process is: {endTime - startTime} seconds")

    # Print Best Hyperparameters
    print(f"Total Training Processes: {training_count}") # total training count
    print(f"Best Set of Hyperparameters in Training Process {highest_reward_training}: {best_params}") # dict: best learning rate + episode
    print(f"Highest Cumulative Reward in Training Process {highest_reward_training}: {best_score}") # best success rate

    print(f"✅ Completed the training pipeline with {alg} successfully.")

if __name__ == "__main__":
    # Remove existing GPU logs
    GPU_LOG_PATH = "logs/plugins"
    if os.path.exists(GPU_LOG_PATH):
        shutil.rmtree(GPU_LOG_PATH)  # Remove directory and all its contents
        print(f"Removed existing GPU log directory: {GPU_LOG_PATH}")
    
    # Baseline: train with MAPPO
    main(alg="MAPPO")