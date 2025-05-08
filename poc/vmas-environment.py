import vmas # Import the vmas package
import numpy as np

env = vmas.make_env(
        scenario="waterfall", # can be scenario name or BaseScenario class
        num_envs=5,
        device="cpu", # Or "cuda" for GPU
        continuous_actions=True,
        wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
        max_steps=None, # Defines the horizon. None is infinite horizon.
        seed=None, # Seed of the environment
        dict_spaces=False, # By default tuple spaces are used with each element in the tuple being an agent.
        # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
        grad_enabled=False, # If grad_enabled the simulator is differentiable and gradients can flow from output to input
        terminated_truncated=False, # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag
        # **kwargs # Additional arguments you want to pass to the scenario initialization
    )

# Run the environment for a few steps
for _ in range(10):
    # Sample random actions for each agent
    actions = np.asarray([env.action_space.sample() for _ in range(env.num_envs)])
        
    # Take a step in the environment
    obs, rewards, done, info = env.step(actions)
        
    # Check if any of the environments are done
    if any(done):
        break