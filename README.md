## Our Deliverables
* [Project Proposal](./proposal%20doc/MARL_Autonomous_Vehicle_Proposal.pdf)
* [Presenting Our Proposal](./MARL_autonomous_vehicle_proposal_presentation.pdf)
* [A Research Summary](./MARL-Lane_Changing-Presentation.pdf)
* Our Interim Results: [[Project Presentation]](./MARL_autonomous_vehicle_project_presentation.pdf) [[Project Report]](./COMP5801%20Multi%20Agent%20RL%20Project%20Report%20-%20Kelvin,%20Kimia,%20Chintan.pdf)

## Environment Outline
<img src="poc/output_tethered_markers.png" alt="Proof of Concept Output" width="400" height="400"/>

## LaTex Guide for Writing Reports
To work with LaTex locally on your IDE, follow the steps below: 
1. Install [MikTex](https://miktex.org/) as the LaTeX Distribution.
2. Install [Perl](https://strawberryperl.com/).
3. Install the VS Code Extension for previewing the LaTex in a PDF format, such as **LaTeX Workshop**.
4. Previewing and Exporting will be available in the `.tex` file. 

## Prerequisites for Executable Codes in this Project
* Create a Virtual Environment with this command on a terminal: `py -m venv marl_env`.
* Update pip to its latest version with this command: `py -m pip install --upgrade pip`.
* From the Virtual Environment, install all the dependencies with this command: `pip install -r requirements.txt`, or `conda list -e > requirements.txt` if you use conda. Note: Use a **full path** for conda.
* Or you may find creating a virtual environment useful if you keep facing dependency issues. Just run the following command before installing libraries from pip: 
`
    python -m venv marl_env
    source marl_env/bin/activate  # (On Windowks: marl_env\Scripts\activate)
`.
* Atari is available via Gymnasium: <url>https://www.gymlibrary.dev/environments/atari/index.html</url>
* Clone the repository <url>https://github.com/damat-le/gym-simplegrid.git</url> to get the codes of a simple grid-based environment for customization. 

## Environment's Representation
### Array Representation
The rgb_array shape Image shape: (480, 640, 4) represents the dimensions and color channels of the rendered image of the environment. Here's a breakdown of what each dimension represents:

Breakdown of the Shape
Height (480): The first dimension (480) represents the **height** of the image in pixels.
Width (640): The second dimension (640) represents the **width** of the image in pixels.
Color Channels (4): The third dimension (4) represents the **color channels** of the image. In this case, there are 4 channels, which typically correspond to the RGBA color model:
* R: Red channel
* G: Green channel
* B: Blue channel
* A: Alpha channel (transparency)

Explanation
Height and Width: The height and width of the image determine the resolution of the rendered image. In this case, the image has a resolution of 480x640 pixels.
Color Channels: The 4 color channels (RGBA) provide information about the color and transparency of each pixel in the image. The alpha channel allows for transparency effects, which can be useful for rendering overlapping objects or semi-transparent elements.

### String Representation
When the environment is rendered in ansi mode, the render method generates a string that represents the current state of the environment. This string typically includes information such as the current step, the agent's position, the reward obtained, whether the episode has ended, and the agent's action.

Example:

`Step: 5, Agent Position: (2, 3), Reward: -1, Done: False, Action: (1, 0)`

### Visual Representation for debugging
![Visual Representation](<env_human_render.png>)

View more states in a simulated game in this directory: [`Simulations/`](./Simulations/).
You might need to run the script [`runtime-environment.py](./runtime-environment.py) to see how a game runs.

### Algorithms
#### Leader's Message
- Distance to the nearest obstacle (obs_dist): int or float
- Relative position of the goal (xg): int, -1 if goal is not in partial observability.
- Relative position of the goal (yg): int, -1 if goal is not in partial observability.
- Whether the path is clear or blocked(path_blocked): 0/1 int
- Leader's action (action): int
- Leader can observe the follower or not (follower_visibility): 0/1 int
- Leaders distance to follower (follower_dist): float
- Leader's suggested action in x direction (action_dx): int
- Leader's suggested action in y direction (action_dy): int
- Leader's current x position (x): int
- Leader's current y position (y): int
- Sample: [-1, 1, 1.0, 0, 0, 0, 2, 2]

#### Encoder Model - for the Leader agent

| Layer (type)       | Output Shape   | Param #  |
|---------------------|----------------|----------|
| input_layer  (InputLayer)        | (None, 8)    | 0        |
| reshape (Reshape)            | (None, 1, 8) | 0        |
| gru (GRU)                | (None, 1, 64) | 14,208   |
| gru_1 (GRU)              | (None, 32)    | 9,408   |

 * Total params: 23,616 (92.25 KB)
 * Trainable params: 23,616 (92.25 KB)
 * Non-trainable params: 0 (0.00 B)
 * Prediction: Outputs an array of 32 values, representing the encoded leader's message communicating to the follower agents.

 #### Decoder Model - for the Follower agent

| Layer (type)              | Output Shape   | Param #  |
|---------------------------|----------------|----------|
| input_layer_1 (InputLayer)| (None, 32)     | 0        |
| repeat_vector (RepeatVector)| (None, 1, 32)| 0        |
| gru_2 (GRU)             | (None, 1, 64) | 18,816   |
| gru_3 (GRU)             | (None, 64)    | 24,960   |
| dense (Dense)             | (None, 8)    | 520      |
 
 * Total params: 44,296 (173.03 KB)
 * Trainable params: 44,296 (173.03 KB)
 * Non-trainable params: 0 (0.00 B)
 * Prediction: Outputs an array of 8 values, representing the probabilities of each possible action. 

 #### Policy Network Models
 * Evaluates the best move for an agent.

 #### Leader's Policy Network

| Layer (type)            | Output Shape   | Param #  |
|--------------------------|----------------|----------|
| input_layer_27 (InputLayer)          | (None, 8)      | 0        |
| reshape_26 (Reshape)              | (None, 1, 8)   | 0        |
| dense_71 (Dense)                | (None, 1, 64)  | 576      |
| dense_72 (Dense)                | (None, 1, 64)  | 4,160    |
| dense_73 (Dense)                | (None, 1, 9)   | 585      |
| reshape_27 (Reshape)              | (None, 9)      | 0        |

* Total params: 5,321 (20.78 KB)
* Trainable params: 5,321 (20.78 KB)
* Non-trainable params: 0 (0.00 B)

#### Follower's Policy Network

| Layer (type)                  | Output Shape   | Param #  |
|-------------------------------|----------------|----------|
| input_layer_28 (InputLayer)   | (None, 2, 8)   | 0        |
| global_average_pooling1d_11   | (None, 8)      | 0        |
| dense_74 (Dense)              | (None, 64)     | 576      |
| dense_75 (Dense)              | (None, 64)     | 4,160    |
| dense_76 (Dense)              | (None, 9)      | 585      |

* Total params: 5,321 (20.78 KB)
* Trainable params: 5,321 (20.78 KB)
* Non-trainable params: 0 (0.00 B)
* Input of the model is a combination of the leader's message and its own observation on the grid. The leader's message is encoded and compressed into 8 values in an array.

 ## Evaluations
 * Run the [`evaluation.py`](./training/evaluation.py) script to plot nicely looking graphs based on metrics we recorded during training. 
 * On your terminal, change directory into `training/`, and then run `tensorboard --logdir=logs`. Open `http://localhost:6006/` or the port it opens to in order to view GPU consumption statistics.
 * Run unit tests: on your terminal, change directory into `tests` with `cd tests`. Then, run this command: `coverage run -m unittest discover`. 
 * Inspect Unit Tests at [./tests/htmlcov/](./tests/htmlcov/)

 ## Execution Guide
* [**TMUX**](tmux.md) for idling long executions