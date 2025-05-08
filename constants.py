from enum import Enum

class ACTION_SPACE(Enum):
    # x: row, y: column
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (-1, 1)
    DOWN_LEFT = (1, -1)
    DOWN_RIGHT = (1, 1)
    STAY = (0, 0)

class REWARDS(Enum):
    SOFT_OBSTACLE = -10
    HARD_OBSTACLE = -50
    WALL = -47
    TARGET = 50
    STEP = -1
    CRASH = OUT_OF_TETHER = -50
    STAY = -3

LEADER_MESSAGE_SIZE = 8
TETHER_TOLERATE_COUNT = 5