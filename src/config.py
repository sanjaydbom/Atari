GAME_MODE = "ALE/Breakout-v5"
ACTION_SPACE = [0,1,2,3]
NUM_ACTIONS = len(ACTION_SPACE)

#Parameters of the environment(environment.py)
FRAMESKIP = 4
FRAME_STACK_SIZE = 4
SCREEN_SIZE = 84

GRAYSCALE = True
SCALE = True

CLIP_REWARD = True
CLIP_BOUND = 1

#Parameters of the agent(agent.py)

GAMMA = 0.99
#RMSProp parameters
LR = 6.25e-5
EPS = 1.5e-4

MINI_BATCH_SIZE = 32

MEMORY_SIZE = int(1e6)
CLIP_GRADIENT = True
MAX_GRADIENT = 10.0

NETWORK_UPDATE_FREQUENCY = 1e4

NUM_BINS = 51
MAX_BIN_VALUE = 10
MIN_BIN_VALUE = -10


#Training parameters(train.py)
MAX_STEPS = 1e7
TRAINING_START_STEP = 8e4

#Testing parameters(test.py)
NUM_TESTS = 10