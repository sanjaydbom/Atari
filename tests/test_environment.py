import pytest
import numpy as np
from src.environment import AtariBreakoutEnv
from src.config import FRAME_STACK_SIZE, SCREEN_SIZE, CLIP_REWARD, CLIP_BOUND, GAME_MODE, FRAMESKIP, GRAYSCALE, SCALE

@pytest.fixture(scope="module")
def env():
    """Create an AtariBreakoutEnv instance for testing.

    Yields an initialized environment and ensures it is closed after tests.
    """
    env = AtariBreakoutEnv(
        game_mode=GAME_MODE,
        reward_clipping=CLIP_REWARD,
        frame_skip=FRAMESKIP,
        screen_size=SCREEN_SIZE,
        stack_size=FRAME_STACK_SIZE,
        grayscale=GRAYSCALE,
        scale=SCALE
    )
    yield env
    env.close()

def test_initialization(env):
    """Test that the environment initializes correctly."""
    assert env is not None, "Environment failed to initialize"
    assert hasattr(env, 'env'), "Gymnasium environment not created"

def test_config_consistency():
    """Test that config.py constants are valid."""
    assert isinstance(GAME_MODE, str), "GAME_MODE must be a string"
    assert isinstance(FRAMESKIP, int) and FRAMESKIP > 0, "FRAMESKIP must be a positive integer"
    assert isinstance(SCREEN_SIZE, int) and SCREEN_SIZE > 0, "SCREEN_SIZE must be a positive integer"
    assert isinstance(FRAME_STACK_SIZE, int) and FRAME_STACK_SIZE > 0, "FRAME_STACK_SIZE must be a positive integer"
    assert isinstance(GRAYSCALE, bool), "GRAYSCALE must be a boolean"
    assert isinstance(SCALE, bool), "SCALE must be a boolean"
    assert isinstance(CLIP_REWARD, bool), "CLIP_REWARD must be a boolean"
    assert isinstance(CLIP_BOUND, (int, float)) and CLIP_BOUND > 0, "CLIP_BOUND must be a positive number"

def test_reset_state_shape(env):
    """Test that reset() returns a state with the correct shape."""
    state = env.reset()
    expected_shape = (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    assert isinstance(state, np.ndarray), "State is not a NumPy array"
    assert state.shape == expected_shape, f"State shape {state.shape} does not match expected {expected_shape}"
    assert state.dtype == np.uint8, f"State dtype {state.dtype} is not uint8"

def test_step_state_shape_and_rewards(env):
    """Test that step() returns correct state shape, clipped rewards, and done flag."""
    env.reset()
    action = 0  # NOOP
    next_state, reward, done = env.step(action)
    
    # Test state shape
    expected_shape = (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    assert isinstance(next_state, np.ndarray), "Next state is not a NumPy array"
    assert next_state.shape == expected_shape, f"Next state shape {next_state.shape} does not match expected {expected_shape}"
    
    # Test reward clipping
    if CLIP_REWARD:
        assert -CLIP_BOUND <= reward <= CLIP_BOUND, f"Reward {reward} is not clipped to [{(-CLIP_BOUND)}, {CLIP_BOUND}]"
    
    # Test done flag
    assert isinstance(done, bool), "Done flag is not a boolean"

def test_action_space(env):
    """Test that step() handles all valid actions from the action space."""
    env.reset()
    action_space = env.env.action_space
    for action in range(action_space.n):
        next_state, reward, done = env.step(action)
        assert isinstance(next_state, np.ndarray), f"Step with action {action} did not return valid state"
        assert isinstance(reward, float), f"Reward for action {action} is not a float"
        assert isinstance(done, bool), f"Done flag for action {action} is not a boolean"

def test_invalid_action(env):
    """Test that step() raises an error for invalid actions."""
    env.reset()
    invalid_action = env.env.action_space.n  # One beyond valid actions
    with pytest.raises(Exception):  # Gymnasium typically raises ValueError or AssertionError
        env.step(invalid_action)

def test_close(env):
    """Test that close() executes without errors."""
    env.close()
