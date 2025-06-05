import pytest
import numpy as np
import torch
from torch import nn, optim
from collections import deque
import random
from src.agent import Agent, AtariDQN, ACTION_SPACE, EPSILON, MIN_EPSILON, EPSILON_REDUCTION, LR, ALPHA, EPS, MEMORY_SIZE, MINI_BATCH_SIZE, GAMMA, CLIP_GRADIENT, MAX_GRADIENT, FRAME_STACK_SIZE, SCREEN_SIZE, NETWORK_VALIDATION_FREQUENCY, TAU

@pytest.fixture
def agent():
    """Fixture to create a fresh Agent instance for each test."""
    return Agent(
        action_space=ACTION_SPACE,
        epsilon=EPSILON,
        min_epsilon=MIN_EPSILON,
        epsilon_reduction=EPSILON_REDUCTION,
        lr=LR,
        alpha=ALPHA,
        eps=EPS,
        gamma=GAMMA,
        mini_batch_size=MINI_BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        clip_gradient=CLIP_GRADIENT,
        max_gradient=MAX_GRADIENT,
        network_validation_frequency=NETWORK_VALIDATION_FREQUENCY,
        tau=TAU
    )

@pytest.fixture
def sample_state():
    """Fixture to create a sample state with correct dimensions."""
    return np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)

def test_initialization(agent):
    """Test Agent initialization with default parameters."""
    assert agent.device in [torch.device("cuda"), torch.device("mps"), torch.device("cpu")]
    assert isinstance(agent.model, AtariDQN)
    assert isinstance(agent.target_model, AtariDQN)
    assert agent.action_space == ACTION_SPACE
    assert agent.epsilon == EPSILON
    assert agent.min_epsilon == MIN_EPSILON
    assert agent.epsilon_reduction == EPSILON_REDUCTION
    assert isinstance(agent.optimizer, optim.RMSprop)
    assert isinstance(agent.loss_fn, nn.SmoothL1Loss)
    assert isinstance(agent.memory, deque)
    assert agent.memory.maxlen == MEMORY_SIZE
    assert agent.gamma == GAMMA
    assert agent.mini_batch_size == MINI_BATCH_SIZE
    assert agent.clip_gradient == CLIP_GRADIENT
    assert agent.max_gradient == MAX_GRADIENT
    assert agent.tau == TAU
    assert agent.input_shape == (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    assert agent.num_steps == 0
    assert agent.validation_frequency == NETWORK_VALIDATION_FREQUENCY

def test_get_action_random(agent, sample_state):
    """Test get_action method for random action selection when epsilon is high."""
    agent.epsilon = 1.0  # Force random action
    action = agent.get_action(sample_state)
    assert action in agent.action_space

def test_get_action_greedy(agent, sample_state):
    """Test get_action method for greedy action selection during evaluation."""
    action = agent.get_action(sample_state, evaluation=True)
    assert action in agent.action_space
    with torch.no_grad():
        state_tensor = torch.tensor(sample_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        expected_action = torch.argmax(agent.model(state_tensor)).item()
        assert action == expected_action

def test_get_action_invalid_state(agent):
    """Test get_action with invalid state dimensions raises ValueError."""
    invalid_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE + 1)
    with pytest.raises(ValueError, match=r"Expected input shape.*"):
        agent.get_action(invalid_state)

def test_update_epsilon(agent):
    """Test epsilon decay and minimum epsilon bound."""
    initial_epsilon = agent.epsilon
    agent.update_epsilon()
    assert agent.epsilon == max(initial_epsilon - agent.epsilon_reduction, agent.min_epsilon)
    agent.epsilon = agent.min_epsilon - 0.1
    agent.update_epsilon()
    assert agent.epsilon == agent.min_epsilon

def test_store_experience(agent, sample_state):
    """Test storing a valid experience in memory."""
    action = random.choice(agent.action_space)
    reward = 1.0
    next_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    done = False
    initial_len = len(agent.memory)
    agent.store_experience(sample_state, action, reward, next_state, done)
    assert len(agent.memory) == initial_len + 1
    stored = agent.memory[-1]
    assert np.array_equal(stored[0], sample_state)
    assert stored[1] == action
    assert stored[2] == reward
    assert np.array_equal(stored[3], next_state)

def test_store_experience_terminal(agent, sample_state):
    """Test storing a terminal state experience (next_state should be None)."""
    action = random.choice(agent.action_space)
    reward = -1.0
    next_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    done = True
    agent.store_experience(sample_state, action, reward, next_state, done)
    stored = agent.memory[-1]
    assert stored[3] is None

def test_store_experience_invalid_state(agent):
    """Test storing experience with invalid state dimensions raises ValueError."""
    invalid_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE + 1)
    action = random.choice(agent.action_space)
    reward = 1.0
    next_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    with pytest.raises(ValueError, match=r"Expected input shape.*"):
        agent.store_experience(invalid_state, action, reward, next_state, False)

def test_store_experience_invalid_action(agent, sample_state):
    """Test storing experience with invalid action raises ValueError."""
    action = max(agent.action_space) + 1  # Invalid action
    reward = 1.0
    next_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    with pytest.raises(ValueError, match=r"Action.*not in action space"):
        agent.store_experience(sample_state, action, reward, next_state, False)

def test_train_insufficient_memory(agent):
    """Test train method returns None when memory is insufficient."""
    assert len(agent.memory) == 0
    assert agent.train() is None

def test_train_with_memory(agent, sample_state):
    """Test train method with sufficient memory for a batch."""
    action = random.choice(agent.action_space)
    reward = 1.0
    next_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    for _ in range(MINI_BATCH_SIZE):
        done = random.choice([True, False])
        agent.store_experience(sample_state, action, reward, next_state, done)
    loss = agent.train()
    assert isinstance(loss, float)
    assert loss >= 0

def test_update_target_network(agent):
    """Test target network update using soft update with tau."""
    original_target_params = [p.clone() for p in agent.target_model.parameters()]
    original_model_params = [p.clone() for p in agent.model.parameters()]
    agent.update_target_network()
    for target_param, model_param, orig_target in zip(agent.target_model.parameters(), agent.model.parameters(), original_target_params):
        expected = agent.tau * model_param.data + (1 - agent.tau) * orig_target
        assert torch.allclose(target_param.data, expected)

def test_save_and_load_model(agent, tmp_path):
    """Test saving and loading model and optimizer state."""
    filename = tmp_path / "model.pt"
    agent.save_model(str(filename))
    assert filename.exists()
    optimizer_path = filename.with_name("model_optimizer.pt")
    assert optimizer_path.exists()
    
    # Modify model and optimizer to verify loading
    original_state = agent.model.state_dict()
    original_optimizer = agent.optimizer.state_dict()
    for param in agent.model.parameters():
        param.data.fill_(0)
    agent.optimizer.zero_grad()

    agent.load_model(str(filename))
    loaded_state = agent.model.state_dict()
    for key in original_state:
        assert torch.allclose(original_state[key], loaded_state[key])
    assert agent.optimizer.state_dict().keys() == original_optimizer.keys()

def test_increment_steps_and_validation(agent):
    """Test step increment and target network update on validation frequency."""
    initial_steps = agent.get_steps()
    for i in range(int(NETWORK_VALIDATION_FREQUENCY)):
        agent.increment_steps()
        assert agent.get_steps() == initial_steps + i + 1
        if (i + 1) == int(NETWORK_VALIDATION_FREQUENCY):
            original_target_params = [p.clone() for p in agent.target_model.parameters()]
            for target_param, model_param in zip(agent.target_model.parameters(), agent.model.parameters()):
                expected = agent.tau * model_param.data + (1 - agent.tau) * target_param.data
                assert torch.allclose(target_param.data, expected)