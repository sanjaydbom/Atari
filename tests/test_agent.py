import pytest
import numpy as np
import torch
from torch import nn, optim
import random
from unittest.mock import patch, MagicMock

# ===============================================================
# Mock Objects and Config for Standalone Testing
# In your actual project, you would import these from your files.
# ===============================================================
from src.model import AtariDQN
from src.priority_replay import PriorityReplay

# Mock config variables
ACTION_SPACE = [0, 1, 2, 3]
EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_REDUCTION = 1e-6
LR = 2.5e-4
ALPHA = 0.95
EPS = 0.01
MINI_BATCH_SIZE = 4 # Use a small batch size for tests
GAMMA = 0.99
CLIP_GRADIENT = True
MAX_GRADIENT = 10.0
FRAME_STACK_SIZE = 4
SCREEN_SIZE = 84
NETWORK_VALIDATION_FREQUENCY = 500
TAU = 0.001
'''
# Patch the real classes/configs with our mocks for the tests
@pytest.fixture(autouse=True)
def patch_imports(monkeypatch):
    """Automatically patches imports for all tests."""
    monkeypatch.setattr('src.model', AtariDQN)
    monkeypatch.setattr('src.priority_replay', PriorityReplay)
    monkeypatch.setattr('src.config', ACTION_SPACE)
    monkeypatch.setattr('src.config', EPSILON)
    monkeypatch.setattr('src.config', MIN_EPSILON)
    monkeypatch.setattr('src.config', EPSILON_REDUCTION)
    monkeypatch.setattr('src.config', LR)
    monkeypatch.setattr('src.config', ALPHA)
    monkeypatch.setattr('src.config', EPS)
    monkeypatch.setattr('src.config', MINI_BATCH_SIZE)
    monkeypatch.setattr('src.config', GAMMA)
    monkeypatch.setattr('src.config', CLIP_GRADIENT)
    monkeypatch.setattr('src.config', MAX_GRADIENT)
    monkeypatch.setattr('src.config', FRAME_STACK_SIZE)
    monkeypatch.setattr('src.config', SCREEN_SIZE)
    monkeypatch.setattr('src.config', NETWORK_VALIDATION_FREQUENCY)
    monkeypatch.setattr('src.config', TAU)'''

# ===============================================================
# Pytest Fixtures
# ===============================================================
@pytest.fixture
def agent():
    """Fixture to create a fresh Agent instance for each test."""
    # *** FIX: Import the Agent class *inside* the fixture. ***
    # This ensures that the import happens AFTER the monkeypatching has been applied,
    # so the Agent class will use our mock AtariDQN and PriorityReplay.
    from src.agent import Agent
    return Agent(mini_batch_size=4)

@pytest.fixture
def sample_state():
    """Fixture to create a sample state with correct dimensions."""
    return np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)

# ===============================================================
# Tests
# ===============================================================
def test_initialization(agent):
    """Test Agent initialization for Rainbow DQN specifications."""
    assert agent.device is not None
    assert isinstance(agent.model, AtariDQN)
    assert isinstance(agent.target_model, AtariDQN)
    assert agent.action_space == ACTION_SPACE
    # Test for distributional properties
    assert isinstance(agent.loss_fn, nn.KLDivLoss)
    assert agent.loss_fn.reduction == 'none' # Crucial for PER
    assert agent.bins.shape == (51,)
    assert agent.bins.min() == -10
    assert agent.bins.max() == 10
    # Test for PER properties
    assert isinstance(agent.memory, PriorityReplay)

def test_get_action_greedy(agent, sample_state):
    """Test get_action correctly selects the action with the highest expected Q-value."""
    # Ensure we are not taking a random action
    agent.epsilon = -1.0
    
    # This is the correct way to determine the best action from a distribution
    with torch.no_grad():
        state_tensor = torch.tensor(sample_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        log_dist = agent.model(state_tensor)
        dist = torch.exp(log_dist)
        q_values = (dist * agent.bins).sum(dim=2)
        expected_action = torch.argmax(q_values).item()

    # Get action from the agent's method
        action = agent.get_action(sample_state, evaluation=True)
    
    assert action == expected_action

def test_store_experience(agent, sample_state):
    """Test storing an experience in the prioritized replay buffer."""
    action = random.choice(agent.action_space)
    reward = 1.0
    next_state = np.random.rand(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    done = False
    td_error = 1.23 # Initial priority
    
    initial_len = len(agent.memory)
    agent.store_experience(sample_state, action, reward, next_state, done, td_error)
    assert len(agent.memory) == initial_len + 1
    
    # Check if the stored priority is correct
    assert agent.memory.tree[0] == td_error

def test_train_insufficient_memory(agent):
    """Test train method returns None when memory is insufficient."""
    assert len(agent.memory) < MINI_BATCH_SIZE
    assert agent.train() is None

def test_train_updates_model_and_priorities(agent, sample_state):
    """Test that a training step updates model weights and memory priorities."""
    # Populate memory
    for _ in range(MINI_BATCH_SIZE):
        agent.store_experience(
            state=sample_state,
            action=random.choice(ACTION_SPACE),
            reward=random.random(),
            next_state=sample_state,
            done=False,
            td_error=1.0
        )

    # Mock the method that updates priorities to check if it's called
    with patch.object(agent.memory, 'update_all_td', wraps=agent.memory.update_all_td) as mock_update_td:
        # Get model parameters before training
        params_before = [p.clone() for p in agent.model.parameters()]
        
        # Run a training step
        loss = agent.train()
        
        # 1. Check for valid loss
        assert isinstance(loss, float)
        assert loss >= 0

        # 2. Check that model parameters have changed
        params_after = list(agent.model.parameters())
        assert any(not torch.equal(p_before, p_after) for p_before, p_after in zip(params_before, params_after))

        # 3. Check that priority update was called correctly
        mock_update_td.assert_called_once()
        # Check that it was called with the correct number of priorities
        call_args, _ = mock_update_td.call_args
        indices, priorities = call_args
        assert len(indices) == MINI_BATCH_SIZE
        assert len(priorities) == MINI_BATCH_SIZE
        assert all(isinstance(p, float) for p in priorities)
