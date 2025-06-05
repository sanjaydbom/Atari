import pytest
import torch
import numpy as np
from collections import deque
from unittest.mock import Mock
from src.agent import Agent

# Mock configuration values for testing
@pytest.fixture
def config():
    return {
        'ACTION_SPACE': [0, 1, 2, 3],
        'EPSILON': 1.0,
        'MIN_EPSILON': 0.1,
        'EPSILON_REDUCTION': 0.1,
        'LR': 2.5e-4,
        'ALPHA': 0.95,
        'EPS': 1e-6,
        'GAMMA': 0.99,
        'MINI_BATCH_SIZE': 2,
        'MEMORY_SIZE': 10,
        'CLIP_GRADIENT': True,
        'MAX_GRADIENT': 10.0,
        'FRAME_STACK_SIZE': 4,
        'SCREEN_SIZE': 84,
        'NETWORK_VALIDATION_FREQUENCY': 5,
        'TAU': 0.001
    }

# Fixture to create an Agent instance
@pytest.fixture
def agent(config):
    class MockAtariDQN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(config['FRAME_STACK_SIZE'] * config['SCREEN_SIZE'] * config['SCREEN_SIZE'], len(config['ACTION_SPACE']))
        
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten input
            return self.fc(x)
    
    # Mock the AtariDQN model
    original_model = Agent.__dict__['model']
    Agent.model = MockAtariDQN()
    Agent.target_model = MockAtariDQN()
    
    agent = Agent(
        action_space=config['ACTION_SPACE'],
        epsilon=config['EPSILON'],
        min_epsilon=config['MIN_EPSILON'],
        epsilon_reduction=config['EPSILON_REDUCTION'],
        lr=config['LR'],
        alpha=config['ALPHA'],
        eps=config['EPS'],
        gamma=config['GAMMA'],
        mini_batch_size=config['MINI_BATCH_SIZE'],
        memory_size=config['MEMORY_SIZE'],
        clip_gradient=config['CLIP_GRADIENT'],
        max_gradient=config['MAX_GRADIENT'],
        network_validation_frequency=config['NETWORK_VALIDATION_FREQUENCY'],
        tau=config['TAU']
    )
    
    # Restore original model
    Agent.model = original_model
    Agent.target_model = original_model
    return agent

# Test device selection
def test_device_selection(config):
    agent = Agent(
        action_space=config['ACTION_SPACE'],
        epsilon=config['EPSILON'],
        min_epsilon=config['MIN_EPSILON'],
        epsilon_reduction=config['EPSILON_REDUCTION'],
        lr=config['LR'],
        alpha=config['ALPHA'],
        eps=config['EPS'],
        gamma=config['GAMMA'],
        mini_batch_size=config['MINI_BATCH_SIZE'],
        memory_size=config['MEMORY_SIZE'],
        clip_gradient=config['CLIP_GRADIENT'],
        max_gradient=config['MAX_GRADIENT'],
        network_validation_frequency=config['NETWORK_VALIDATION_FREQUENCY'],
        tau=config['TAU']
    )
    assert agent.device.type in ['cuda', 'mps', 'cpu']

# Test get_action with invalid state shape
def test_get_action_invalid_shape(agent, config):
    invalid_state = np.zeros((2, config['SCREEN_SIZE'], config['SCREEN_SIZE']))
    with pytest.raises(ValueError, match=f"Expected input shape {agent.input_shape}, got {invalid_state.shape}"):
        agent.get_action(invalid_state)

# Test get_action in exploration mode
def test_get_action_exploration(agent, config):
    state = np.zeros(agent.input_shape, dtype=np.float32)
    agent.epsilon = 1.0  # Force exploration
    action = agent.get_action(state)
    assert action in config['ACTION_SPACE']

# Test get_action in exploitation mode
def test_get_action_exploitation(agent, config):
    state = np.zeros(agent.input_shape, dtype=np.float32)
    agent.epsilon = 0.0  # Force exploitation
    action = agent.get_action(state)
    assert action in config['ACTION_SPACE']
    
    # Check evaluation mode (no exploration)
    action_eval = agent.get_action(state, evaluation=True)
    assert action_eval in config['ACTION_SPACE']

# Test update_epsilon
def test_update_epsilon(agent, config):
    initial_epsilon = agent.epsilon
    agent.update_epsilon()
    assert agent.epsilon == max(initial_epsilon - config['EPSILON_REDUCTION'], config['MIN_EPSILON'])
    agent.epsilon = config['MIN_EPSILON'] - 0.05
    agent.update_epsilon()
    assert agent.epsilon == config['MIN_EPSILON']

# Test store_experience with valid inputs
def test_store_experience_valid(agent, config):
    state = np.zeros(agent.input_shape, dtype=np.float32)
    action = 1
    reward = 1.0
    next_state = np.ones(agent.input_shape, dtype=np.float32)
    done = False
    agent.store_experience(state, action, reward, next_state, done)
    assert len(agent.memory) == 1
    stored = agent.memory[-1]
    assert np.array_equal(stored[0], state)
    assert stored[1] == action
    assert stored[2] == reward
    assert np.array_equal(stored[3], next_state)

# Test store_experience with terminal state
def test_store_experience_terminal(agent, config):
    state = np.zeros(agent.input_shape, dtype=np.float32)
    action = 1
    reward = 1.0
    next_state = np.ones(agent.input_shape, dtype=np.float32)
    done = True
    agent.store_experience(state, action, reward, next_state, done)
    assert len(agent.memory) == 1
    stored = agent.memory[-1]
    assert stored[3] is None

# Test store_experience with invalid inputs
def test_store_experience_invalid(agent, config):
    invalid_state = np.zeros((2, config['SCREEN_SIZE'], config['SCREEN_SIZE']))
    action = 1
    reward = 1.0
    next_state = np.ones(agent.input_shape, dtype=np.float32)
    with pytest.raises(ValueError, match=f"Expected input shape {agent.input_shape}, got {invalid_state.shape}"):
        agent.store_experience(invalid_state, action, reward, next_state, False)
    
    state = np.zeros(agent.input_shape, dtype=np.float32)
    invalid_action = 999
    with pytest.raises(ValueError, match=f"Action {invalid_action} not in action space"):
        agent.store_experience(state, invalid_action, reward, next_state, False)

# Test train with insufficient memory
def test_train_insufficient_memory(agent, config):
    assert agent.train() is None

# Test train with sufficient memory
def test_train_sufficient_memory(agent, config):
    state = np.zeros(agent.input_shape, dtype=np.float32)
    action = 1
    reward = 1.0
    next_state = np.ones(agent.input_shape, dtype=np.float32)
    done = False
    for _ in range(config['MINI_BATCH_SIZE']):
        agent.store_experience(state, action, reward, next_state, done)
    
    loss = agent.train()
    assert isinstance(loss, float)
    assert loss >= 0

# Test update_target_network
def test_update_target_network(agent, config):
    initial_params = [p.clone() for p in agent.target_model.parameters()]
    agent.update_target_network()
    for init_p, target_p, model_p in zip(initial_params, agent.target_model.parameters(), agent.model.parameters()):
        expected = config['TAU'] * model_p.data + (1 - config['TAU']) * init_p.data
        assert torch.allclose(target_p.data, expected)

# Test increment_steps and target network update
def test_increment_steps(agent, config):
    initial_steps = agent.get_steps()
    agent.increment_steps()
    assert agent.get_steps() == initial_steps + 1
    
    agent.num_steps = config['NETWORK_VALIDATION_FREQUENCY'] - 1
    initial_params = [p.clone() for p in agent.target_model.parameters()]
    agent.increment_steps()
    for init_p, target_p, model_p in zip(initial_params, agent.target_model.parameters(), agent.model.parameters()):
        expected = config['TAU'] * model_p.data + (1 - config['TAU']) * init_p.data
        assert torch.allclose(target_p.data, expected)

# Test save and load model
def test_save_load_model(agent, tmp_path):
    filename = tmp_path / "model.pt"
    agent.save_model(filename)
    assert filename.exists()
    assert (filename.with_name("model_optimizer.pt")).exists()
    
    # Modify model and optimizer, then load back
    original_params = [p.clone() for p in agent.model.parameters()]
    original_opt_state = agent.optimizer.state_dict().copy()
    for param in agent.model.parameters():
        param.data += 1.0
    agent.optimizer.zero_grad()
    
    agent.load_model(filename)
    for orig_p, loaded_p in zip(original_params, agent.model.parameters()):
        assert torch.allclose(orig_p, loaded_p)
    assert agent.optimizer.state_dict()['param_groups'][0]['lr'] == original_opt_state['param_groups'][0]['lr']