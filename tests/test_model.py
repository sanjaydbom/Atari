# test_model.py
#
# A comprehensive test suite for the AtariDQN model using pytest.
#
# To run these tests, you need to have pytest and torch installed:
# pip install pytest torch
#
# Then, simply run pytest from your terminal in the same directory as this file:
# pytest

import pytest
import torch

from src.config import FRAME_STACK_SIZE, SCREEN_SIZE
import torch.nn as nn
import math

# =============================================================================
# Helper function and Model Class Definitions
# (Copied from your files to make this test suite self-contained)
# =============================================================================

import torch
import math
from torch import nn

def fun(x):
    return x.sign().mul(x.abs().sqrt())

class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, func = None):
        super(NoisyLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mu_w = nn.Parameter(torch.empty(output_size, input_size))
        self.sigma_w = nn.Parameter(torch.empty(output_size, input_size))

        self.mu_b = nn.Parameter(torch.empty(output_size))
        self.sigma_b = nn.Parameter(torch.empty(output_size))

        if func is None:
            self.func = fun
        else:
            self.func = func

        self.reset()

    def reset(self):
        nn.init.kaiming_uniform_(self.mu_w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.mu_b, -bound, bound)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_w)
        sigma_init_const = 0.5 / math.sqrt(fan_in)

        nn.init.constant_(self.sigma_w, sigma_init_const)
        nn.init.constant_(self.sigma_b, sigma_init_const)

    def forward(self, x):
        if self.training:
            input_epsilon = torch.randn(self.input_size).to(x.device)
            output_epsilon = torch.randn(self.output_size).to(x.device)

            f_out = self.func(output_epsilon)
            f_in = self.func(input_epsilon)

            self.epsilon_w = f_out.outer(f_in)
            self.epsilon_b = f_out
            weight = self.mu_w + self.sigma_w * self.epsilon_w
            bias = self.mu_b + self.sigma_b * self.epsilon_b
        else:
            weight = self.mu_w.to(x.device)
            bias = self.mu_b.to(x.device)
        
        return nn.functional.linear(x,weight,bias)


class NoisyConv2d(nn.Module):
    def __init__(self, input_shape, num_kernels, kernel_size, stride=1, std_init=0.5):
        super(NoisyConv2d,self).__init__()
        self.input_shape = input_shape
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.std_init=std_init
        self.mu_k = nn.Parameter(torch.empty(num_kernels, input_shape, kernel_size, kernel_size))
        self.sigma_k = nn.Parameter(torch.empty(num_kernels, input_shape, kernel_size, kernel_size))

        self.mu_b = nn.Parameter(torch.empty(num_kernels))
        self.sigma_b = nn.Parameter(torch.empty(num_kernels))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.mu_k, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_k)
        bound = 1 / fan_in ** 0.5
        nn.init.uniform_(self.mu_b, -bound, bound)
        nn.init.constant_(self.sigma_k, self.std_init)
        nn.init.constant_(self.sigma_b, self.std_init)


    def forward(self, x:torch.tensor):
        if self.training:
            epsilon_in = torch.randn(1,self.input_shape,1,1).to(x.device)
            epsilon_out = torch.randn(self.num_kernels,1,1,1).to(x.device)

            f_in = fun(epsilon_in)
            f_out = fun(epsilon_out)

            epsilon_kernel = (f_out * f_in).expand_as(self.mu_k)

            epsilon_bias = f_out.squeeze()

            noisy_weights = self.mu_k + self.sigma_k * epsilon_kernel
            noisy_bias = self.mu_b + self.sigma_b * epsilon_bias
        else:
            noisy_weights = self.mu_k
            noisy_bias = self.mu_b

        return nn.functional.conv2d(x,noisy_weights,noisy_bias,self.stride)

        

class AtariDQN(nn.Module):
    """Dueling Convolution Neural Network for Deep Q-Learning on Atari Games
    
    Processes stacked frames from AtariBreakoutEnv and outputs Q-values for each action.
    Architecture follows Wang et al., 2016.

    Args:
        input_shape (tuple): Shape of input state (stack_size, height, width).
        num_actions (int): Number of actions in the environment's action space.

    Attributes:
        conv (nn.Sequential): Convolutional layers for feature extraction.
        fc (nn.Sequential): Fully connected layers.
        state_layer(nn.Linear): Fully connected layer to compute state value
        advantage_layer(nn.Linear): Fully connected layer to compute action advantages

    Notes:
        See https://arxiv.org/pdf/1511.06581 for more information
    
    """
    def __init__(self, input_shape: tuple = (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE), num_actions: int = 4):
        super(AtariDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            NoisyConv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            NoisyConv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            NoisyConv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU()
        )
        self.state_layer = nn.Linear(512,1)
        self.advantage_layer = nn.Linear(512,num_actions)

    def _get_conv_out(self, shape: tuple) -> int:
        """Calculate the output size of the convolutional layers.

        Args:
            shape (tuple): Input shape (stack_size, height, width).

        Returns:
            int: Number of elements in the flattened conv output.
        """
        
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))
    


    def forward(self, x):
        """Forward pass through network
        
        Normalizes input pixel values and processes through convolutional and fully connected layers
        to produce Q-values for each action.

        Args:
            x (torch.Tensor): Input state of shape (batch_size, stack_size, height, width).

        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, num_actions).

        Raises:
            ValueError: If input shape does not match expected input_shape.        
        """
        if x.shape[1:] != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {x.shape[1:]}")
        
        x = x / 255.0
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        state = self.state_layer(x)
        action = self.advantage_layer(x)
        state_action = state + action - torch.mean(action)
        return state_action


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def model_params():
    """Provides common model parameters for tests."""
    return {
        "input_shape": (4, 84, 84), # (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
        "num_actions": 4
    }

@pytest.fixture
def dqn_model(model_params):
    """Provides an instance of the AtariDQN model for testing."""
    return AtariDQN(**model_params)

# =============================================================================
# Test Cases
# =============================================================================

def test_model_creation(dqn_model, model_params):
    """Tests if the model is created successfully with the correct parameters."""
    assert dqn_model is not None
    assert dqn_model.input_shape == model_params["input_shape"]
    assert dqn_model.num_actions == model_params["num_actions"]
    print("\n✓ Model Creation Test Passed")

def test_forward_pass_output_shape(dqn_model, model_params):
    """Ensures the forward pass produces an output tensor of the correct shape."""
    batch_size = 16
    dummy_input = torch.rand(batch_size, *model_params["input_shape"])
    
    output = dqn_model(dummy_input)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, model_params["num_actions"])
    print("✓ Forward Pass Output Shape Test Passed")

def test_determinism_in_eval_mode(dqn_model, model_params):
    """
    Hard Test: Verifies that the model is deterministic in evaluation mode.
    This is critical for Noisy Nets, which should not use noise during inference.
    """
    dqn_model.eval()  # Set model to evaluation mode
    
    dummy_input = torch.rand(1, *model_params["input_shape"])
    
    # Run the same input through the model twice
    output1 = dqn_model(dummy_input)
    output2 = dqn_model(dummy_input)
    
    assert torch.equal(output1, output2), "Model outputs should be identical in eval mode."
    print("✓ Determinism in Eval Mode Test Passed")

def test_stochasticity_in_train_mode(dqn_model, model_params):
    """
    Hard Test: Verifies that the model is stochastic in training mode.
    This confirms that the Noisy Net layers are active and applying noise.
    """
    dqn_model.train()  # Set model to training mode
    
    dummy_input = torch.rand(1, *model_params["input_shape"])
    
    # Run the same input through the model twice
    output1 = dqn_model(dummy_input)
    output2 = dqn_model(dummy_input)
    
    assert not torch.equal(output1, output2), "Model outputs should be different in train mode."
    print("✓ Stochasticity in Train Mode Test Passed")

def test_input_shape_validation(dqn_model, model_params):
    """Edge Case: Ensures the model raises a ValueError for incorrect input shapes."""
    correct_shape = model_params["input_shape"]
    
    # Incorrect number of channels
    wrong_shape_input = torch.rand(1, correct_shape[0] - 1, *correct_shape[1:])
    with pytest.raises(ValueError, match="Expected input shape"):
        dqn_model(wrong_shape_input)
        
    # Incorrect height/width
    wrong_shape_input = torch.rand(1, *correct_shape[0:2], correct_shape[2] - 1)
    with pytest.raises(ValueError, match="Expected input shape"):
        dqn_model(wrong_shape_input)
    print("✓ Input Shape Validation Test Passed")

def test_conv_out_calculation(dqn_model, model_params):
    """Tests the internal _get_conv_out helper method for correctness."""
    # Manually calculate the expected output size after convolutions
    # Input: (B, C, 84, 84)
    # Conv1: k=8, s=4 -> floor((84 - 8) / 4) + 1 = 19 + 1 = 20 -> (B, 32, 20, 20)
    # Conv2: k=4, s=2 -> floor((20 - 4) / 2) + 1 = 8 + 1 = 9  -> (B, 64, 9, 9)
    # Conv3: k=3, s=1 -> floor((9 - 3) / 1) + 1 = 6 + 1 = 7  -> (B, 64, 7, 7)
    # Flattened size = 64 * 7 * 7 = 3136
    expected_size = 3136
    
    calculated_size = dqn_model._get_conv_out(model_params["input_shape"])
    assert calculated_size == expected_size
    print("✓ Convolutional Output Size Calculation Test Passed")

@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available for testing")
def test_device_movement(dqn_model, model_params):
    """Edge Case: Tests if the model and data can be moved to a GPU device."""
    device = torch.device("mps")
    model = dqn_model.to(device)
    dummy_input = torch.rand(4, *model_params["input_shape"]).to(device)
    
    try:
        output = model(dummy_input)
        assert output.device.type == "mps"
    except Exception as e:
        pytest.fail(f"Model failed on MPS device: {e}")
    print("✓ Device Movement (MPS) Test Passed")

def test_input_data_types(dqn_model, model_params):
    """Edge Case: Tests if the model handles different input data types like uint8."""
    # uint8 is a common format for screen captures
    uint8_input = torch.randint(0, 256, (4, *model_params["input_shape"]), dtype=torch.uint8)
    
    try:
        # The model should internally normalize this to a float tensor
        output = dqn_model(uint8_input)
        assert output.dtype == torch.float32
    except RuntimeError as e:
        pytest.fail(f"Model failed to process uint8 input: {e}")
    print("✓ Input Data Types (uint8) Test Passed")

def test_zero_batch_size(dqn_model, model_params):
    """Edge Case: Tests if the model can handle an input with a batch size of 0."""
    zero_batch_input = torch.empty(0, *model_params["input_shape"])
    
    output = dqn_model(zero_batch_input)
    
    assert output.shape == (0, model_params["num_actions"])
    print("✓ Zero Batch Size Test Passed")

def test_noisy_layer_gradients(dqn_model, model_params):
    """
    Hard Test: Verifies that the noise parameters (sigma) receive gradients during backpropagation.
    This confirms they are part of the computation graph and are being trained.
    """
    dqn_model.train()
    optimizer = torch.optim.Adam(dqn_model.parameters(), lr=1e-4)
    
    dummy_input = torch.rand(2, *model_params["input_shape"])
    output = dqn_model(dummy_input)
    loss = output.mean()
    
    optimizer.zero_grad()
    loss.backward()
    
    found_sigma_grad = False
    for name, param in dqn_model.named_parameters():
        if "sigma" in name:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.sum(torch.abs(param.grad)) > 0, f"Gradient for {name} is zero"
            found_sigma_grad = True
    
    assert found_sigma_grad, "No sigma parameters were found or they did not receive gradients."
    print("✓ Noisy Layer Gradient Test Passed")
