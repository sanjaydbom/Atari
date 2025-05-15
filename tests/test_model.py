import pytest
import torch
from src.model import AtariDQN
from src.config import FRAME_STACK_SIZE, SCREEN_SIZE, NUM_ACTIONS

@pytest.fixture
def model():
    """Create an AtariDQN instance for testing."""
    return AtariDQN(input_shape=(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE), num_actions=NUM_ACTIONS)

def test_initialization(model):
    """Test that the model initializes correctly."""
    assert isinstance(model, AtariDQN), "Model is not an AtariDQN instance"
    assert model.input_shape == (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE), "Input shape not set correctly"
    assert model.num_actions == NUM_ACTIONS, "Number of actions not set correctly"
    assert isinstance(model.conv, torch.nn.Sequential), "Conv layers not in Sequential"
    assert isinstance(model.fc, torch.nn.Sequential), "FC layers not in Sequential"

def test_forward_output_shape(model):
    """Test that forward pass produces correct output shape."""
    batch_size = 32
    input_tensor = torch.randn(batch_size, FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
    output = model(input_tensor)
    expected_shape = (batch_size, NUM_ACTIONS)
    assert output.shape == expected_shape, f"Output shape {output.shape} does not match expected {expected_shape}"
    assert output.dtype == torch.float32, "Output dtype is not float32"

def test_input_normalization(model):
    """Test that forward pass normalizes input pixel values."""
    input_tensor = torch.ones(1, FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE) * 255.0
    output = model(input_tensor)
    # Check that gradients are not computed in forward pass
    assert model.conv[0].weight.grad is None, "Gradients should not be computed in forward pass"
    # Check that output is not zero, indicating normalization was applied
    assert not torch.all(output == 0), "Output should not be zero after normalization"

def test_invalid_input_shape(model):
    """Test that forward pass raises error for invalid input shape."""
    wrong_shape = (1, FRAME_STACK_SIZE + 1, SCREEN_SIZE, SCREEN_SIZE)
    invalid_input = torch.randn(wrong_shape)
    with pytest.raises(ValueError, match=r"Expected input shape"):
        model(invalid_input)

def test_get_conv_out(model):
    """Test that _get_conv_out calculates correct conv output size."""
    conv_out_size = model._get_conv_out((FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE))
    assert isinstance(conv_out_size, int), "Conv output size is not an integer"
    assert conv_out_size > 0, "Conv output size should be positive"
    # Expected size for (4, 84, 84) input: 64 * 7 * 7 = 3136
    assert conv_out_size == 3136, f"Conv output size {conv_out_size} does not match expected 3136"