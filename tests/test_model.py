import torch
import pytest
from src.model import AtariDQN

# Constants for the tests
FRAME_STACK_SIZE = 4
SCREEN_SIZE = 84
NUM_ACTIONS = 4

# --- Pytest Fixtures ---

@pytest.fixture
def dummy_model():
    """Provides a dummy AtariDQN model for testing."""
    return AtariDQN(
        input_shape=(FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE),
        num_actions=NUM_ACTIONS
    )

@pytest.fixture
def sample_input():
    """Provides a sample input tensor for the model."""
    return torch.randn(1, FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)

@pytest.fixture
def sample_batch_input():
    """Provides a batch of sample input tensors."""
    batch_size = 8
    return torch.randn(batch_size, FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)

# --- Test Cases ---

def test_model_initialization(dummy_model):
    """
    Tests if the AtariDQN model is initialized with the correct architecture.
    """
    assert isinstance(dummy_model, AtariDQN), "Model is not an instance of AtariDQN"
    
    # Check for convolutional layers
    assert hasattr(dummy_model, 'conv'), "Model does not have convolutional layers"
    assert len(dummy_model.conv) == 6, "Incorrect number of convolutional layers or sub-modules"
    
    # Check for fully connected layers
    assert hasattr(dummy_model, 'fc'), "Model does not have fully connected layers"
    assert len(dummy_model.fc) == 2, "Incorrect number of fully connected layers or sub-modules"
    
    # Check for dueling streams
    assert hasattr(dummy_model, 'state_layer'), "Model does not have a state_layer"
    assert hasattr(dummy_model, 'advantage_layer'), "Model does not have an advantage_layer"

def test_forward_pass(dummy_model, sample_input):
    """
    Tests the forward pass of the model to ensure it produces the correct output shape.
    """
    output = dummy_model(sample_input)
    
    assert output.shape == (1, NUM_ACTIONS), f"Expected output shape (1, {NUM_ACTIONS}), but got {output.shape}"

def test_input_validation(dummy_model):
    """
    Tests that the model raises a ValueError for incorrect input shapes.
    """
    # Create an input with an incorrect shape
    incorrect_input = torch.randn(1, FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE + 1)
    
    with pytest.raises(ValueError, match=r"Expected input shape .* got .*"):
        dummy_model(incorrect_input)

def test_batch_processing(dummy_model, sample_batch_input):
    """
    Tests the model's ability to process a batch of inputs.
    """
    batch_size = sample_batch_input.shape[0]
    output = dummy_model(sample_batch_input)
    
    assert output.shape == (batch_size, NUM_ACTIONS), f"Expected output shape ({batch_size}, {NUM_ACTIONS}), but got {output.shape}"

def test_data_type_and_range(dummy_model, sample_input):
    """
    Tests that the model handles different data types and normalizes pixel values correctly.
    """
    # Test with integer input that mimics pixel values
    int_input = (sample_input.clamp(0, 1) * 255).to(torch.uint8)
    output = dummy_model(int_input.float())  # Convert to float for the model
    
    assert output.dtype == torch.float32, f"Expected output dtype float32, but got {output.dtype}"

def test_dueling_architecture_aggregation(dummy_model, sample_input):
    """
    Tests if the dueling architecture aggregation is correctly implemented.
    """
    # Manually perform the forward pass to inspect intermediate values
    x = sample_input / 255.0  # Normalization
    conv_out = dummy_model.conv(x)
    flattened = torch.flatten(conv_out, start_dim=1)
    fc_out = dummy_model.fc(flattened)
    
    state_value = dummy_model.state_layer(fc_out)
    advantage_values = dummy_model.advantage_layer(fc_out)
    
    # Calculate the expected Q-values using the aggregation formula from the paper
    expected_q_values = state_value + (advantage_values - advantage_values.mean(dim=1, keepdim=True))
    
    # Get the actual Q-values from the model's forward pass
    actual_q_values = dummy_model(sample_input)
    
    # Assert that the manually calculated values are close to the model's output
    assert torch.allclose(actual_q_values, expected_q_values, atol=1e-6), "Dueling architecture aggregation calculation is incorrect"

