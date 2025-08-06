import torch
import torch.nn as nn
import math
import pytest

# Assuming your NoisyLinear class is defined in a file named 'noisy_linear.module'
# For example, if your NoisyLinear class is in 'noisy_linear_layer.py', you would use:
# from noisy_linear_layer import NoisyLinear, fun
# Or, if it's directly in 'noisy_linear.py' as previously, you might just need:
# from noisy_linear import NoisyLinear, fun
# I will include a placeholder import. You will need to adjust this
# based on where your 'NoisyLinear' class is actually saved.
from src.Noisy_Linear import NoisyLinear, fun

# --- Pytest Test Cases ---


@pytest.fixture
def noisy_linear_layer():
    """Pytest fixture to create a NoisyLinear layer."""
    return NoisyLinear(10, 5)

def test_initialization(noisy_linear_layer):
    """
    Tests that the NoisyLinear layer is initialized correctly.
    """
    assert noisy_linear_layer.input_size == 10
    assert noisy_linear_layer.output_size == 5
    assert isinstance(noisy_linear_layer.weight_mu, nn.Parameter)
    assert isinstance(noisy_linear_layer.weight_sigma, nn.Parameter)
    assert isinstance(noisy_linear_layer.bias_mu, nn.Parameter)
    assert isinstance(noisy_linear_layer.bias_sigma, nn.Parameter)
    assert 'weight_epsilon' in noisy_linear_layer._buffers
    assert 'bias_epsilon' in noisy_linear_layer._buffers

def test_forward_pass_shape(noisy_linear_layer):
    """
    Tests the shape of the output of the forward pass.
    """
    input_tensor = torch.randn(3, 10) # Batch size of 3
    output = noisy_linear_layer(input_tensor)
    assert output.shape == (3, 5)

def test_forward_pass_training_vs_eval(noisy_linear_layer):
    """
    Tests that the output is different in training and evaluation modes.
    """
    input_tensor = torch.randn(1, 10)

    # Training mode
    noisy_linear_layer.train()
    output_train_1 = noisy_linear_layer(input_tensor)
    noisy_linear_layer.reset_noise() # Get new noise
    output_train_2 = noisy_linear_layer(input_tensor)

    # Evaluation mode
    noisy_linear_layer.eval()
    output_eval_1 = noisy_linear_layer(input_tensor)
    output_eval_2 = noisy_linear_layer(input_tensor)

    # In training mode, outputs should be different due to noise
    assert not torch.allclose(output_train_1, output_train_2)

    # In evaluation mode, outputs should be identical
    assert torch.allclose(output_eval_1, output_eval_2)

    # The training output should be different from the eval output
    assert not torch.allclose(output_train_1, output_eval_1)

def test_noise_disabled_when_sigma_is_zero(noisy_linear_layer):
    """
    Tests that noise has no effect when sigma parameters are zero, even in training mode.
    """
    input_tensor = torch.randn(1, 10)
    noisy_linear_layer.train()

    # Manually set sigma parameters to zero
    with torch.no_grad():
        noisy_linear_layer.weight_sigma.fill_(0.0)
        noisy_linear_layer.bias_sigma.fill_(0.0)

    # First forward pass
    output_train_1 = noisy_linear_layer(input_tensor)

    # Reset noise and do a second pass
    noisy_linear_layer.reset_noise()
    output_train_2 = noisy_linear_layer(input_tensor)

    # With zero sigma, outputs should be identical even in training mode with different noise
    assert torch.allclose(output_train_1, output_train_2)

    # The output should also be identical to the evaluation mode output
    noisy_linear_layer.eval()
    output_eval = noisy_linear_layer(input_tensor)
    assert torch.allclose(output_train_1, output_eval)

def test_reset_method(noisy_linear_layer):
    """
    Tests that the reset method changes the weights and biases.
    """
    # Store the original parameters
    original_weight_mu = noisy_linear_layer.weight_mu.clone()
    original_bias_mu = noisy_linear_layer.bias_mu.clone()
    original_weight_sigma = noisy_linear_layer.weight_sigma.clone()
    original_bias_sigma = noisy_linear_layer.bias_sigma.clone()

    noisy_linear_layer.reset()

    # Check that the parameters have been updated
    assert not torch.allclose(original_weight_mu, noisy_linear_layer.weight_mu)
    assert not torch.allclose(original_bias_mu, noisy_linear_layer.bias_mu)
    assert not torch.allclose(original_weight_sigma, noisy_linear_layer.weight_sigma)
    assert not torch.allclose(original_bias_sigma, noisy_linear_layer.bias_sigma)

def test_activation_function():
    """
    Tests that the activation function is applied correctly.
    """
    layer_with_relu = NoisyLinear(10, 5, func=torch.relu)
    input_tensor = torch.randn(3, 10)
    output = layer_with_relu(input_tensor)

    # Check if all values in the output are non-negative
    assert torch.all(output >= 0)

if __name__ == "__main__":
    pytest.main()
