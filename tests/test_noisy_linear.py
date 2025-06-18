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

# Define a fixture for a common NoisyLinear instance
@pytest.fixture
def noisy_linear_layer():
    """Provides a default NoisyLinear layer for testing."""
    return NoisyLinear(input_size=10, output_size=5)

# Test 1: Initialization of parameters
def test_initialization(noisy_linear_layer):
    """
    Tests if the parameters (mu_w, sigma_w, mu_b, sigma_b) are
    correctly initialized as nn.Parameter and have the right shapes.
    """
    assert isinstance(noisy_linear_layer.mu_w, nn.Parameter)
    assert isinstance(noisy_linear_layer.sigma_w, nn.Parameter)
    assert isinstance(noisy_linear_layer.mu_b, nn.Parameter)
    assert isinstance(noisy_linear_layer.sigma_b, nn.Parameter)

    assert noisy_linear_layer.mu_w.shape == (5, 10)
    assert noisy_linear_layer.sigma_w.shape == (5, 10)
    assert noisy_linear_layer.mu_b.shape == (5,)
    assert noisy_linear_layer.sigma_b.shape == (5,)

    # Check if sigma values are initialized to a small constant (approx 0.5/sqrt(fan_in))
    # Due to fan_in calculation, this will be roughly 0.5/sqrt(10)
    expected_sigma_init = 0.5 / math.sqrt(10)
    assert torch.allclose(noisy_linear_layer.sigma_w, torch.full((5, 10), expected_sigma_init), atol=1e-6)
    assert torch.allclose(noisy_linear_layer.sigma_b, torch.full((5,), expected_sigma_init), atol=1e-6)


# Test 2: Output shape
def test_output_shape(noisy_linear_layer):
    """
    Tests if the output tensor has the expected shape for a given input.
    """
    input_tensor = torch.randn(1, 10) # Batch size 1, input_size 10
    output = noisy_linear_layer(input_tensor)
    assert output.shape == (1, 5) # Batch size 1, output_size 5

    input_tensor_batch = torch.randn(32, 10) # Batch size 32
    output_batch = noisy_linear_layer(input_tensor_batch)
    assert output_batch.shape == (32, 5)


# Test 3: Noise in training mode (stochasticity)
def test_training_mode_noise(noisy_linear_layer):
    """
    Tests that the output is different for multiple calls in training mode,
    indicating the presence of noise.
    """
    noisy_linear_layer.train() # Set to training mode
    input_tensor = torch.randn(1, 10)

    output1 = noisy_linear_layer(input_tensor)
    output2 = noisy_linear_layer(input_tensor)

    # Outputs should be different due to random noise
    assert not torch.allclose(output1, output2, atol=1e-6)


# Test 4: Determinism in evaluation mode
def test_eval_mode_determinism(noisy_linear_layer):
    """
    Tests that the output is identical for multiple calls in evaluation mode,
    as noise should be turned off.
    """
    noisy_linear_layer.eval() # Set to evaluation mode
    input_tensor = torch.randn(1, 10)

    with torch.no_grad(): # No need to track gradients in eval mode
        output1 = noisy_linear_layer(input_tensor)
        output2 = noisy_linear_layer(input_tensor)

    # Outputs should be identical in eval mode
    assert torch.allclose(output1, output2, atol=1e-6)


# Test 5: Gradients can be computed for parameters (trainability)
def test_parameter_trainability(noisy_linear_layer):
    """
    Tests that gradients can be computed for mu_w, sigma_w, mu_b, and sigma_b,
    confirming they are learnable.
    """
    noisy_linear_layer.train()
    input_tensor = torch.randn(1, 10)
    output = noisy_linear_layer(input_tensor)
    
    # Create a dummy loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Check if gradients exist for all parameters
    assert noisy_linear_layer.mu_w.grad is not None
    assert noisy_linear_layer.sigma_w.grad is not None
    assert noisy_linear_layer.mu_b.grad is not None
    assert noisy_linear_layer.sigma_b.grad is not None

    # Check if gradients are not all zeros (implying non-trivial computation)
    assert not torch.allclose(noisy_linear_layer.mu_w.grad, torch.zeros_like(noisy_linear_layer.mu_w.grad))
    assert not torch.allclose(noisy_linear_layer.sigma_w.grad, torch.zeros_like(noisy_linear_layer.sigma_w.grad))
    assert not torch.allclose(noisy_linear_layer.mu_b.grad, torch.zeros_like(noisy_linear_layer.mu_b.grad))
    assert not torch.allclose(noisy_linear_layer.sigma_b.grad, torch.zeros_like(noisy_linear_layer.sigma_b.grad))


# Test 6: Custom noise function
def test_custom_func():
    """
    Tests if the layer correctly uses a custom noise activation function.
    """
    def custom_noisy_func(x):
        return x * 2 # A simple custom function

    layer = NoisyLinear(input_size=5, output_size=2, func=custom_noisy_func)
    layer.train()
    input_tensor = torch.randn(1, 5)

    # To truly test the custom function, we could mock torch.randn
    # or inspect intermediate values, but a simpler check is to see if
    # the 'func' attribute is indeed our custom function.
    assert layer.func == custom_noisy_func

    # Further, check that noise is still applied (indirectly, by checking stochasticity)
    output1 = layer(input_tensor)
    output2 = layer(input_tensor)
    assert not torch.allclose(output1, output2, atol=1e-6)


# Test 7: Reset method re-initializes parameters
def test_reset_method(noisy_linear_layer):
    """
    Tests if the reset method changes the parameter values, indicating
    re-initialization.
    """
    # Store initial parameter values
    initial_mu_w = noisy_linear_layer.mu_w.clone().detach()
    initial_sigma_w = noisy_linear_layer.sigma_w.clone().detach()
    initial_mu_b = noisy_linear_layer.mu_b.clone().detach()
    initial_sigma_b = noisy_linear_layer.sigma_b.clone().detach()

    # Apply some dummy training steps to change parameters
    noisy_linear_layer.train()
    optimizer = torch.optim.SGD(noisy_linear_layer.parameters(), lr=0.01)
    for _ in range(5):
        input_tensor = torch.randn(1, 10)
        output = noisy_linear_layer(input_tensor)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check that parameters have changed after dummy training
    assert not torch.allclose(noisy_linear_layer.mu_w, initial_mu_w)
    assert not torch.allclose(noisy_linear_layer.sigma_w, initial_sigma_w)
    assert not torch.allclose(noisy_linear_layer.mu_b, initial_mu_b)
    assert not torch.allclose(noisy_linear_layer.sigma_b, initial_sigma_b)

    # Now call reset
    noisy_linear_layer.reset()

    # Verify that parameters are now different from the values after training
    # (and likely close to their original initialization values, though not identical due to randomness)
    # The key is they are no longer the 'trained' values
    assert not torch.allclose(noisy_linear_layer.mu_w, initial_mu_w) # New random init should be different from original
    
    # For sigma, they should return to the specific constant value
    expected_sigma_init = 0.5 / math.sqrt(10)
    assert torch.allclose(noisy_linear_layer.sigma_w, torch.full((5, 10), expected_sigma_init), atol=1e-6)
    assert torch.allclose(noisy_linear_layer.sigma_b, torch.full((5,), expected_sigma_init), atol=1e-6)
