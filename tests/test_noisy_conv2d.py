import torch
import torch.nn.functional as F
import pytest
import math

# Assuming your NoisyConv2d class and fun function are in a file named 'Noisy_Conv2d.py'
# Make sure 'Noisy_Conv2d.py' is in the same directory or accessible in your Python path.
from src.Noisy_Conv2d import NoisyConv2d, fun


@pytest.fixture
def setup_noisy_conv2d_params():
    """
    Pytest fixture to set up common parameters for NoisyConv2d tests.
    This replaces the setUp method from unittest.
    """
    input_shape = 3  # e.g., RGB image
    num_kernels = 16 # e.g., 16 output features
    kernel_size = 3  # 3x3 kernel
    stride = 1
    std_init = 0.5
    batch_size = 2
    input_height = 10
    input_width = 10
    input_tensor = torch.randn(batch_size, input_shape, input_height, input_width)
    return {
        "input_shape": input_shape,
        "num_kernels": num_kernels,
        "kernel_size": kernel_size,
        "stride": stride,
        "std_init": std_init,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "input_tensor": input_tensor,
    }

def test_initialization(setup_noisy_conv2d_params):
    """
    Tests if the parameters (mu_k, sigma_k, mu_b, sigma_b) are
    correctly initialized with the expected shapes and initial values.
    """
    print("\n--- Running test_initialization ---")
    params = setup_noisy_conv2d_params
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        params["std_init"]
    )

    # Check shapes of the parameters
    assert layer.mu_k.shape == (params["num_kernels"], params["input_shape"], params["kernel_size"], params["kernel_size"])
    assert layer.sigma_k.shape == (params["num_kernels"], params["input_shape"], params["kernel_size"], params["kernel_size"])
    assert layer.mu_b.shape == (params["num_kernels"],)
    assert layer.sigma_b.shape == (params["num_kernels"],)

    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.mu_k)
    # Check initial values of sigma_k and sigma_b
    assert torch.all(layer.sigma_k == params["std_init"] / math.sqrt(fan_in))
    assert torch.all(layer.sigma_b == params["std_init"] / math.sqrt(fan_in))

    print("Initialization test passed: Parameter shapes and initial sigma values are correct.")

def test_output_shape(setup_noisy_conv2d_params):
    """
    Tests if the output shape of the NoisyConv2d layer is correct
    based on standard convolution output shape calculations.
    """
    print("\n--- Running test_output_shape ---")
    params = setup_noisy_conv2d_params
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"]
    )
    output = layer(params["input_tensor"])

    # Calculate expected output dimensions for convolution (no padding)
    expected_height = (params["input_height"] - params["kernel_size"]) // params["stride"] + 1
    expected_width = (params["input_width"] - params["kernel_size"]) // params["stride"] + 1
    expected_shape = (params["batch_size"], params["num_kernels"], expected_height, expected_width)

    assert output.shape == expected_shape
    print(f"Output shape test passed: Expected {expected_shape}, Got {output.shape}")

def test_noise_generation_and_application(setup_noisy_conv2d_params):
    """
    Tests if the noise is correctly generated and applied to weights and biases.
    This involves checking the presence of noise and verifying the formula.
    Since noise is random, we'll check that two consecutive calls produce different outputs.
    """
    print("\n--- Running test_noise_generation_and_application ---")
    params = setup_noisy_conv2d_params
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        params["std_init"]
    )

    # Run forward pass multiple times to observe randomness
    output1 = layer(params["input_tensor"])
    output2 = layer(params["input_tensor"])

    # Due to noise, outputs should be different.
    # We use a small tolerance (1e-6) for floating-point comparisons.
    assert not torch.allclose(output1, output2, atol=1e-6), "Outputs should be different due to noise."
    print("Noise generation and application test passed: Consecutive outputs are different.")

def test_fun_function():
    """
    Tests the 'fun' activation function based on its definition.
    fun(x) = x.sign() * x.abs().sqrt()
    """
    print("\n--- Running test_fun_function ---")
    test_tensor = torch.tensor([-4.0, 0.0, 9.0, -25.0, 0.5, -0.25])
    expected_output = torch.tensor([-2.0, 0.0, 3.0, -5.0, math.sqrt(0.5), -math.sqrt(0.25)])
    
    # Use torch.allclose for floating point comparison
    assert torch.allclose(fun(test_tensor), expected_output, atol=1e-7)
    print("fun function test passed: Correctly computes sign * sqrt(abs).")

def test_stride_effect(setup_noisy_conv2d_params):
    """
    Tests if changing the stride parameter correctly affects the output shape.
    """
    print("\n--- Running test_stride_effect ---")
    params = setup_noisy_conv2d_params
    
    # Test with stride = 2
    stride_val = 2
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        stride=stride_val
    )
    output = layer(params["input_tensor"])

    expected_height = (params["input_height"] - params["kernel_size"]) // stride_val + 1
    expected_width = (params["input_width"] - params["kernel_size"]) // stride_val + 1
    expected_shape = (params["batch_size"], params["num_kernels"], expected_height, expected_width)
    assert output.shape == expected_shape
    print(f"Stride effect test passed (stride={stride_val}): Expected {expected_shape}, Got {output.shape}")

    # Test with stride = 1 (already covered by output_shape, but good for explicit test)
    stride_val = 1
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        stride=stride_val
    )
    output = layer(params["input_tensor"])
    expected_height = (params["input_height"] - params["kernel_size"]) // stride_val + 1
    expected_width = (params["input_width"] - params["kernel_size"]) // stride_val + 1
    expected_shape = (params["batch_size"], params["num_kernels"], expected_height, expected_width)
    assert output.shape == expected_shape
    print(f"Stride effect test passed (stride={stride_val}): Expected {expected_shape}, Got {output.shape}")


def test_no_noise_when_sigma_zero(setup_noisy_conv2d_params):
    """
    Tests if the layer behaves like a standard Conv2d if sigma_k and sigma_b are zero.
    This is achieved by manually setting sigma_k and sigma_b to zero after initialization.
    """
    print("\n--- Running test_no_noise_when_sigma_zero ---")
    params = setup_noisy_conv2d_params
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        std_init=0.0 # Initialize with 0 std_init
    )

    # Manually ensure sigma parameters are zero (or very close to it)
    with torch.no_grad():
        layer.sigma_k.zero_()
        layer.sigma_b.zero_()

    # Now, run the forward pass multiple times. Outputs should be identical.
    output1 = layer(params["input_tensor"])
    output2 = layer(params["input_tensor"])

    # If sigma is zero, the noise terms should be zero, and outputs should be identical.
    assert torch.allclose(output1, output2, atol=1e-7), "Outputs should be identical if sigma is zero."

    # Compare with a standard Conv2d (if you want to be extra thorough)
    # Note: Requires same initialization if comparing exact values.
    # For this test, simply checking that with sigma=0, repeated calls give same output is sufficient.
    print("No noise when sigma is zero test passed: Consecutive outputs are identical.")

def test_device_consistency(setup_noisy_conv2d_params):
    """
    Tests if the module correctly handles inputs on different devices (CPU/CUDA).
    """
    print("\n--- Running test_device_consistency ---")
    params = setup_noisy_conv2d_params
    
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        params["std_init"]
    )

    # Test on CPU
    output_cpu = layer(params["input_tensor"].cpu())
    assert output_cpu.device.type == 'cpu'
    print("Device consistency test passed (CPU).")

    # Test on CUDA if available
    if torch.cuda.is_available():
        layer.to('cuda')
        input_cuda = params["input_tensor"].to('cuda')
        output_cuda = layer(input_cuda)
        assert output_cuda.device.type == 'cuda'
        print("Device consistency test passed (CUDA).")
    else:
        print("CUDA not available, skipping CUDA device consistency test.")

def test_gradients_flow(setup_noisy_conv2d_params):
    """
    Tests if gradients can flow through the noisy layer,
    which is essential for training.
    """
    print("\n--- Running test_gradients_flow ---")
    params = setup_noisy_conv2d_params
    
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        params["std_init"]
    )
    
    # Ensure input tensor requires gradients
    input_tensor_requires_grad = params["input_tensor"].clone().requires_grad_(True)

    output = layer(input_tensor_requires_grad)
    
    # Create a dummy loss
    loss = output.sum()
    
    # Perform backward pass
    loss.backward()
    
    # Check if gradients are computed for the parameters
    assert layer.mu_k.grad is not None
    assert layer.sigma_k.grad is not None
    assert layer.mu_b.grad is not None
    assert layer.sigma_b.grad is not None
    
    # Check if gradients are computed for the input
    assert input_tensor_requires_grad.grad is not None
    
    print("Gradients flow test passed: Gradients computed for parameters and input.")

def test_reproducibility_with_seed(setup_noisy_conv2d_params):
    """
    Tests if setting a PyTorch manual seed makes the noise reproducible,
    as it should for debugging and consistent experiments.
    """
    print("\n--- Running test_reproducibility_with_seed ---")
    params = setup_noisy_conv2d_params

    # First run with a seed
    torch.manual_seed(42)
    layer1 = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        params["std_init"]
    )
    output1 = layer1(params["input_tensor"])

    # Second run with the same seed
    torch.manual_seed(42)
    layer2 = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        params["std_init"]
    )
    output2 = layer2(params["input_tensor"])

    # Outputs should be identical if the seed provides full reproducibility
    assert torch.allclose(output1, output2, atol=1e-7), "Outputs should be reproducible with a fixed seed."
    print("Reproducibility test passed: Outputs are identical with the same seed.")

def test_bias_handling(setup_noisy_conv2d_params): # Corrected fixture name here
    """
    Tests if the bias term is correctly added.
    This can be checked by passing a zero input and verifying the output
    is just the bias (convolved with a 1-kernel if you imagine it).
    More directly, when the input and weights are such that the convolution output
    is zero, the result should be solely due to the bias.
    """
    print("\n--- Running test_bias_handling ---")
    params = setup_noisy_conv2d_params
    
    input_zero = torch.zeros_like(params["input_tensor"])
    
    layer = NoisyConv2d(
        params["input_shape"],
        params["num_kernels"],
        params["kernel_size"],
        params["stride"],
        std_init=0.0 # Set std_init to 0 to make noise deterministic
    )

    # Set weights to zero so only bias contributes (or almost zero)
    with torch.no_grad():
        layer.mu_k.zero_()
        layer.sigma_k.zero_() # Ensure no weight noise

    output = layer(input_zero)

    # For zero input and zero weights, the output feature maps should be
    # essentially filled with the noisy bias values.
    # The output shape will be (batch_size, num_kernels, H_out, W_out)
    # Each feature map should be constant and equal to the corresponding noisy_bias.
    
    # Recalculate the noisy_bias given the zeroed sigmas and a fixed noise state.
    # We need to simulate the forward pass for bias calculation when sigmas are 0.
    # In this specific case (std_init=0), noisy_bias will simply be mu_b.
    
    # Get the expected output bias for each channel
    # Since std_init is 0.0, layer.sigma_b will be all zeros.
    # Thus, noisy_bias will be exactly layer.mu_b.
    expected_noisy_bias = layer.mu_b

    for b in range(params["batch_size"]):
        for k in range(params["num_kernels"]):
            # Each spatial location in the output feature map for a given kernel
            # should be equal to the bias for that kernel, since input and weights are zero.
            assert torch.allclose(output[b, k, :, :], expected_noisy_bias[k], atol=1e-7)
    
    print("Bias handling test passed: Output correctly reflects bias when weights and input are zero.")


# You can run these tests using `pytest <your_test_file_name.py>` from your terminal.
# Make sure you have pytest installed: `pip install pytest`
