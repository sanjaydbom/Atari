import torch
import math
from torch import nn

def fun(x):
    return x.sign().mul(x.abs().sqrt())

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

        self.reset()

    def reset(self):
        nn.init.kaiming_uniform_(self.mu_k, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_k)
        bound = 0.5 / (fan_in ** 0.5)
        nn.init.uniform_(self.mu_b, -bound, bound)
        nn.init.constant_(self.sigma_k, self.std_init / math.sqrt(fan_in))
        nn.init.constant_(self.sigma_b, self.std_init / math.sqrt(fan_in))


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

        