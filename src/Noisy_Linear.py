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
        bound = 0.5 / math.sqrt(fan_in)
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


    