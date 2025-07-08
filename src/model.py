import torch
from torch import nn
from .config import FRAME_STACK_SIZE, SCREEN_SIZE
from .Noisy_Linear import NoisyLinear, fun
from .Noisy_Conv2d import NoisyConv2d

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
    def __init__(self, input_shape: tuple = (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE), num_actions: int = 4, num_bins = 51):
        super(AtariDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_bins = num_bins
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
        self.state_layer = nn.Linear(512,num_bins)
        self.advantage_layer = nn.Linear(512,num_actions * num_bins)

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
        value = self.state_layer(x)
        advantage = self.advantage_layer(x)

        advantage = advantage.view(-1,self.num_actions,self.num_bins)
        value = value.unsqueeze(1).expand_as(advantage)
        mean_advantage = advantage.mean(dim=1, keepdim = True)

        state_action = value + advantage - mean_advantage

        state_action = nn.functional.log_softmax(state_action, dim=-1)
        return state_action
