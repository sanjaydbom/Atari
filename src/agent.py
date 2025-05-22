from collections import deque
import random
import typing

import torch
from torch import nn, optim
import numpy as np

from model import AtariDQN
from .config import ACTION_SPACE,EPSILON,MIN_EPSILON,EPSILON_REDUCTION,LR,ALPHA,EPS, MEMORY_SIZE,MINI_BATCH_SIZE,GAMMA,CLIP_GRADIENT,MAX_GRADIENT, FRAME_STACK_SIZE, SCREEN_SIZE, NETWORK_VALIDATION_FREQUENCY, TAU

class Agent:
    """Agent to learn how to play Atari games

    Implements a Deep Q-Learning agent with an experience replay buffer and a target network to store memories and a secondary target network to evaluate action choices
    The network architecture and training process are based on the methods described in Mnih et al. (2015), 'Human-level control through deep reinforcement learning'.
    
    Args:
        action_space (list[int]): the actions that we can take. Default [0,1,2,3]
        epsilon (float): the probability of choosing a random action. Default 1.0 or 100%
        min_epsilon (float): the smallest value we let epsilon get to. Default 0.1
        epsilon_reduction (float): The amount we reduce epsilon by after each step. Default 9e-7
        lr (float): the amount we update the parameters by. Default 2.5e-4
        alpha, eps (float): parameters used to control RMSprop
        gamma (float): how much we care about future values, higher means we care more about future values. Default 0.99
        mini_batch_size (int): how many memories we train on at once, higher is more stable, lower is faster. Default 32
        memory_size (int): how big our memory is. Default 300,000  
        clip_gradient (bool): enables clipping gradient. Default True
        max_gradient (float): how much to clip gradient by. Default 10.0

    Attributes:
        device (Torch.device): Where our training will be run. If the environment has a GPU, then we use it otherwise we use the CPU
        model (Model): the neural network we will be training
        target_model (Model): the network used to evaluate our actions
        action_space (int): the number of actions we can take. Default 4
        epsilon (float): the probability of choosing a random action. Default 1.0 or 100%
        min_epsilon (float): the smalled value we let epsilon get to. Default 0.1
        epsilon_reduction (float): The amount we reduce epsilon by after each step. Default 9e-7
        optimizer (nn.optim.RMSprop): The optimizer that we use to update the parameters in model
        loss_fn (nn.SmoothL1Loss): The function we will use to compute the loss
        memory (deque): used to store our actions to use later to train the model
        gamma (float): how much we care about future values, higher means we care more about future values. Default 0.99
        mini_batch_size (int): how many memories we train on at once, higher is more stable, lower is faster. Default 32
        clip_gradient (bool): enables clipping gradient. Default True
        max_gradient (float): how much to clip gradient by. Default 10.0
        tau (float): Parameter that controls how much we update the target model by. Default 0.0001
        input_shape (int,int,int): The dimensions of the state array that we are expecting. Default (FRAME_STACK, SCREEN_SIZE, SCREEN_SIZE)
        num_steps (int): how many steps we have taken
        validation_frequency (int): How frequently to we test our model. Default every 5,000 steps

    Notes:
        To learn more about epsilon decay visit "Reinforcement Learning: An Introduction" by Barto and Sutton
        To learn more about RMSprop, visit http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf 
        To learn more about Network-Target Network Architecture, visit https://www.nature.com/articles/nature14236
    """
    def __init__(self,action_space: list[int] = ACTION_SPACE, epsilon: float = EPSILON, min_epsilon: float = MIN_EPSILON, epsilon_reduction: float = EPSILON_REDUCTION, lr: float = LR, alpha: float = ALPHA, eps: float = EPS, gamma: float = GAMMA, mini_batch_size: int = MINI_BATCH_SIZE, memory_size: int = MEMORY_SIZE, clip_gradient: bool = CLIP_GRADIENT, max_gradient: float = MAX_GRADIENT, network_validation_frequency = NETWORK_VALIDATION_FREQUENCY, tau = TAU):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.mps.is_available() else self.device)
        self.model = AtariDQN().to(device=self.device)
        self.target_model = AtariDQN().to(device=self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.action_space = action_space

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_reduction = epsilon_reduction

        self.optimizer = optim.RMSprop(self.model.parameters(), lr, alpha = alpha, eps = eps)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.clip_gradient = clip_gradient
        self.max_gradient = max_gradient
        self.tau = tau

        self.input_shape = (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
        self.num_steps = 0
        self.validation_frequency = network_validation_frequency

    def get_action(self, state: np.ndarray, evaluation: bool = False) -> int:
        """Chooses action according to epsilon-greedy policy

        With a probability of epsilon, we choose a random action, and with a probability of 1-epsilon, we choose
        the best action possible. This ensures that we can explore all states properly, but also that we learn
        the best actions.

        Args:
            state (np.ndarray): the array that represents the state of size (FRAME_STACK, SCREEN_SIZE, SCREEN_SIZE).
            evaluation (bool): tells if we are getting the action for a test or not. If it is a test, we always choose 
                         the best move. 
        
        Returns:
            action (int): resulting action-0 is no op, 1 is fire, 2 is right, 3 is left

        Raises:
            Value Error: If the state array is of the wrong dimensions
        """
        if state.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {state.shape}")
        
        if random.random() < self.epsilon and not evaluation:
            return random.choice(self.action_space)
        else:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
                state = state.to(self.device)
                action_values = self.model(state)
                action = torch.argmax(action_values).item()
                return action
            
    def update_epsilon(self) -> None:
        """Reduces epsilon by epsilon_reduction

        It ensures that epsilon isn't lower than min_epsilon
        """
        self.epsilon = max(self.epsilon - self.epsilon_reduction, self.min_epsilon)

    def store_experience(self,state:np.ndarray, action:int, reward:float, next_state:np.ndarray, done:bool) -> None:
        """Saves the state,action,reward,next_state tuple

        Saves the necessary variables that we need update the model. We set next_state to None if we are done
        because that means that next_state is a terminal state so we don't need to evaluate it, so setting it to
        None saves space.

        Args:
            state (np.ndarray): Array that represents the state of dimensions (FRAME_STACK, SCREEN_SIZE, SCREEN_SIZE)
            action (int): The action that we took in the state
            reward (float): The reward we got from the action
            next_state (np.ndarray): Array that represents the state resulting from the action of dimensions (FRAME_STACK, SCREEN_SIZE, SCREEN_SIZE)
            done (bool): Represents if next_state is terminal

        Raises:
            Value Error if state or next_state are of the wrong size, or if action is not a valid action
        """
        if state.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {state.shape}")
        if not done and next_state.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {next_state.shape}")
        if isinstance(action, int) or action not in self.action_space:
            raise ValueError(f"Action {action} not in action space")
        
        if done:
            next_state = None
        self.memory.append((state,action,reward,next_state))

    def train(self) -> typing.Optional[float]:
        """Replays the memories and updates the model

        We select mini_batch_size number of random memories and we update model, with gradient clipping if
        it is enabled. See the __init__ for the optimizer and the loss function. The target value is 
        reward + gamma * (value of the next_state), and the actual value is the predicted value of the 
        current state. We use a Double DQN (van Hasselt et al., 2016) to optimize training

        Either returns None if no backprop was done, or a float stating the loss value
        """
        if len(self.memory) < self.mini_batch_size:
            return None
        
        memories = random.sample(self.memory, k = self.mini_batch_size)
        state, action, reward, next_state = zip(*memories)

        states = torch.from_numpy(np.array(state)).float().to(self.device)
        actions = torch.tensor(action, dtype = torch.long).to(device=self.device)
        rewards = torch.tensor(reward, dtype = torch.float32).to(device=self.device)
        non_terminal_mask = torch.tensor([ns is not None for ns in next_state], dtype=torch.bool).to(self.device)
        next_states = torch.stack([ns if ns is not None else torch.zeros_like(torch.tensor(self.input_shape)) for ns in next_state]).to(device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            target = torch.zeros(self.mini_batch_size)
            next_action = self.model(next_states[non_terminal_mask]).argmax(1)
            target[non_terminal_mask] = self.gamma * self.target_model(next_states[non_terminal_mask]).gather(1, next_action.unsqueeze(1)).squeeze()
            target += rewards
            target = target.unsqueeze(1)
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self) -> None:
        """Resets the target network's parameters to the model's parameters

        We update the target infrequently to stabilize training
        See https://www.nature.com/articles/nature14236 for more detail
        """
        for target_param, param in zip(self.target_model.parameters(),self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)

    def save_model(self, filename: str = "model.pt") -> None:
        """Saves the models parameters to a file called model.pt for later use
        """
        torch.save(self.model.state_dict(), filename)
        optimizer_path = filename.replace('.pt', '_optimizer.pt')
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def increment_steps(self):
        self.num_steps += 1
        if self.num_steps % self.validation_frequency == 0:
            self.update_target_network()
            

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename), weights_only = True)
        optimizer_path = filename.replace('.pt', '_optimizer.pt')
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        