from collections import deque
import random
import typing

import torch
from torch import nn, optim
import numpy as np

from .model import AtariDQN
from .priority_replay import PriorityReplay
from .config import ACTION_SPACE,LR,EPS,MINI_BATCH_SIZE,GAMMA,CLIP_GRADIENT,MAX_GRADIENT, FRAME_STACK_SIZE, SCREEN_SIZE, NETWORK_UPDATE_FREQUENCY, NUM_BINS, MAX_BIN_VALUE, MIN_BIN_VALUE

class Agent:
    """Agent to learn how to play Atari games

    Implements a Deep Q-Learning agent with an experience replay buffer and a target network to store memories and a secondary target network to evaluate action choices
    The network architecture and training process are based on the methods described in Mnih et al. (2015), 'Human-level control through deep reinforcement learning'.
    
    Args:
        action_space (list[int]): the actions that we can take. Default [0,1,2,3]
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
        To learn more about RMSprop, visit http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf 
        To learn more about Network-Target Network Architecture, visit https://www.nature.com/articles/nature14236
    """
    def __init__(self,action_space: list[int] = ACTION_SPACE, lr: float = LR, eps: float = EPS, gamma: float = GAMMA, mini_batch_size: int = MINI_BATCH_SIZE, clip_gradient: bool = CLIP_GRADIENT, max_gradient: float = MAX_GRADIENT, network_validation_frequency = NETWORK_UPDATE_FREQUENCY, num_bins = NUM_BINS, max_bin_value = MAX_BIN_VALUE, min_bin_value = MIN_BIN_VALUE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.mps.is_available() else self.device)
        self.model = AtariDQN().to(device=self.device)
        self.target_model = AtariDQN().to(device=self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.action_space = action_space

        self.optimizer = optim.Adam(self.model.parameters(), lr, eps = eps)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        self.memory = PriorityReplay()

        self.gamma = gamma
        self.mini_batch_size = mini_batch_size
        self.clip_gradient = clip_gradient
        self.max_gradient = max_gradient


        self.input_shape = (FRAME_STACK_SIZE, SCREEN_SIZE, SCREEN_SIZE)
        self.num_steps = 0
        self.validation_frequency = network_validation_frequency

        self.bins = torch.linspace(min_bin_value,max_bin_value,num_bins).to(self.device)
        self.num_bins = num_bins
        self.min_bin_value = min_bin_value
        self.max_bin_value = max_bin_value

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
        else:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
                state = state.to(self.device)
                dist = self.model(state)
                q_values = (dist * self.bins).sum(dim = 2)
                action = torch.argmax(q_values).item()
                return action
                

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
        if not isinstance(action, int) or action not in self.action_space:
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
        memories, indicies, weights = self.memory.sample(self.mini_batch_size)
        state, action, reward, next_state = zip(*memories)

        states = torch.from_numpy(np.array(state)).float().to(self.device)
        actions = torch.tensor(action, dtype = torch.long).to(device=self.device)
        rewards = torch.tensor(reward, dtype = torch.float32).to(device=self.device)
        non_terminal_mask = torch.tensor([ns is not None for ns in next_state], dtype=torch.bool).to(self.device)
        next_states = torch.stack([torch.from_numpy(ns).float() if ns is not None else torch.zeros(self.input_shape) for ns in next_state]).to(device=self.device)
        q_values = self.model(states)[torch.arange(states.shape[0]), actions].to(self.device)
        with torch.no_grad():
            distributions = self.model(next_states[non_terminal_mask]).to(self.device)#(non terminal state, num actions, num bins)
            distributions = torch.nn.functional.softmax(distributions, dim = -1)

            expected_rewards = (distributions * self.bins.unsqueeze(0).unsqueeze(0)).sum(dim=-1).to(self.device)#(non terminal states, num_actions)
            best_actions = torch.argmax(expected_rewards, dim = -1).to(self.device)#(non terminal states)
            distributions = torch.nn.functional.softmax(self.target_model(next_states[non_terminal_mask]).to(self.device), dim = -1)

            Tz = torch.zeros(MINI_BATCH_SIZE, self.num_bins).to(self.device)#will hold the reward bins that are being shifted
            Tz[non_terminal_mask] += rewards[non_terminal_mask].unsqueeze(1) + self.gamma * self.bins 
            clamped_rewards = torch.clamp(Tz,self.min_bin_value,self.max_bin_value).to(self.device)#remapping the reward bins onto out regular bins
            b = ((clamped_rewards - self.min_bin_value) * (self.num_bins - 1) / (self.max_bin_value - self.min_bin_value)).to(self.device)
            l = torch.floor(b).type(torch.int64) #lower bin for each reward
            u = torch.ceil(b).type(torch.int64) #upper bin for each reward
            weights_l = u.float()-b
            weights_u = b - l.float()
            l_clamped = torch.clamp(l, 0, self.num_bins - 1)
            u_clamped = torch.clamp(u, 0, self.num_bins - 1)
            target = torch.zeros(self.mini_batch_size, self.num_bins).to(self.device) #will hold out target distribution
            probs_to_project = distributions[torch.arange(distributions.shape[0]),best_actions]
            target[non_terminal_mask] = target[non_terminal_mask].scatter_add_(1,l_clamped[non_terminal_mask], weights_l[non_terminal_mask] * probs_to_project)
            target[non_terminal_mask] = target[non_terminal_mask].scatter_add_(1,u_clamped[non_terminal_mask], weights_u[non_terminal_mask] * probs_to_project)
            terminal_rewards = rewards[~non_terminal_mask]
            clamped_terminal_rewards = torch.clamp(terminal_rewards,self.min_bin_value,self.max_bin_value).to(self.device)
            b = ((clamped_terminal_rewards - self.min_bin_value) * (self.num_bins - 1) / (self.max_bin_value - self.min_bin_value)).to(self.device)
            l = torch.floor(b).type(torch.int64) #lower bin for each reward
            u = torch.ceil(b).type(torch.int64) #upper bin for each reward

            weights_l = u.float()-b
            weights_u = b - l.float()
            l_clamped = torch.clamp(l, 0, self.num_bins - 1)
            u_clamped = torch.clamp(u, 0, self.num_bins - 1)

            target[~non_terminal_mask] = target[~non_terminal_mask].scatter_add_(1,l_clamped.unsqueeze(1), weights_l.unsqueeze(1))
            target[~non_terminal_mask] = target[~non_terminal_mask].scatter_add_(1,u_clamped.unsqueeze(1), weights_u.unsqueeze(1))

            log_softmax_loss = torch.nn.functional.log_softmax(q_values,-1)
            unreduced_loss = -(target* log_softmax_loss).sum(dim = 1)

            priorities = unreduced_loss.abs().detach().cpu().tolist()
            self.memory.update_all_td(indicies, priorities)

        log_softmax_loss = torch.nn.functional.log_softmax(q_values,-1)
        unreduced_loss = -(target * log_softmax_loss).sum(dim = 1)
        loss = (unreduced_loss * torch.tensor(weights).to(self.device)).mean()
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
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename: str = "model.pt") -> None:
        """Saves the models parameters to a file called model.pt for later use
        """
        torch.save(self.model.state_dict(), filename)
        optimizer_path = filename.replace('.pt', '_optimizer.pt')
        torch.save(self.optimizer.state_dict(), optimizer_path)
        '''self.memory.save_memory()
        return self.memory.get_data()'''

    def increment_steps(self):
        self.num_steps += 1
        self.memory.anneal_beta()
        if self.num_steps % self.validation_frequency == 0:
            self.update_target_network()

            
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, weights_only = True))
        self.model.to(self.device)
        optimizer_path = filename.replace('.pt', '_optimizer.pt')
        self.optimizer.load_state_dict(torch.load(optimizer_path))

    def get_steps(self):
        return self.num_steps   