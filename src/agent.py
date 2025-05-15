from collections import deque
from model import Model
import torch
from torch import optim
from torch import nn
import random
from .config import ACTION_SPACE,EPSILON,MIN_EPSILON,EPSILON_REDUCTION,LR,ALPHA,EPS, MEMORY_SIZE,MINI_BATCH_SIZE,GAMMA,CLIP_GRADIENT,MAX_GRADIENT

class Agent():
    def __init__(self,action_space = ACTION_SPACE, epsilon = EPSILON, min_epsilon = MIN_EPSILON, epsilon_reduction = EPSILON_REDUCTION):
        self.model = Model()
        self.target_model = Model()
        self.target_model.load_state_dict(self.model.state_dict())

        self.action_space = action_space

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_reduction = epsilon_reduction

        self.optimizer = optim.RMSprop(self.model.parameters(), LR, alpha = ALPHA, eps = EPS)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.memory = deque(maxlen=MEMORY_SIZE)

    def get_action(self, state, test = False):
        if random.random() < self.epsilon and not test:
            random.choice(ACTION_SPACE)
        else:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
                action_values = self.model(state)
                return torch.argmax(action_values)
            
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_reduction, self.min_epsilon)

    def remember(self,state,action,reward,next_state, done):
        if done:
            next_state = None
        self.memory.append((state,action,reward,next_state))

    def replay(self):
        memories = random.sample(self.memory, k = MINI_BATCH_SIZE)
        state, action, reward, next_state = zip(*memories)

        states = torch.stack([s for s in state])
        actions = torch.tensor(action, dtype = torch.long)
        rewards = torch.tensor(reward, dtype = torch.float32)
        non_terminal_states = [ns is not None for ns in next_state]
        next_states = torch.stack([ns if ns is not None else torch.zeros_like(states[0]) for ns in next_state])

        q_values = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            target = torch.zeros(MINI_BATCH_SIZE)
            next_action = self.model(next_states[non_terminal_states]).argmax(1)
            target[non_terminal_states] = GAMMA * self.target_model(next_states[non_terminal_states]).gather(1, next_action.unsqueeze(1)).squeeze()
            target += rewards
            target = target.unsqueeze(1)
        loss = self.loss_fn(q_values, target)

        optim.zero_grad()
        loss.backward()
        if CLIP_GRADIENT:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRADIENT)
        optim.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pt")