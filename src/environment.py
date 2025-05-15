import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward
from .config import GAME_MODE,FRAMESKIP,FRAME_STACK_SIZE,SCREEN_SIZE,GRAYSCALE,SCALE,CLIP_REWARD,CLIP_BOUND

gym.register_envs(ale_py)
class AtariBreakoutEnv():
    def __init__(self, game_mode = GAME_MODE, reward_clipping = CLIP_REWARD, frame_skip = FRAMESKIP, screen_size = SCREEN_SIZE, stack_size = FRAME_STACK_SIZE, grayscale = GRAYSCALE, scale = SCALE, render_mode = None):
        """
        Initialize the environment and does the preprocessing

        We first create the environment with the game. We then start the preprocessing as described by Mnih et al., 2015 by greyscaling 
        and resizing the image, and then applying a frameskip.Then we use FrameStackObservation to stack the last stack_size frames 
        and return it to the agent

        Args:
            game_mode (string): the name and version of the game we want gymnasium to make. Default ALE/Breakout-v5
            reward_clipping (bool): enables reward clipping. Default True
            frame_skip (int): the number of frames to skip in the preprocessing. Default 4
            screen_size (int): the length and width of the screen. Default 84
            stack_size (int): how many of the previous frames will we stack and return to the model. Default 4
            greyscale (bool): enables greyscaling of image. Default True
            scale (bool): enables scaling of image. Default True

        Returns:
            None        
        """
        self.env = gym.make(game_mode, frameskip = 1, render_mode = render_mode)
        self.env = AtariPreprocessing(self.env, frame_skip = frame_skip, grayscale_obs = grayscale, scale_obs = scale, screen_size = screen_size)
        self.env = FrameStackObservation(self.env, stack_size = stack_size)
        if reward_clipping:
            self.env = TransformReward(self.env, lambda r: max(min(r, CLIP_BOUND), -1 * CLIP_BOUND))
    
    
    def reset(self):
        """
        Resets the game

        Resets the game back to the starting state. Use when game is over and we want to start the next game

        Args:
            None
            
        Returns:
            - start_state (np.ndarray): array of the starting state
        """

        start_state, _ = self.env.reset()
        return start_state

    
    def step(self, action):
        """
        Execute the action and return the next state, reward, and the done flag

        The action is applied to the environment, and the resulting state is preprocessed (grayscale, resized, stacked).
        Rewards are optionally clipped to [-1, 1] to stabilize DQN training, as described in Mnih et al., 2015. The reward clipping
        is done in the __init__

        Args:
            action (int): Action from the environments action space. (0 for No Operation, 1 for Fire, 2 for right, and 3 for left)

        Returns:
            - next_state (np.ndarray): Next state of shape (stack_size, screen_size, screen_size).
            - reward (float): Reward for the action, clipped to [-1, 1] if reward_clipping is True.
            - done (bool): Whether the episode has ended (terminated or truncated).
        """

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def close(self):
        """
        Closes the environment

        Cleans up the memory. If a closed environment is closed again, an error won't be raise

        Args:
            None
        
        Returns:
            None
        
        Notes:
            See Gymnasium documentation for more details: https://gymnasium.farama.org/api/env/#gymnasium.Env.close
        """
        self.env.close()