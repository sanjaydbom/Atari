from .config import NUM_TESTS
from .environment import AtariBreakoutEnv
import numpy as np

class ModelEvaluator():
    def __init__(self,model, env = AtariBreakoutEnv, num_tests = NUM_TESTS):
        self.model = model
        self.env = env()
        self.num_tests = num_tests

    def evaluate(self):
        scores = []
        for i in range(self.num_tests):
            score = 0
            state = self.env.reset()
            while True:
                action = self.model.get_action(state, True)
                state,reward,done = self.env.step(action)
                score += reward
                if done:
                    scores.append(score)
                    break
        return {
            "Average score" : np.mean(scores),
            "Standard Deviation" : np.std(scores),
            "Max" : np.max(scores),
            "Min" : np.min(scores),
            "Median" : np.median(scores)
        }
