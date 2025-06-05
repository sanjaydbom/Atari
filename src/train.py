from .environment import AtariBreakoutEnv
from .agent import Agent
from .train import ModelEvaluator
from .config import MAX_STEPS, TRAINING_START_STEP, NETWORK_VALIDATION_FREQUENCY

def train():
    env = AtariBreakoutEnv()
    agent = Agent()
    steps = 0
    while steps <= MAX_STEPS:
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state,reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            if steps >= TRAINING_START_STEP:
                agent.train()
            agent.increment_steps()
            steps = agent.get_steps()

            if steps % NETWORK_VALIDATION_FREQUENCY == 0:
                evaluator = ModelEvaluator(agent)
                testing_data = evaluator.evaluate()
                print(f"{steps * 100 / MAX_STEPS}% done: Validation Loss {testing_data["Average score"]:.2f}")
            if done:
                break

if __name__ == "__main__":
    train()