import csv
import os

from .environment import AtariBreakoutEnv
from .agent import Agent
from .test import ModelEvaluator
from .config import MAX_STEPS, TRAINING_START_STEP, NETWORK_VALIDATION_FREQUENCY

def train():
    env = AtariBreakoutEnv()
    agent = Agent()
    steps = 0
    log_file = 'experiment_logs/training_log.csv'
    while steps <= MAX_STEPS:
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state,reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            if steps >= TRAINING_START_STEP:
                loss = agent.train()
            agent.increment_steps()
            steps = agent.get_steps()

            if steps % NETWORK_VALIDATION_FREQUENCY == 0:
                evaluator = ModelEvaluator(agent)
                testing_data = evaluator.evaluate()
                print(f"{steps * 100 / MAX_STEPS}% done: Validation Loss {testing_data["Average score"]:.2f}")

                file_exists = os.path.isfile(log_file)
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)

                    if not file_exists:
                        writer.writerow(['step', 'loss', 'mean_score', 'median_score', 'std_score', 'max_score', 'min_score'])

                    writer.writerow([steps, loss, testing_data["mean_score"], testing_data["median_score"], testing_data["std_score"], testing_data["max_score"], testing_data["min_score"]])

            if done:
                break

if __name__ == "__main__":
    train()