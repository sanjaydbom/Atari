import csv
import os
import json
import numpy as np
from .environment import AtariBreakoutEnv
from .agent import Agent
from .test import ModelEvaluator
from .config import MAX_STEPS, TRAINING_START_STEP

def train():
    env = AtariBreakoutEnv()
    agent = Agent()
    log_file = 'experiment_logs/training_log.csv'
    loss = -1.0
    prev_loss = [0] * 20
    while agent.get_steps() <= MAX_STEPS:
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state,reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            if agent.get_steps() >= TRAINING_START_STEP:
                loss = agent.train()
            agent.increment_steps()

            if agent.get_steps() > 0 and (agent.get_steps() % (1000)) == 0:
                evaluator = ModelEvaluator(agent)
                testing_data = evaluator.evaluate()
                average = testing_data["Average score"]
                prev_loss[(agent.get_steps() // 1000) % 20] = average
                print(f"{agent.get_steps() * 100 / MAX_STEPS:.2f}% done: Val Score {average:.2f}, Std Dev {testing_data["Standard Deviation"]:.2f}, Loss {loss:.2f}, Avg Val Score: {np.mean(prev_loss):.2f}")

                file_exists = os.path.isfile(log_file)
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)

                    if not file_exists:
                        writer.writerow(['step', 'loss', 'mean_score', 'median_score', 'std_score', 'max_score', 'min_score'])

                    writer.writerow([agent.get_steps(), round(loss,2), round(testing_data["Average score"],2), testing_data["Median"], round(testing_data["Standard Deviation"],2), testing_data["Max"], testing_data["Min"]])
            if agent.get_steps() % 10000 == 0 and agent.get_steps() != 0:
                agent.save_model("agent_weights/model.pt")
                '''data = {"Steps" : agent.get_steps(), "Size" : size, "Data_Pointer" : data_pointer}
                with open('agent_weights/data.json', 'w') as f:
                    json.dump(data, f, indent= 4)'''
            if done:
                break
            state = next_state

    env.close()

if __name__ == "__main__":
    train()