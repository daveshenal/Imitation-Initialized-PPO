from stable_baselines3 import PPO
from env import CarEnv
import os

# Load the environment and the trained model
env = CarEnv()  # Make sure this matches the training environment
model_path = "models/models_2/ppo_model.zip"

# Load the trained model
print("Loading the trained model...")
model = PPO.load(model_path, env=env)

# Reset environment to start a new episode
obs = env.reset()

# Run the environment to make the agent act
done = False
while not done:
    action, _states = model.predict(obs)  # Get action based on current observation
    obs, reward, done, info = env.step(action)  # Take the action and get new observation

    print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
    # You can print or visualize other details like the car's position, the camera image, etc.

# Optionally save model after some inference or interactions if needed
# model.save("new_trained_model.zip")  # Saves the updated model if fine-tuned further