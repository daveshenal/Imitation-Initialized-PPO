'''
You need to run this to start training a new RL model/policy.
Pre-requisite: Carla Sim should be running before you start training.

Terminology:
	Episode		 : one go of the car trying to "live" in the simulation and earn max rewards.
				   Episode start from spawning a new car and ends with either car crashing or 
                   episode duration limit running out.

	Timestep	 : one frame through simulation: the car gets a camera image, reward from prior 
                   step and then it makes a decision on control input and sends it to simulation.

	Reward logic : each timestep a logic is applied to calculate a reward from latest step. 
				   This logic represents you describing the desired behaviour the car needs to learn.

	Policy/model : our objective, what we need learn as part of RL. This is the latest set of rules 
                   on what to do at a given camera image.

	Iterations   : RL training sessions (multiple episodes but fixed number of timesteps). We define 
     			   number of timesteps per iteration in "TIMESTEPS" variable later.
'''

from stable_baselines3 import PPO #PPO
import os
from env import CarEnv

print('Start of training script')


print('Setting folders for logs and models')
# Define fixed directories for models and logs so the same files are reused
models_dir = "models/models_2"
logdir = "logs/logs_2"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)



print('Connecting to environment...')
env = CarEnv()
env.reset()
print('Environment has been reset')



# Define the path to the latest saved model
model_path = f"{models_dir}/ppo_model.zip"

# Load the existing model if it exists, or start a new one
if os.path.exists(model_path):
    print("Loading existing model for continued training...")
    model = PPO.load(model_path, env=env, tensorboard_log=logdir)
else:
    print("No saved model found; training a new one...")
    model = PPO('MultiInputPolicy', env, verbose=1, learning_rate=0.001, tensorboard_log=logdir)
    


# Set training parameters
TIMESTEPS = 1000     # number of timesteps per training iteration
iterations_per_run = 20  # number of iterations per run

for i in range(iterations_per_run):
    print(f'Starting iteration {i + 1} of {iterations_per_run}...')
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    print(f'Iteration {i + 1} complete. Saving model...')
    model.save(model_path)


# python train.py

# netstat -aon | findstr :2000
# taskkill /PID <PID> /F