import numpy as np
from numpy import True_
from my_environments import Pressfit
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.models import MujocoWorldBase

register_env(Pressfit)

# create environment instance
env = suite.make(
    env_name="Pressfit", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True_,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
