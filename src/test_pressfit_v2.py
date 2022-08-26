from multiprocessing.connection import wait
import numpy as np
from numpy import True_
from my_environments import PressfitV2
from my_models.grippers.tetrapack_gripper import TetrapackGripper
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.models import MujocoWorldBase
from utils.common import register_gripper

register_env(PressfitV2)
register_gripper(TetrapackGripper)

# create environment instance
env = suite.make(
    env_name="PressfitV2", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types="TetrapackGripper",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera=None,
)

# reset the environment
env.reset()
done = True

while True: 
    env.render()
    if done:
        action = [0, 0, 0, 0, 0, 0, 0] # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on displa
        done = False