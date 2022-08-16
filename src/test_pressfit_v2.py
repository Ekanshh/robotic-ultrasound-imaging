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
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display