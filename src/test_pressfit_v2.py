import numpy as np
from numpy import True_
from sympy import false
from my_environments import PressfitV2
from my_models.grippers.tetrapack_gripper import TetrapackGripper
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.models import MujocoWorldBase
from utils.common import register_gripper
from copy import deepcopy
import robosuite.utils.transform_utils as T

register_env(PressfitV2)
register_gripper(TetrapackGripper)
from robosuite import load_controller_config

# def calc_action(observation, env):
#     target_pos = [-9.49949353e-02, -1.40693069e-06,  1.63983968e+00] 
#     target_ori = [1, 0, 0, 0]
#     print(f"Target: {target_pos}")
#     cur_ori = observation['robot0_eef_quat']
#     cur_pos = observation['robot0_eef_pos']
#     print(f"Current: {cur_pos}")
#     cur_mat = T.quat2mat(cur_ori)
#     cur_mat = cur_mat @ T.euler2mat(np.array([0, 0, np.pi/4]))
#     pos_dif = target_pos - cur_pos
#     print(f"Pose Diff: {pos_dif}")
#     rot_dif= np.matmul(cur_mat.T, T.get_orientation_error(target_ori, cur_ori))
#     action = np.hstack([pos_dif, rot_dif, np.array([-1])])  # last number is for the gripper
#     return action

controller_config= load_controller_config(default_controller="OSC_POSE")
# controller_config['impedance_mode']='tracking'

# create environment instance
env = suite.make(
    env_name="PressfitV2", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types="TetrapackGripper",
    # gripper_types='default',
    controller_configs= controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
obs= env.reset()
done = False
while True: 

    if not done:
        # action = calc_action(obs, env)
        action = np.array([0.16617313, -0.54680493, -0.23968219, -2.14380896, -0.856529,    3.19363658, -0.05338716])[:6]

        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()
        done = True

    else:

        env.render()  # render on display
