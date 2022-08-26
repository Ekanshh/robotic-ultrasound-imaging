import numpy as np
import robosuite as suite

def calc_action(observation):
    target_pos = np.array([-0.35, 0.5, 1.35])
    target_ori = np.array([1, 0, 0, 0])

    cur_ori = observation['robot1_eef_quat']
    cur_pos = observation['robot1_eef_pos']
    cur_mat = quat2mat(cur_ori)
    cur_mat = cur_mat @ euler2mat(np.array([0, 0, -np.pi/2]))
    
    pos_dif = target_pos - observation['robot1_eef_pos']
    rot_dif= np.matmul(cur_mat.T, get_orientation_error(target_ori, cur_ori))
    action = np.hstack([pos_dif, rot_dif, np.array([-1])])  # last number is for the gripper
    return action`

env = suite.make(
            'Pressfit',
            robots='Panda',
            gripper_types="TetrapackGripper",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=False,
            control_freq=50,
            render_camera=None,
            horizon=2000,      
        )

robosuite_simulation_controller_test(env, env.horizon)