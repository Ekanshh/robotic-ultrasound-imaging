from cgi import print_environ_usage
import numpy as np
from my_models.objects.xml_objects import TetrapackObject

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
from robosuite.utils.observables import Observable, sensor


# Python imports
from distutils.log import warn
import re
from warnings import WarningMessage
import numpy as np
from my_models.arenas.pressfit_arena import PressfitArena
# Local imports
from my_models.objects import ContainerWithTetrapacksObject
# Robosuite imports
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import find_elements
from robosuite.models.base import MujocoModel

import xml.etree.ElementTree as ET


class PressfitV2(SingleArmEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="TetrapackGripper",
        table_full_size=(1.0, 1.0, 0.05),       # Table properties
        table_friction= 100*(1., 5e-3, 1e-4),
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="customview",
        render_collision_mesh=True,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="customview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # Assert that the gripper type is None
        # assert gripper_types is None, "Tried to specify gripper other than None in TwoArmPegInHole environment!"

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset =  np.array((0.2, 0, 1.4))

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):

        reward = 0

        # Right location and angle
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            # Grab relevant values
            t, d, cos = self._compute_orientation()
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, scale sparse reward so that the max reward is identical to its dense version
        else:
            reward *= 5.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 5.0

        return reward

    def _load_model(self):

        super()._load_model()

        # Adjust base pose(s) accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Add arena and robot
        mujoco_arena = PressfitArena(
            table_full_size = self.table_full_size,
            table_offset = self.table_offset
        )
        table_top = mujoco_arena.table_top_abs

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0666432116509934, 1.4903257668114777e-08, 2.0563394967349096],
            quat=[0.6530979871749878, 0.27104058861732483, 0.27104055881500244, 0.6530978679656982],
        )

        # initialize objects of interest
        self.hole = ContainerWithTetrapacksObject(name="container_with_tetrapacks")
        
        # To check if the hole is correctly shown in simulation
        # self.peg_radius=(0.01, 0.03)
        # self.peg_length=0.07
        # self.peg = CylinderObject(
        #     name="peg",
        #     size_min=(self.peg_radius[0], self.peg_length),
        #     size_max=(self.peg_radius[1], self.peg_length),
        #     rgba=[0, 1, 0, 1],
        #     joints=None,
        # )
        # peg_obj = self.peg.get_obj()
        # peg_obj.set('pos', '-0.242 0.059 1.585' )
        # peg_obj.set('quat', '-0.500 0.500 0.500 -0.500')

        # Load hole object
        hole_obj = self.hole.get_obj()
        hole_obj.set('quat', '0.707 0. 0. 0.707')
        hole_obj.set('pos', '0.2 0.0 1.4')

       

        # Append appropriate objects to arms

        # task includes arena, robot, and objects of interest
        # We don't add peg and hole directly since they were already appended to the robots
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.hole]
        )


        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.hole)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id('container_with_tetrapacks_tetrapack_4')
        print(f"Hole Body ID: {self.hole_body_id}")
        self.peg_body_id = self.sim.model.body_name2id(self.robots[0].gripper.root_body)
        print(f"Peg Body ID: {self.peg_body_id}")

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of peg and hole
            @sensor(modality=modality)
            def hole_pos(obs_cache):
                print(100 * '-')
                print(f"Observables| Hole Pose:= {self.sim.data.body_xpos[self.hole_body_id]}")
                return np.array(self.sim.data.body_xpos[self.hole_body_id])

            @sensor(modality=modality)
            def hole_quat(obs_cache):
                print(f"Observables| Hole Quat:= {T.convert_quat(self.sim.data.body_xquat[self.hole_body_id], to='xyzw')} ")
                return T.convert_quat(self.sim.data.body_xquat[self.hole_body_id], to="xyzw")

            @sensor(modality=modality)
            def peg_to_hole(obs_cache):
                print(f"Observables| Peg to hole:= {(obs_cache['hole_pos'] - np.array(self.sim.data.body_xpos[self.peg_body_id]) if 'hole_pos' in obs_cache else np.zeros(3))} ")
                return (
                    obs_cache["hole_pos"] - np.array(self.sim.data.body_xpos[self.peg_body_id])
                    if "hole_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                print(f"Observables| Peg Quat:= {T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to='xyzw')} ")
                return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")

            # Relative orientation parameters
            @sensor(modality=modality)
            def angle(obs_cache):
                t, d, cos = self._compute_orientation()
                obs_cache["t"] = t
                obs_cache["d"] = d
                print(f"Observables| Relative angle between peg and hole:= {cos} ")
                return cos

            @sensor(modality=modality)
            def t(obs_cache):
                print(f"Observables| Parallel distance peg and hole:= {obs_cache['t'] if 't' in obs_cache else 0.0} ")
                return obs_cache["t"] if "t" in obs_cache else 0.0

            @sensor(modality=modality)
            def d(obs_cache):
                print(f"Observables| Perpendicular distance between peg and hole:= {obs_cache['d'] if 'd' in obs_cache else 0.0} ")
                return obs_cache["d"] if "d" in obs_cache else 0.0

            sensors = [hole_pos, hole_quat, peg_to_hole, peg_quat, angle, t, d]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.robots[0].set_robot_joint_positions([0.000, -1.019, -0.005, -2.498, 0.050, 2.942, 0.785])

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        t, d, cos = self._compute_orientation()

        return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - peg_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, peg_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)),
        )

    def _peg_pose_in_hole_frame(self):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.

        Returns:
            np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos(self.peg.root_body)
        peg_rot_in_world = self.sim.data.get_body_xmat(self.peg.root_body).reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        # World frame
        hole_pos_in_world = self.sim.data.get_body_xpos(self.hole.root_body)
        hole_rot_in_world = self.sim.data.get_body_xmat(self.hole.root_body).reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(peg_pose_in_world, world_pose_in_hole)
        return peg_pose_in_hole