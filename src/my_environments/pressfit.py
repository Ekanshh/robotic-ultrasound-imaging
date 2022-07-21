
# Python imports
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


class Pressfit(SingleArmEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,
        initialization_noise="default",
        use_latch=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="customview",
        render_collision_mesh=False,
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

        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = np.array((1.0, 1.0, 0.05))
        self.table_offset =  np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = True

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
        reward = 0.0
        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = PressfitArena(
            table_full_size = self.table_full_size,
            table_offset = self.table_offset
        )
        table_top = mujoco_arena.table_top_abs
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        # camera mode="fixed" name="frontview" pos="3.0041728459287134 -8.978002622257044e-08 1.6182244956398628" quat="0.5608418583869934 0.43064647912979126 0.4306463599205017 0.5608418583869934"
        # mujoco_arena.set_camera(
        #     camera_name="agentview",
        #     pos="-0.9758789255999046 -1.9886114909882024 1.2507276767101316",
        #     quat="0.7047281265258789 0.6816641688346863 -0.16726869344711304 -0.10350527614355087")

        ################################
        # Initialize objects of interest
        ################################

        # Object 1: Container with tetrapacks
        self.container = ContainerWithTetrapacksObject(
            name="container_with_tetrapacks",
        )
        

        container_obj= self.container.get_obj()
        container_obj.set('pos', '0.2 0.0 1.0')
        container_obj.set('quat', '0.707 0. 0. 0.707')

        ###########################
        # Robot end-effector object
        ###########################
        
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.container]
        )

        self.model.merge_assets(self.container)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_success(self):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)
