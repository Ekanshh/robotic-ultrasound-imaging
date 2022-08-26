# Python imports
import re
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

from my_models.grippers import TetrapackGripper

class Pressfit(SingleArmEnv):
    def __init__(
        self,       # DESCRIPTION OF PARAMETERS AT: https://robosuite.ai/docs/simulation/environment.html?highlight=use_camera_obs#:~:text=dimension%20%3Artype%3A%20int-,Robot%20Environment,%EF%83%81,-class
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types=None, 
        initialization_noise="default",
        table_full_size=(1.0, 1.0, 0.05),       # Table properties
        table_friction= 100*(1., 5e-3, 1e-4),   
        use_camera_obs=False,        # if True, every observation includes rendered image(s)
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,         #  If true, render the simulation state in a viewer instead of headless mode.
        has_offscreen_renderer=False,        #  True if using off-screen rendering
        render_camera="customview",       # Name of camera to render if has_renderer is True. Setting this value to ‘None’ will result in the default angle being applied, which is useful as it can be dragged / panned by the user using the mouse
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
        early_termination= False,
        save_data = False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):

        # Check if the gripper is correct
        # assert gripper_types == "TetrapackGripper",\
        #     "Tried to specify gripper other than TetrapackGripper in Pressfit environment!"

        # Check if the robot is correct
        assert robots == "Panda", \
            "Robot must be Panda!"

        # Setting table properties
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset =  np.array((0.2, 0, 1.4))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="RethinkMount",
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

        ################################
        # Initialize objects of interest
        ################################

        # Object 1: Container with tetrapacks
        self.container = ContainerWithTetrapacksObject(
            name="container_with_tetrapacks",
        )
    
        container_obj= self.container.get_obj()
        container_obj.set('pos', '0.2 0.0 1.4')
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
        self.gripper_id = self.sim.model.body_name2id(self.robots[0].gripper.root_body)
        print(self.gripper_id)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        gripper_contact = self._check_gripper_contact_with_table(self.robots[0].gripper)
        gripper_contact = self._check_gripper_contact_with_tetrapacks(self.robots[0].gripper)
        # self._get_target_position()

        sensors= []

        # probe information
        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"       # Need to use this modality since proprio obs cannot be empty in GymWrapper


        @sensor(modality=modality)
        def chec_ext_contact(obs_cache):
            return self._check_gripper_contact_with_tetrapacks(self.robots[0].gripper)

        @sensor(modality=modality)
        def eef_contact_force(obs_cache):
            # if self.sim.data.cfrc_ext[self.gripper_id][-1:]:  
            print(f"EEF contact force: {self.sim.data.cfrc_ext[self.gripper_id][-3:]}")
            return self.sim.data.cfrc_ext[self.gripper_id][-3:]

        @sensor(modality=modality)
        def eef_torque(obs_cache):
            print(f"EEF Torque: {self.robots[0].ee_torque}")
            return self.robots[0].ee_torque

        @sensor(modality=modality)
        def eef_vel(obs_cache):
            # print(f"EEF Velocity: {self.robots[0]._hand_vel}")
            return self.robots[0]._hand_vel

        sensors += [
            chec_ext_contact,
            eef_contact_force,
            eef_torque, 
            eef_vel]

        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _check_gripper_contact_with_table(self, model):
        """Get the contact force at the gripper object

        Args:
            model (MujocoModel): An instance of MujocoModel

        Returns:
            boolean: True when gripper touch the table, otherwise false
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        
        # Initialize default gripper contact with table
        self.has_touched_tetrapackss = False
        # Hard-coded regex pattern extracted from the autogenerated composite mujoco gripper model
        gripper_regex_pattern = "^gripper0_EEG[0-9]+_[0-9]+_[0-9]+$"
        # print("Checking gripper contact with table")
        # Loop through all the contacts points
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # Get the corresponding geometry pairs <g1, g2>
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            # print(g1, g2)
            # Match the hard-coded gripper regex with geometry pairs
            match1 = re.search(gripper_regex_pattern, g1)
            match2 = re.search(gripper_regex_pattern, g2)
            # Check if match found
            if match1 != None or match2 != None:
                if g1 == "table_collision"  or g2 == "table_collision":
                    print("Table Collision:", g1, g2)
                    self.has_touched_tetrapackss = True       # Gripper in contact with table
                    print(40 * '-' + "Gripper in contact with table"+ 40 * '-')
                    return self.has_touched_tetrapackss
    
    def _check_gripper_contact_with_tetrapacks(self, model):
        """Get the contact force at the gripper object

        Args:
            model (MujocoModel): An instance of MujocoModel

        Returns:
            boolean: True when gripper touch the table, otherwise false
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        
        # Initialize default gripper contact with table
        self.has_touched_tetrapacks = False
        # Hard-coded regex pattern extracted from the autogenerated composite mujoco gripper model
        gripper_regex_pattern = "^gripper0_EEG[0-9]+_[0-9]+_[0-9]+$"
        tetrapacks_regex_pattern = "container_with_tetrapacks_g[0-9]+"
        # print("Checking gripper contact with table")
        # Loop through all the contacts points
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # Get the corresponding geometry pairs <g1, g2>
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            # Match the hard-coded gripper regex with geometry pairs
            match1 = re.search(gripper_regex_pattern, g1)
            match2 = re.search(tetrapacks_regex_pattern, g2)
            match3 = re.search(tetrapacks_regex_pattern, g1)
            match4 = re.search(gripper_regex_pattern, g2)
            # Check if match found
            if match1 != None and match2 != None or match3 != None and match4 != None: 
                print(match1, match2, match3, match4)
                print("EEF collision with Tetrapacks: ", g1, g2)
                self.has_touched_tetrapackss = True       # Gripper in contact with tetrapacks
                return self.has_touched_tetrapackss

    # Reference: https://gist.github.com/machinaut/209c44e8c55245c0d0f0094693053158
    def _get_gripper_contact_force(self, model):
        """Check if the gripper is in contact with the table or not

        Args:
            model (MujocoModel): An instance of MujocoModel

        Returns:
            boolean: True when gripper touch the table, otherwise false
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        
        # Initialize default gripper contact force
        self.contact_force = np.zeros(6)
        # Hard-coded regex pattern extracted from the autogenerated composite mujoco gripper model
        gripper_regex_pattern = "^gripper0_EEG[0-9]+_[0-9]+_[0-9]+$"

        print("Checking gripper contact force")
        # Loop through all the contacts points
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # Get the corresponding geometry pairs <g1, g2>
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            # print(g1, g2)
    
    def _get_target_position(self,):
        self.hole_body_id = self.sim.model.body_name2id('container_with_tetrapacks_tetrapack_4')
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        print("Print container id: ", self.hole_body_id)
        print("Print tetrapack 4 position: ", hole_pos)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Set robot initial joint positions 
        # Hard-coded values generated from tune_robot_joints.py to start robot near the container
        # self.robots[0].set_robot_joint_positions([0.000, -1.019, -0.005, -2.498, 0.050, 2.942, 0.785])
        # Hard-coded values generated from tune_robot_joints.py to establish robot collision with table 
        # self.robots[0].set_robot_joint_positions([0.000, -0.569, -0.005, -2.498, 0.050, 2.942, 0.785])
        # Hard-coded values generated from tune_robot_joints.py to establish robot collision with container tetrapacks 
        # self.robots[0].set_robot_joint_positions([0.050, 0.056, -0.005, -1.693, 0.050, 2.942, 0.785])
        # test
        self.robots[0].set_robot_joint_positions([0.000, -0.269, -0.005, -1.798, 0.000, 2.942, 0.785])
        


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
