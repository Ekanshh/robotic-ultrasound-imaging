from turtle import pos
from my_models.objects import  ContainerWithTetrapacksObject
from my_models.objects.xml_objects import SoftBoxObject
from robosuite.models import MujocoWorldBase


from robosuite.models.robots import Panda
from my_environments import Pressfit
from utils.common import register_gripper
from robosuite.environments.base import register_env

world = MujocoWorldBase()

from my_models.grippers import TetrapackGripper

from robosuite.models.objects.composite import HammerObject, PotWithHandlesObject
from robosuite.models.objects.composite_body import HingedBoxObject, ContainerWithBox
from robosuite.utils.mjcf_utils import new_joint

register_env(Pressfit)
register_gripper(TetrapackGripper)

mujoco_robot = Panda()

gripper =TetrapackGripper()
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

mujoco_arena = Pressfit(robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types="TetrapackGripper",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,)
world.merge(mujoco_arena)

model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()