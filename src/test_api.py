from turtle import pos
from my_models.objects import  ContainerWithTetrapacksObject
from my_models.objects.xml_objects import SoftBoxObject
from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()

from robosuite.models.robots import Panda

mujoco_robot = Panda()

from my_models.grippers import TetrapackGripper

gripper = TetrapackGripper()
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

from robosuite.models.objects.composite import HammerObject, PotWithHandlesObject
from robosuite.models.objects.composite_body import HingedBoxObject, ContainerWithBox
from robosuite.utils.mjcf_utils import new_joint

# .sphere = ContainerWithTetrapacksObject(
#     name="container_with_tetrapacks")
# mujoco_obj= sphere.get_obj()
# mujoco_obj.set('pos', '1.0 0 1.0')
# mujoco_obj.set('quat', '0.707 0. 0. 0.707')
# world.merge_assets(sphere)
# worldworldbody.append(mujoco_obj)

box = SoftBoxObject(
    name="soft_box"
)
mujoco_obj2=box.get_obj()
# print(mujoco_obj2.bottom_offset)
mujoco_obj2.set('pos', '1.0 0. 1.0')
world.merge_assets(box)
world.worldbody.append(mujoco_obj2)

hammer = HammerObject(
    name="hammer"
)
mj_obj= hammer.get_obj()
mj_obj.set('pos', '2.0 0. 1.0')
world.merge_assets(hammer)
world.worldbody.append(mj_obj)

# hingedbox = HingedBoxObject(
#     name="hingedbox"
# )
# mj_obj2= hingedbox.get_obj()
# mj_obj2.set('pos', '0.5 0.5 1.0')
# world.merge_assets(hingedbox)
# world.worldbody.append(mj_obj2)

# hingedbox2 = ContainerWithBox(
#     name="container_with_box"
# )
# mj_obj3= hingedbox2.get_obj()
# mj_obj3.set('pos', '0.5 0.5 1.0')
# world.merge_assets(hingedbox2)
# world.worldbody.append(mj_obj3)

model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()