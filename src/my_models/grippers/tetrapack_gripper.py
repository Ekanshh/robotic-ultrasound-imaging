import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel

class TetrapackGripper(GripperModel):
    """
    Ultrasound Probe Gripper.
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__("my_models/assets/grippers/tetrapack_gripper.xml", idn=idn)

    def format_action(self, action):
        return action

    def _get_composite_element(self):
        return self._obj.find("./composite")

    def set_damping(self, damping):
        """
        Helper function to override the soft body's damping directly in the XML
        Args:
            damping (float, must be greater than zero): damping parameter to override the ones specified in the XML
        """
        assert damping > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        stiffness = float(solref_str[0])

        solref = np.array([stiffness, -damping])
        composite.set('solrefsmooth', array_to_string(solref))


    def set_stiffness(self, stiffness):
        """
        Helper function to override the soft body's stiffness directly in the XML
        Args:
            stiffness (float, must be greater than zero): stiffness parameter to override the ones specified in the XML
        """
        assert stiffness > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        damping = float(solref_str[1])

        solref = np.array([-stiffness, damping])
        composite.set('solrefsmooth', array_to_string(solref))
    @property
    def init_qpos(self):
        return None

    @property
    def _important_geoms(self):
        return {
            "probe": ["probe_collision"]
        }