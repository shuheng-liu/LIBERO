from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

from libero.libero.envs._compat import ROBOSUITE_GE_15

if ROBOSUITE_GE_15:
    # robosuite >= 1.5 removed robots.single_arm.SingleArm; fixed-base arms use FixedBaseRobot.
    from robosuite.robots.fixed_base_robot import FixedBaseRobot as _FixedArmRobot
else:
    from robosuite.robots.single_arm import SingleArm as _FixedArmRobot
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": _FixedArmRobot,
        "OnTheGroundPanda": _FixedArmRobot,
    }
)
