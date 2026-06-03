"""robosuite version-compatibility flag.

robosuite 1.5 reworked the env, robot, controller and mount/base APIs that
LIBERO builds on. This flag lets the env code support both 1.4 and >= 1.5 from a
single source. Feature-detected (no version-string parsing): the
``robosuite.robots.single_arm`` module exists only on robosuite <= 1.4.
"""

try:
    import robosuite.robots.single_arm  # noqa: F401  (removed in robosuite >= 1.5)

    ROBOSUITE_GE_15 = False
except ImportError:
    ROBOSUITE_GE_15 = True
