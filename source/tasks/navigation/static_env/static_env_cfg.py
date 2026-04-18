from dataclasses import dataclass, field
from typing import Tuple


# -------------------------------------------------
# Robot
# -------------------------------------------------
@dataclass
class RobotCfg:
    start_pos: Tuple[float, float] = (-10.0, -10.0)

    u_min: Tuple[float, float] = (-2.0, -1.0)
    u_max: Tuple[float, float] = (2.0, 1.0)

    wheel_base: float = 0.28


# -------------------------------------------------
# Obstacles
# -------------------------------------------------
@dataclass
class ObstacleCfg:
    map_size: Tuple[int, int] = (100, 100)
    cell_size: float = 0.1

    detect_range: float = 4.0

    num_circle_obs: int = 10
    num_rectangle_obs: int = 5

    circle_radius_range: Tuple[float, float] = (2.0, 5.0)
    rectangle_size_range: Tuple[float, float] = (3.0, 8.0)

    seed: int = 42


# -------------------------------------------------
# Scene
# -------------------------------------------------
@dataclass
class SceneCfg:
    robot: RobotCfg = field(default_factory=RobotCfg)
    obstacle: ObstacleCfg = field(default_factory=ObstacleCfg)

    goal_pos: Tuple[float, float] = (10.0, 10.0)


# -------------------------------------------------
# Dynamics 
# -------------------------------------------------
@dataclass
class DynamicsCfg:
    m: float = 1.0
    lf: float = 0.5
    lr: float = 0.5
    Iz: float = 0.1
    kf: float = 1.0
    kr: float = 1.0

    u_min: Tuple[float, float] = (-2.0, -1.0)
    u_max: Tuple[float, float] = (2.0, 1.0)

    v_max: float = 2.0
    w_max: float = 2.0

    process_noise_std: float = 0.0


# -------------------------------------------------
# Simulation
# -------------------------------------------------
@dataclass
class SimCfg:
    dt: float = 0.1


# -------------------------------------------------
# Top-level Env Config
# -------------------------------------------------
@dataclass
class NavigationEnvCfg:
    scene: SceneCfg = field(default_factory=SceneCfg)
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)
    sim: SimCfg = field(default_factory=SimCfg)