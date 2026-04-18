from dataclasses import dataclass, field
from typing import Tuple
import torch

from source.tasks.navigation.navigation_cfg import NavigationMPPIEnvCfg
from source.simulator.dynamics.dynamics.car_dynamics import CarDynamics
from source.tasks.navigation.static_env.mdp.cost import NavigationCost
from source.simulator.dynamics.collision.collision_checker import CollisionChecker


# -------------------------------------------------
# Robot
# -------------------------------------------------
@dataclass
class RobotCfg:
    start_pos: Tuple[float, float] = (-10.0, -10.0)

    u_min: Tuple[float, float] = (-2.0, -1.0)
    u_max: Tuple[float, float] = (2.0, 1.0)

    wheel_base: float = 0.28
    radius: float = 0.5


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
# Static Navigation MPPI Config
# -------------------------------------------------
@dataclass
class NavigationStaticMPPIEnvCfg(NavigationMPPIEnvCfg):
    """
    Static navigation config.
    Inherits common navigation MPPI settings from NavigationMPPIEnvCfg.
    """

    scene: SceneCfg = field(default_factory=SceneCfg)
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)
    sim: SimCfg = field(default_factory=SimCfg)

    def __post_init__(self):
        super().__post_init__()

        device = torch.device(self.device)
        robot = self.scene.robot

        # -------------------------
        # sync static env params
        # -------------------------
        self.dt = self.sim.dt
        self.max_steps = self.terminations.max_steps

        # -------------------------
        # control limits
        # -------------------------
        self.u_min = torch.tensor(
            robot.u_min,
            dtype=self.dtype,
            device=device,
        )
        self.u_max = torch.tensor(
            robot.u_max,
            dtype=self.dtype,
            device=device,
        )

        assert self.u_min.shape == (self.action_dim,)
        assert self.u_max.shape == (self.action_dim,)

        # -------------------------
        # goal
        # -------------------------
        self.goal = torch.tensor(
            self.scene.goal_pos,
            dtype=self.dtype,
            device=device,
        )

        # -------------------------
        # init state
        # state = [x, y, theta, u, v, w]
        # -------------------------
        self.init_state = torch.zeros(
            self.state_dim,
            dtype=self.dtype,
            device=device,
        )
        self.init_state[0:2] = torch.tensor(
            robot.start_pos,
            dtype=self.dtype,
            device=device,
        )

        # -------------------------
        # runtime bindings
        # -------------------------
        self.dynamics_model = CarDynamics(self.dynamics)

        self.cost_func = NavigationCost(
            cfg=self.cost,
            goal=self.goal,
        )

        self.validate()