from dataclasses import dataclass, field
from typing import Tuple
import torch

from source.tasks.navigation.navigation_cfg import NavigationMPPIEnvCfg
from source.simulator.dynamics.dynamics.car_dynamics import CarDynamics
from source.tasks.navigation.static_env.mdp.cost import NavigationCost



# Robot
@dataclass
class RobotCfg:
    start_pos: Tuple[float, float] = (-40.0, -40.0)

    u_min: Tuple[float, float] = (-10.0, -0.2)    # why u_min, u_max are also in DynamicsCfg?
    u_max: Tuple[float, float] = (10.0, 0.2)

    wheel_base: float = 0.28
    radius: float = 0.5


# Obstacles
@dataclass
class ObstacleCfg:
    map_size: Tuple[int, int] = (120, 120)
    cell_size: float = 0.1

    detect_range: float = 20.0

    num_circle_obs: int = 18
    num_rectangle_obs: int = 20

    circle_radius_range: Tuple[float, float] = (2.0, 5.0)
    rectangle_size_range: Tuple[float, float] = (3.0, 8.0)

    seed: int = 42


# Scene
@dataclass
class SceneCfg:
    robot: RobotCfg = field(default_factory=RobotCfg)
    obstacle: ObstacleCfg = field(default_factory=ObstacleCfg)

    goal_pos: Tuple[float, float] = (40.0, 40.0)  # map_size/2 - 20


# Dynamics
@dataclass
class DynamicsCfg:
    m: float = 1.5
    lf: float = 0.13
    lr: float = 0.15
    Iz: float = 0.014
    kf: float = -120.0
    kr: float = -140.0

    u_min: Tuple[float, float] = (-10.0, -0.2)
    u_max: Tuple[float, float] = (10.0, 0.2)

    v_max: float = 30.0
    w_max: float = 1.0

    process_noise_std: float = 0.0


# Simulation
@dataclass
class SimCfg:
    dt: float = 0.1


# Static Navigation MPPI Config
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

        # sync static env params
        self.dt = self.sim.dt
        self.max_steps = self.terminations.max_steps

        # control limits
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

        # goal
        self.goal = torch.tensor(
            self.scene.goal_pos,
            dtype=self.dtype,
            device=device,
        )

        # init state
        # state = [x, y, theta, u, v, w]
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

        # runtime bindings
        self.dynamics_model = CarDynamics(self.dynamics)

        self.cost_func = None


        self.validate()