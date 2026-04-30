from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch

from source.tasks.navigation.navigation_cfg import NavigationMPPIEnvCfg


@dataclass
class TaskCfg:
    name: str = "navigation_static"


# Robot
@dataclass
class RobotCfg:
    start_pos: Tuple[float, float] = (-45.0, -45.0)
    goal_pos: Tuple[float, float] = (45.0, 45.0)   # map_size/2 - 15

    u_min: Tuple[float, float] = (-10.0, -0.2)   
    u_max: Tuple[float, float] = (10.0, 0.2)

    wheel_base: float = 0.28
    radius: float = 1.0


# Obstacles
@dataclass
class ObstacleCfg:
    map_size: Tuple[int, int] = (100, 100)
    cell_size: float = 0.1

    detect_range: float = 20.0

    num_circle_obs: int = 20
    num_rectangle_obs: int = 21

    circle_radius_range: Tuple[float, float] = (2.0, 5.0)
    rectangle_size_range: Tuple[float, float] = (3.0, 8.0)

    seed: int = 42


# Scene
@dataclass
class SceneCfg:
    robot: RobotCfg = field(default_factory=RobotCfg)
    obstacle: ObstacleCfg = field(default_factory=ObstacleCfg)



# Dynamics
@dataclass
class DynamicsCfg:
    m: float = 1.5
    lf: float = 0.13
    lr: float = 0.15
    Iz: float = 0.014
    kf: float = -120.0
    kr: float = -140.0

    u_min: Optional[Tuple[float, float]] = None
    u_max: Optional[Tuple[float, float]] = None

    v_max: float = 30.0
    w_max: float = 1.0

    process_noise_std: float = 0.0

    x_lim: Tuple[float, float] = (-50.0, 50.0)
    y_lim: Tuple[float, float] = (-50.0, 50.0)


# Simulation
@dataclass
class SimCfg:
    dt: float = 0.1

@dataclass
class RendererCfg:
    enable: bool = True

@dataclass
class TerminationCfg:
    use_goal: bool = True
    use_collision: bool = True
    use_out_of_bounds: bool = True
    use_timeout: bool = True

    max_steps: int = 500
    goal_tolerance: float = 0.5

    goal: Optional[torch.Tensor] = None  


# Static Navigation MPPI Config
@dataclass
class NavigationStaticMPPIEnvCfg(NavigationMPPIEnvCfg):
    """
    Static navigation config.
    Inherits common navigation MPPI settings from NavigationMPPIEnvCfg.
    """
    task: TaskCfg= field(default_factory=TaskCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)
    sim: SimCfg = field(default_factory=SimCfg)

    renderer: RendererCfg = field(default_factory=RendererCfg)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)

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
        self.dynamics.u_min = robot.u_min
        self.dynamics.u_max = robot.u_max

        assert self.u_min.shape == (self.action_dim,)
        assert self.u_max.shape == (self.action_dim,)

        # goal
        self.goal = torch.tensor(
            self.scene.robot.goal_pos,
            dtype=self.dtype,
            device=device,
        )

        self.terminations.goal = self.goal

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
        self.dynamics_model = None

        self.cost_func = None


        self.validate()