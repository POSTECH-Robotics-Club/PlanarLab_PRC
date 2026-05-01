from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch

from source.tasks.navigation.navigation_cfg import NavigationMPPIEnvCfg



# Task
@dataclass
class TaskCfg:
    name: str = "navigation_track"



# Track / Map
@dataclass
class TrackCfg:
    map_size: Tuple[int, int] = (80, 80)
    cell_size: float = 0.01

    detect_range: float = 20.0

    lane_width: float = 8

    csv_path: str = str("simulator/scene/track_generator/path.csv")



# Obstacles
@dataclass
class ObstacleCfg:
    map_size: Tuple[int, int] = (80, 80)
    cell_size: float = 0.1

    num_circle_obs: int = 20
    num_rectangle_obs: int = 0

    circle_radius_range: Tuple[float, float] = (0.5, 1.5)
    rectangle_size_range: Tuple[float, float] = (1.5, 2.5)

    detect_range: float = 20.0
    speed_range: tuple[float, float] = (0.0, 0.0)

    seed: int = 42



# Robot / Control
@dataclass
class RobotCfg:
    # control: [accel, steer]
    u_min: Tuple[float, float] = (-5.0, -0.2)
    u_max: Tuple[float, float] = (5.0, 0.2)

    wheel_base: float = 0.28
    v_max: float = 10.0

    target_v: float = 8.0

    start_pos: Tuple[float, float] = (-28.0, -29.5)

    radius: float = 1.0



# Dynamics
@dataclass
class DynamicsCfg:
    dt: float = 0.1

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

    x_lim: Tuple[float, float] = (-50.0, 50.0)
    y_lim: Tuple[float, float] = (-50.0, 50.0)

# Track scene
@dataclass
class SceneCfg:
    robot: RobotCfg = field(default_factory=RobotCfg)
    obstacle: ObstacleCfg = field(default_factory=ObstacleCfg)
    track: TrackCfg = field(default_factory=TrackCfg)

    goal_pos: Tuple[float, float] = (-28.0, 26.0)



# Termination
@dataclass
class TerminationCfg:
    max_steps: int = 500
    goal_tolerance: float = 1.0

    use_goal: bool = True
    use_collision: bool = True
    use_timeout: bool = True

    goal: Optional[torch.Tensor] = None



# Renderer
@dataclass
class RendererCfg:
    enable: bool = True



# Main Config
@dataclass
class Navigation2DTrackMPPIEnvCfg(NavigationMPPIEnvCfg):
    task: TaskCfg = field(default_factory=TaskCfg)

    scene: SceneCfg = field(default_factory=SceneCfg)
    track: TrackCfg = field(default_factory=TrackCfg)
    obstacle: ObstacleCfg = field(default_factory=ObstacleCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)

    renderer: RendererCfg = field(default_factory=RendererCfg)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)

    def __post_init__(self):
        super().__post_init__()

        device = torch.device(self.device)

        # basic sync
        self.dt = self.dynamics.dt
        self.max_steps = self.terminations.max_steps

        # control limits
        self.u_min = torch.tensor(
            self.robot.u_min, dtype=self.dtype, device=device
        )
        self.u_max = torch.tensor(
            self.robot.u_max, dtype=self.dtype, device=device
        )

        assert self.u_min.shape == (self.action_dim,)
        assert self.u_max.shape == (self.action_dim,)

        # velocity
        self.v_max = torch.tensor(
            self.robot.v_max, dtype=self.dtype, device=device
        )

        self.target_v = torch.tensor(
            self.robot.target_v, dtype=self.dtype, device=device
        )

        # goal
        self.goal = torch.tensor(
            self.scene.goal_pos,
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
            self.robot.start_pos,
            dtype=self.dtype,
            device=device,
        )


        self.dynamics_model = None
        self.cost_func = None

        self.validate()