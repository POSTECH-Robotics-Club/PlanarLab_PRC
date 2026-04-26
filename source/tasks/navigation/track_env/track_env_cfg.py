from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch

from source.tasks.navigation.navigation_cfg import NavigationMPPIEnvCfg


# -----------------
# Task
# -----------------
@dataclass
class TaskCfg:
    name: str = "navigation_2d_track"


# -----------------
# Track / Map
# -----------------
@dataclass
class TrackCfg:
    map_size: int = 100
    cell_size: float = 0.1

    detect_range: float = 20.0

    # lane
    lane_width: float = 6.5

    # path
    csv_path: str = "envs/track_env/track_generator/path.csv"


# -----------------
# Obstacles
# -----------------
@dataclass
class ObstacleCfg:
    num_circle_obs: int = 20
    num_rectangle_obs: int = 20

    circle_radius_range: Tuple[float, float] = (0.5, 1.5)

    rectangle_width_range: Tuple[float, float] = (1.5, 2.5)
    rectangle_height_range: Tuple[float, float] = (1.5, 2.5)

    seed: int = 42


# -----------------
# Robot / Control
# -----------------
@dataclass
class RobotCfg:
    # control: [accel, steer]
    u_min: Tuple[float, float] = (-5.0, -0.3)
    u_max: Tuple[float, float] = (5.0, 0.3)

    wheel_base: float = 1.0
    v_max: float = 10.0

    target_v: float = 5.0


# -----------------
# Dynamics
# -----------------
@dataclass
class DynamicsCfg:
    dt: float = 0.1


# -----------------
# Termination
# -----------------
@dataclass
class TerminationCfg:
    max_steps: int = 500
    goal_tolerance: float = 1.0

    use_goal: bool = True
    use_collision: bool = True
    use_timeout: bool = True

    goal: Optional[torch.Tensor] = None


# -----------------
# Renderer
# -----------------
@dataclass
class RendererCfg:
    enable: bool = True


# -----------------
# Main Config
# -----------------
@dataclass
class Navigation2DTrackMPPIEnvCfg(NavigationMPPIEnvCfg):
    task: TaskCfg = field(default_factory=TaskCfg)

    track: TrackCfg = field(default_factory=TrackCfg)
    obstacle: ObstacleCfg = field(default_factory=ObstacleCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)

    renderer: RendererCfg = field(default_factory=RendererCfg)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)

    def __post_init__(self):
        super().__post_init__()

        device = torch.device(self.device)

        # -----------------
        # basic sync
        # -----------------
        self.dt = self.dynamics.dt
        self.max_steps = self.terminations.max_steps

        # -----------------
        # control limits
        # -----------------
        self.u_min = torch.tensor(
            self.robot.u_min, dtype=self.dtype, device=device
        )
        self.u_max = torch.tensor(
            self.robot.u_max, dtype=self.dtype, device=device
        )

        assert self.u_min.shape == (self.action_dim,)
        assert self.u_max.shape == (self.action_dim,)

        # -----------------
        # velocity
        # -----------------
        self.v_max = torch.tensor(
            self.robot.v_max, dtype=self.dtype, device=device
        )

        self.target_v = torch.tensor(
            self.robot.target_v, dtype=self.dtype, device=device
        )

        # -----------------
        # placeholders (runtime binding)
        # -----------------
        self.goal = None
        self.init_state = None

        self.dynamics_model = None
        self.cost_func = None

        self.validate()