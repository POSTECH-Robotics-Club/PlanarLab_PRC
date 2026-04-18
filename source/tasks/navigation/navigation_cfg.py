from dataclasses import dataclass, field
import torch

from source.simulator.envs.mppi_env_cfg import MPPIEnvCfg


# =========================
# Cost
# =========================
@dataclass
class CostCfg:
    target_v: float = 8.0

    goal_weight: float = 1.0
    vel_weight: float = 1.0
    control_weight: float = 0.1
    collision_weight: float = 5.0

    goal_tolerance: float = 0.2


# =========================
# Termination
# =========================
@dataclass
class TerminationCfg:
    use_goal: bool = True
    use_collision: bool = True
    use_out_of_bounds: bool = True
    use_timeout: bool = True

    max_steps: int = 200
    goal_tolerance: float = 0.2


# =========================
# Base Navigation MPPI Config
# =========================
@dataclass
class NavigationMPPIEnvCfg(MPPIEnvCfg):
    """
    Base config for navigation + MPPI.
    Concrete env configs (e.g. static_env) should inherit from this.
    """

    cost: CostCfg = field(default_factory=CostCfg)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)

    horizon: int = 30
    num_samples: int = 1500

    collision_checker: object | None = None

    def __post_init__(self):
        super().__post_init__()

        # navigation common defaults
        self.state_dim = 6
        self.action_dim = 2
        self.max_steps = self.terminations.max_steps