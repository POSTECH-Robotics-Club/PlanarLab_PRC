from dataclasses import dataclass, field

from source.simulator.envs.mppi_env_cfg import MPPIEnvCfg
from source.tasks.navigation.static_env.mdp.cost import NavigationCost



# Cost (fine tuned)
@dataclass
class CostCfg:
    class_type = NavigationCost

    target_v: float = 8.0

    goal_weight: float = 80.0
    vel_weight: float = 15.0
    control_weight: float = 0.1
    collision_weight: float = 1500.0



# Base Navigation MPPI Config - more functions will be added
@dataclass
class NavigationMPPIEnvCfg(MPPIEnvCfg):
    """
    Base config for navigation + MPPI.
    Concrete env configs (e.g. static_env) should inherit from this.
    """

    cost: CostCfg = field(default_factory=CostCfg)
    # terminations: TerminationCfg = field(default_factory=TerminationCfg)

    horizon: int = 20
    num_samples: int = 3000

    def __post_init__(self):
        super().__post_init__()

        # navigation common defaults
        self.state_dim = 6
        self.action_dim = 2
