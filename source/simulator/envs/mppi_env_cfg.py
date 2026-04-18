from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import torch


@dataclass
class MPPIEnvCfg:
    """
    Generic MPPI config (env + controller runtime slots)
    """

    # =========================
    # Env
    # =========================
    dt: float = 0.1
    max_steps: int = 200

    state_dim: int = 6
    action_dim: int = 2

    init_state: torch.Tensor = field(
        default_factory=lambda: torch.zeros(6)
    )

    goal: Optional[torch.Tensor] = None
    goal_tolerance: float = 0.2

    # =========================
    # MPPI
    # =========================
    horizon: int = 20
    num_samples: int = 1000

    lambda_: float = 1.0
    auto_lambda: bool = False
    exploration: float = 0.0

    sigmas: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.3, 0.3])
    )

    # =========================
    # Control limits
    # =========================
    u_min: Optional[torch.Tensor] = None
    u_max: Optional[torch.Tensor] = None

    # =========================
    # Runtime
    # =========================
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    seed: int = 42

    dynamics_model: Optional[Any] = None
    cost_func: Optional[Callable] = None
    collision_checker: Optional[Any] = None
    render_fn: Optional[Callable] = None

    # =========================
    # Post init
    # =========================
    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu" 
        
        self.init_state = self.init_state.to(self.device)
        self.sigmas = self.sigmas.to(self.device)

        if self.goal is not None:
            self.goal = self.goal.to(self.device)

    # =========================
    # Validation
    # =========================
    def validate(self):
        assert self.dynamics_model is not None
        assert self.cost_func is not None

        assert self.u_min is not None
        assert self.u_max is not None

        assert self.u_min.shape == (self.action_dim,)
        assert self.u_max.shape == (self.action_dim,)