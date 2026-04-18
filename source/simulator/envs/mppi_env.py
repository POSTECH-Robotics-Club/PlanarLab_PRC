# source/envs/navigation_env.py

import torch

from source.scene.navigation_scene import NavigationScene
from source.sim.dynamics.car_dynamics import CarDynamics
from source.tasks.obstacle.static_env.static_env_cfg import NavigationEnvCfg


class MPPIEnv:
    """
    Low-level navigation environment.

    Responsibilities:
    - Maintain simulation state
    - Handle dynamics propagation
    - Own scene representation (ObstacleMap)
    """

    def __init__(self, cfg: NavigationEnvCfg):
        self.cfg = cfg

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------
        # Scene setup
        # -------------------------

        self.scene = NavigationScene(cfg.scene, self.device)

        # -------------------------
        # Dynamics 
        # -------------------------
        self.dt = cfg.sim.dt
        self.dynamics = CarDynamics(cfg.dynamics)

        # -------------------------
        # State
        # -------------------------
        self.state_dim = 6
        self.state = torch.zeros(self.state_dim, device=self.device)

        # -------------------------
        # Goal
        # -------------------------
        self.goal = torch.tensor(self.scene.goal, device=self.device)

    # -------------------------
    # Core API
    # -------------------------
    def reset(self) -> torch.Tensor:
        """Reset environment state"""
        self.state[:] = 0.0
        return self.state.clone()

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamics step.

        Args:
            action: (2,) tensor

        Returns:
            next_state: (6,)
        """
        # Add batch dimension
        state_batch = self.state.unsqueeze(0)      # (1, 6)
        action_batch = action.unsqueeze(0)         # (1, 2)

        # Use CarDynamics.step()
        next_state = self.dynamics.step(
            state_batch,
            action_batch,
            self.dt,
        )

        # Remove batch dimension
        self.state = next_state.squeeze(0)

        return self.state.clone()

    # -------------------------
    # Utility
    # -------------------------
    def get_state(self) -> torch.Tensor:
        return self.state

    def set_state(self, state: torch.Tensor):
        self.state = state.to(self.device)

    def dynamics_fn(self, state, action):
        return self.dynamics.step(state, action, self.cfg.sim.dt)