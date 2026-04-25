from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import torch
from torch.distributed.distributed_c10d import dist_config

from source.tasks.navigation.static_env.registry.static_env_registry import TASK_REGISTRY


class MPPIEnv:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        self.dt = cfg.dt
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim

        self.u_min = cfg.u_min
        self.u_max = cfg.u_max

        # --- components (determined by cfg)---
        task = TASK_REGISTRY[cfg.task.name]
        self.scene = task["scene"](
            cfg.scene,
            device=self.device,
            dtype=self.dtype,
        )

        self.collision_checker = self.scene.collision_checker

        self.dynamics_model = task["dynamics"](cfg.dynamics)

        self.cost = task["cost"](
            cfg.cost,
            goal=self.scene.goal,
            collision_checker=self.scene.collision_checker,
        )

        self.termination = task["termination"](
            cfg.terminations,
            scene=self.scene,
        )

        self.renderer = task["renderer"](self)


        # state
        self.state: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.extras: Dict[str, Any] = {}

        print("[INFO] MPPIEnv initialized")

    # dynamics interface (possible to override)
    def dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.dynamics_model is None:
            raise NotImplementedError("Dynamics model not set")

        return self.dynamics_model.step(state, action, self.dt)


    # reset
    def reset(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.scene is not None and hasattr(self.scene, "reset"):
            self.scene.reset()

        if state is None:
            state = self.cfg.init_state.clone().to(self.device)

        self.state = state.clone()
        self.step_count = 0
        return self.state


    # single step
    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        if self.state is None:
            raise RuntimeError("Env must be reset before step().")

        action = torch.clamp(action, self.u_min, self.u_max)

        next_state = self.dynamics(
            self.state.unsqueeze(0),
            action.unsqueeze(0)
        ).squeeze(0)

        self.state = next_state
        self.step_count += 1

        collision = None
        if self.collision_checker is not None:
            collision = self.collision_checker.check_point(
                next_state.unsqueeze(0)
            ).squeeze(0)


        done = self.termination(next_state, collision, self.step_count)

        return next_state, done, self.extras


    def check_trajectory_collision(self, states: torch.Tensor):
        """
        Args:
            states: (B, T, state_dim)
        Returns:
            collision: (B, T) or (B,)
        """

        if self.collision_checker is None:
            return None

        B, T, _ = states.shape

        # flatten for vectorized check
        flat_states = states.reshape(B * T, -1)

        collision = self.collision_checker.check_point(flat_states)

        return collision.reshape(B, T)

    def cost_function(self, state, action, info):
        return self.cost.compute_cost(state, action, info)

    def render(self, *args, **kwargs):
        if self.renderer is not None:
            return self.renderer.render(*args, **kwargs)

    def close(self):
        pass
