from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import torch

from .mppi_env_cfg import MPPIEnvCfg


class MPPIEnv:
    def __init__(self, cfg: MPPIEnvCfg):
        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        if cfg.dynamics_model is None:
            raise ValueError("cfg.dynamics_model must not be None")
        self._dynamics_model = cfg.dynamics_model

        self.dt = cfg.dt

        # if cfg.cost_func is None:
        #     raise ValueError("cfg.cost_func must not be None")
        self._cost_func = cfg.cost_func

        self.u_min = cfg.u_min
        self.u_max = cfg.u_max

        # manager-based style runtime scene
        self.scene: Any = None
        self.collision_checker: Any = None
        self.renderer: Any = None

        self.state: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.extras: Dict[str, Any] = {}

        print("[INFO] MPPIEnv initialized")

    def reset(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.scene is not None and hasattr(self.scene, "reset"):
            self.scene.reset()

        if state is None:
            print("state initialized")
            state = self.cfg.init_state.clone().to(self.device)

        self.state = state.clone()
        self.step_count = 0
        return self.state

    def dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._dynamics_model.step(state, action, self.dt)

    def cost_function(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        info: Dict,
    ) -> torch.Tensor:
        return self._cost_func(state, action, info)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        if self.state is None:
            raise RuntimeError("Env must be reset before step().")

        if self.u_min is not None and self.u_max is not None:
            action = torch.clamp(action, self.u_min, self.u_max)

        next_state = self.dynamics(
            self.state.unsqueeze(0),
            action.unsqueeze(0),
        ).squeeze(0)

        collision = None
        if self.collision_checker is not None:
            collision = self.collision_checker.check_point(
                next_state.unsqueeze(0)
            ).squeeze(0)

        # # single step cost calculation
        # cost = self.cost_function(
        #     next_state.unsqueeze(0),
        #     action.unsqueeze(0),
        #     {
        #         "env": self,
        #         "collision": collision,
        #     },
        # ).squeeze(0)

        done = self._compute_done(next_state, collision)

        self.state = next_state
        self.step_count += 1

        return next_state, done, self.extras

    def _compute_done(
        self,
        state: torch.Tensor,
        collision: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.cfg.terminations
        done = torch.tensor(False, device=self.device)

        if cfg.use_goal:
            goal = self.cfg.goal
            dist = torch.norm(state[:2] - goal)
            done |= dist < cfg.goal_tolerance

        if cfg.use_collision and collision is not None:
            done |= collision.bool()

        if cfg.use_out_of_bounds and self.scene is not None:
            x, y = state[0], state[1]
            done |= (
                (x < self.scene.map.x_lim[0]) |
                (x > self.scene.map.x_lim[1]) |
                (y < self.scene.map.y_lim[0]) |
                (y > self.scene.map.y_lim[1])
            )

        if cfg.use_timeout:
            done |= self.step_count >= cfg.max_steps

        return done

    def check_trajectory_collision(self, states: torch.Tensor):
        if self.collision_checker is None:
            return None
        return self.collision_checker.check_trajectory(states)

    def render(self, *args, **kwargs):
        if hasattr(self, "renderer") and self.renderer is not None:
            return self.renderer.render(*args, **kwargs)
        return None

    def close(self):
        pass