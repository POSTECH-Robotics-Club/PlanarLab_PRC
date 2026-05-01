import torch
from typing import Dict


class NavigationCost:
    """
    Cost function for MPPI.
    Pure cost computation only.
    """

    def __init__(self, cfg, goal=None, collision_checker=None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.goal = goal if goal is not None else torch.tensor([50.0, 50.0])
        self.goal = self.goal.to(self.device)

        self.w_goal = cfg.goal_weight
        self.w_vel = cfg.vel_weight
        self.control_weight = cfg.control_weight
        self.collision_weight = cfg.collision_weight

        self.target_v = torch.tensor(cfg.target_v, device=self.device)

        self.collision_checker = collision_checker

    def compute_cost(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        info: Dict | None = None,
    ) -> torch.Tensor:
        if state.device != self.goal.device:
            goal = self.goal.to(state.device)
            target_v = self.target_v.to(state.device)
        else:
            goal = self.goal
            target_v = self.target_v


        pos = state[..., :2]

        # state = [x, y, theta, u, v, w]
        u = state[..., 3]
        v = state[..., 4]
        vel = torch.sqrt(u**2 + v**2 + 1e-6)

        goal_cost = torch.norm(pos - goal, dim=-1)

        effective_target_v = target_v * (1 - torch.exp(-3 * goal_cost))
        vel_cost = (vel - effective_target_v)**2

        cost = self.w_goal * goal_cost + self.w_vel * vel_cost

        # regulize the control action u
        # if action is not None:
        #     control_cost = torch.sum(action**2, dim=-1)
        #     cost = cost + self.control_weight * control_cost

        
        collision = None
        if self.collision_checker is not None:
            collision = self.collision_checker.check_point(state)

        if collision is not None:
            cost = cost + self.collision_weight * collision.float()
    
        return cost.view(-1)

    def __call__(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        info: dict | None = None,
    ) -> torch.Tensor:
        collision = None
        if info is not None:
            collision = info.get("collision", None)

        return self.compute_cost(
            state=state,
            action=action,
            info=info,
        )
