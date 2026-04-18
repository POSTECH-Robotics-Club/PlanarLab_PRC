import torch


class NavigationCost:
    """
    Cost function for MPPI.
    Pure cost computation only.
    """

    def __init__(self, cfg, goal=None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.goal = goal if goal is not None else torch.tensor([50.0, 50.0])
        self.goal = self.goal.to(self.device)

        self.w_goal = cfg.goal_weight
        self.w_vel = cfg.vel_weight
        self.w_control = cfg.control_weight
        self.w_collision = cfg.collision_weight

        self.target_v = torch.tensor(cfg.target_v, device=self.device)

    def compute_cost(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        collision: torch.Tensor | None = None,
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
        vel_cost = (vel - target_v) ** 2

        cost = self.w_goal * goal_cost + self.w_vel * vel_cost

        if action is not None:
            control_cost = torch.sum(action**2, dim=-1)
            cost = cost + self.w_control * control_cost

        if collision is not None:
            cost = cost + self.w_collision * collision.float()

        return cost

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
            collision=collision,
        )

    def compute_trajectory_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        collisions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        step_costs = self.compute_cost(
            state=states,
            action=actions,
            collision=collisions,
        )
        return step_costs.sum(dim=-1)