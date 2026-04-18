import torch


class NavigationCost:
    """
    Cost function for MPPI / TDMPC.

    NOTE:
    - NO simulation logic here
    - PURE function only
    """

    def __init__(self, cfg, goal=None):
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # default goal (move to correct device once)
        self.goal = goal if goal is not None else torch.tensor([50.0, 50.0])
        self.goal = self.goal.to(self.device)

        self.w_goal = 2.0
        self.w_vel = 1.0
        self.w_collision = 1000.0

        self.target_v = torch.tensor(cfg.target_v, device=self.device)

    # -------------------------
    # core cost
    # -------------------------
    def __call__(self, state: torch.Tensor, collision: torch.Tensor = None):
        """
        state: (B, 6) or (B, H, 6)
        collision: (B,) or (B, H)
        """

        # ensure device consistency
        if state.device != self.goal.device:
            goal = self.goal.to(state.device)
            target_v = self.target_v.to(state.device)
        else:
            goal = self.goal
            target_v = self.target_v

        pos = state[..., :2]

        # more accurate velocity (u, v)
        u = state[..., 3]
        v = state[..., 4]
        vel = torch.sqrt(u**2 + v**2 + 1e-6)

        # goal distance cost
        goal_cost = torch.norm(pos - goal, dim=-1)

        # velocity tracking cost
        vel_cost = (vel - target_v) ** 2

        cost = self.w_goal * goal_cost + self.w_vel * vel_cost

        # collision penalty
        if collision is not None:
            cost = cost + self.w_collision * collision.float()

        return cost

    # -------------------------
    # trajectory cost reduction
    # -------------------------
    def trajectory_cost(self, states: torch.Tensor, collisions: torch.Tensor = None):
        """
        states: (K, H, 6)
        collisions: (K, H)
        """

        # reuse core cost
        total = self(states, collisions)

        # sum over horizon
        return total.sum(dim=-1)