import torch


class TrackTermination:
    def __init__(self, cfg, scene):
        self.cfg = cfg
        self.scene = scene

    def __call__(self, state: torch.Tensor, collision, step_count: int):
        device = state.device

        if state.dim() == 1:
            state = state.unsqueeze(0)

        B = state.shape[0]
        done = torch.zeros(B, device=device, dtype=torch.bool)


        # goal
        if getattr(self.cfg, "use_goal", False):
            goal = self.scene.goal.to(device)
            dist = torch.norm(state[:, :2] - goal[:2], dim=-1)
            done |= dist < self.cfg.goal_tolerance


        # collision (obstacles)
        if getattr(self.cfg, "use_collision", False) and collision is not None:
            if not torch.is_tensor(collision):
                collision = torch.tensor(collision, device=device)

            collision = collision.to(device).bool()

            if collision.dim() == 0:
                collision = collision.expand(B)

            done |= collision


        # out of bounds
        if getattr(self.cfg, "use_out_of_bounds", False):
            x, y = state[:, 0], state[:, 1]

            x_min, x_max = self.scene.map.x_lim
            y_min, y_max = self.scene.map.y_lim

            oob = (
                (x < x_min) |
                (x > x_max) |
                (y < y_min) |
                (y > y_max)
            )
            done |= oob


        # lane violation (OFF TRACK)
        if getattr(self.cfg, "use_lane_violation", True):
            lane_map = self.scene.map._map_torch  # (H, W)

            # convert world → grid
            x_occ = (state[:, 0] / self.scene.map._cell_size
                     + self.scene.map._torch_cell_map_origin[0]).long()

            y_occ = (state[:, 1] / self.scene.map._cell_size
                     + self.scene.map._torch_cell_map_origin[1]).long()

            H, W = lane_map.shape

            # clamp
            x_occ_clamped = torch.clamp(x_occ, 0, H - 1)
            y_occ_clamped = torch.clamp(y_occ, 0, W - 1)

            # inside lane = 0, outside lane = 1 (너 lane map 구조 기준)
            off_lane = lane_map[x_occ_clamped, y_occ_clamped].bool()

            done |= off_lane


        # timeout
        if getattr(self.cfg, "use_timeout", False):
            done |= (step_count >= self.cfg.max_steps)

        return done[0] if done.shape[0] == 1 else done