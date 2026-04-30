import torch

class NavigationTermination:
    def __init__(self, cfg, scene):
        self.cfg = cfg
        self.scene = scene

    def __call__(self, state, collision, step_count):
        done = torch.tensor(False, device=state.device)

        if self.cfg.use_goal:
            dist = torch.norm(state[:2] - self.cfg.goal)
            done |= dist < self.cfg.goal_tolerance

        if self.cfg.use_collision and collision is not None:
            done |= collision.bool()

        if self.cfg.use_out_of_bounds:
            x, y = state[0], state[1]
            done |= (
                (x < self.scene.map.x_lim[0]) |
                (x > self.scene.map.x_lim[1]) |
                (y < self.scene.map.y_lim[0]) |
                (y > self.scene.map.y_lim[1])
            )

        if self.cfg.use_timeout:
            done |= step_count >= self.cfg.max_steps

        return done