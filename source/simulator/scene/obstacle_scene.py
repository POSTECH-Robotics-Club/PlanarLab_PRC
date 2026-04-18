# source/scene/navigation_scene.py

import torch

from source.scene.obstacle_map import ObstacleMap
from source.scene.obstacle_generator import generate_random_obstacles


class NavigationScene:
    """
    Navigation scene (Isaac Lab style)

    Responsibilities:
    - Define world geometry
    - Generate obstacles
    - Store goal
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # -------------------------
        # Map setup
        # -------------------------
        if isinstance(cfg.map_size, int):
            map_size = (cfg.map_size, cfg.map_size)
        else:
            map_size = cfg.map_size

        self.map = ObstacleMap(
            map_size=map_size,
            cell_size=cfg.cell_size,
            device=device,
        )

        # -------------------------
        # Obstacle generation
        # -------------------------
        generate_random_obstacles(
            self.map,
            cfg.x_range,
            cfg.y_range,
            cfg.num_circle_obs,
            cfg.circle_radius_range,
            cfg.num_rectangle_obs,
            cfg.rectangle_width_range,
            cfg.rectangle_height_range,
            cfg.max_attempts,
            cfg.seed,
        )

        self.map.convert_to_torch()

        # -------------------------
        # Goal
        # -------------------------
        self.goal = torch.tensor(cfg.goal, device=device)

    # (optional)
    def reset(self):
        """
        Regenerate obstacles if needed
        """
        self.map.clear_obstacles()

        generate_random_obstacles(
            self.map,
            self.cfg.x_range,
            self.cfg.y_range,
            self.cfg.num_circle_obs,
            self.cfg.circle_radius_range,
            self.cfg.num_rectangle_obs,
            self.cfg.rectangle_width_range,
            self.cfg.rectangle_height_range,
            self.cfg.max_attempts,
            self.cfg.seed,
        )

        self.map.convert_to_torch()