import torch

from source.simulator.scene.obstacle_map import ObstacleMap
from source.simulator.scene.obstacle_generator import generate_random_obstacles
from source.simulator.dynamics.collision.collision_checker import CollisionChecker


class ObstacleScene:
    """
    Obstacle scene runtime object.

    Responsibilities:
    - define world geometry
    - generate obstacles
    - store goal
    - provide collision checker
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        scene_cfg = cfg.scene
        obs_cfg = scene_cfg.obstacle
        robot_cfg = scene_cfg.robot

        # -------------------------
        # Map setup
        # -------------------------
        self.map = ObstacleMap(
            map_size=obs_cfg.map_size,
            cell_size=obs_cfg.cell_size,
            device=device,
            dtype=cfg.dtype,
        )

        # -------------------------
        # Obstacle generation
        # -------------------------
        x_half = obs_cfg.map_size[0] / 2.0
        y_half = obs_cfg.map_size[1] / 2.0

        generate_random_obstacles(
            obstacle_map=self.map,
            random_x_range=(-x_half, x_half),
            random_y_range=(-y_half, y_half),
            num_circle_obs=obs_cfg.num_circle_obs,
            radius_range=obs_cfg.circle_radius_range,
            num_rectangle_obs=obs_cfg.num_rectangle_obs,
            width_range=obs_cfg.rectangle_size_range,
            height_range=obs_cfg.rectangle_size_range,
            max_iteration=100,
            seed=obs_cfg.seed,
        )

        self.map.convert_to_torch()

        # -------------------------
        # Goal
        # -------------------------
        self.goal = torch.tensor(
            scene_cfg.goal_pos,
            device=device,
            dtype=cfg.dtype,
        )

        # -------------------------
        # Collision checker
        # -------------------------
        self.collision_checker = CollisionChecker(
            obstacle_map=self.map,
            robot_radius=robot_cfg.radius,
            detect_range=obs_cfg.detect_range,
        )

    def reset(self):
        obs_cfg = self.cfg.scene.obstacle

        self.map.clear_obstacles()

        x_half = obs_cfg.map_size[0] / 2.0
        y_half = obs_cfg.map_size[1] / 2.0

        generate_random_obstacles(
            obstacle_map=self.map,
            random_x_range=(-x_half, x_half),
            random_y_range=(-y_half, y_half),
            num_circle_obs=obs_cfg.num_circle_obs,
            radius_range=obs_cfg.circle_radius_range,
            num_rectangle_obs=obs_cfg.num_rectangle_obs,
            width_range=obs_cfg.rectangle_size_range,
            height_range=obs_cfg.rectangle_size_range,
            max_iteration=100,
            seed=obs_cfg.seed,
        )

        self.map.convert_to_torch()