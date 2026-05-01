import torch
import numpy as np

from source.simulator.scene.obstacle_map import ObstacleMap
from source.simulator.scene.obstacle_generator import generate_random_obstacles
from source.simulator.dynamics.collision.collision_checker import CollisionChecker

from pathlib import Path

from source.simulator.scene.track_map import TrackMap
from source.simulator.scene.track_generator.path_generate import (
    make_side_lane,
    make_csv_simple_path,
)

@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

class TrackScene:
    """
    Track-based scene for navigation/racing tasks.

    Responsibilities:
    - lane / centerline definition
    - start / goal
    - obstacle generation on track
    - collision checking
    - sampling on lane
    """

    def __init__(self, cfg, device, dtype):
        self.device = device
        self.dtype = dtype
        self.cfg = cfg

        obs_cfg = cfg.obstacle
        robot_cfg = cfg.robot
        track_cfg = cfg.track

        self.goal = torch.tensor(
            cfg.goal_pos,
            device=device,
            dtype=dtype,
        )


        # path / lane
        BASE_DIR = Path(__file__).resolve().parents[2]

        centerline = make_csv_simple_path(BASE_DIR /track_cfg.csv_path)


        self.lane_width = track_cfg.lane_width

        right, left = make_side_lane(
            centerline,
            lane_width=self.lane_width,
        )

        self.right_lane = torch.tensor(right, device=device, dtype=dtype)
        self.left_lane = torch.tensor(left, device=device, dtype=dtype)

        self.centerline = torch.tensor(centerline, device=device, dtype=dtype)

        self.map = TrackMap(
            lane=centerline,
            right_lane = right,
            left_lane = left,
            lane_width=self.lane_width * 0.8,
            map_size=obs_cfg.map_size,
            cell_size=obs_cfg.cell_size,
            device=device,
            dtype=dtype,
        )


        self._generate_obstacles()


        # start / goal
        self.start_pos = self.centerline[0, :2].clone()
        self.goal_pos = self.centerline[-1, :2].clone()


        # robot state
        self.robot_state = torch.zeros(6, device=device, dtype=dtype)
        self.robot_state[:2] = self.start_pos

        direction = self.centerline[1, :2] - self.centerline[0, :2]
        self.robot_state[2] = angle_normalize(
            torch.atan2(
                self.centerline[2][1] - self.start_pos[1],
                self.centerline[2][0] - self.start_pos[0],
            )
        )


        # collision checker
        self.collision_checker = CollisionChecker(
            map=self.map,
            robot_radius=robot_cfg.radius,
            detect_range=obs_cfg.detect_range,
        )


    # reset
    def reset(self):
        self.map.clear_obstacles()
        self._generate_obstacles()

        self.collision_checker = CollisionChecker(
            map=self.map,
            robot_radius=self.cfg.robot.radius,
            detect_range=self.cfg.obstacle.detect_range,
        )

        self.robot_state[:2] = self.start_pos

        direction = self.centerline[1, :2] - self.centerline[0, :2]
        self.robot_state[2] = torch.atan2(direction[1], direction[0])\

    # for dynamic environment
    def step(self, dt: float):
        pass



    # obstacle generation
    def _generate_obstacles(self):
        obs_cfg = self.cfg.obstacle

        x_half = obs_cfg.map_size[0] / 2
        y_half = obs_cfg.map_size[1] / 2

        generate_random_obstacles(
            obstacle_map=self.map,
            random_x_range=(-x_half, x_half),
            random_y_range=(-y_half, y_half),
            num_circle_obs=obs_cfg.num_circle_obs,
            radius_range=obs_cfg.circle_radius_range,
            num_rectangle_obs=obs_cfg.num_rectangle_obs,
            width_range=obs_cfg.rectangle_size_range,
            height_range=obs_cfg.rectangle_size_range,
            speed_range=obs_cfg.speed_range,
            max_iteration=200,
            seed=obs_cfg.seed,
        )

        self.map.convert_to_torch()


    # sampling on lane (IMPORTANT for MPPI / exploration)
    def sample_point_on_lane(self, lateral_range=2.0):
        idx = np.random.randint(0, len(self.centerline))

        center = self.centerline[idx, :2].cpu().numpy()

        if idx < len(self.centerline) - 1:
            next_p = self.centerline[idx + 1, :2].cpu().numpy()
        else:
            next_p = self.centerline[idx - 1, :2].cpu().numpy()

        direction = next_p - center
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        normal = np.array([-direction[1], direction[0]])
        offset = np.random.uniform(-lateral_range, lateral_range)

        return center + offset * normal