"""
Obstacle map (grid + geometry + visualization)
Kohei Honda, 2023 (refactored)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from math import ceil
from matplotlib import pyplot as plt
import os


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float
    velocity: np.ndarray  # (vx, vy)


@dataclass
class RectangleObstacle:
    center: np.ndarray
    width: float
    height: float
    velocity: np.ndarray  # (vx, vy)


class ObstacleMap:
    def __init__(
        self,
        map_size: Tuple[int, int] = (20, 20),
        cell_size: float = 0.001,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ):
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._dtype = dtype

        assert len(map_size) == 2
        assert map_size[0] % 2 == 0
        assert map_size[1] % 2 == 0
        assert cell_size > 0

        self._cell_size = cell_size

        cell_map_dim = [
            ceil(map_size[0] / cell_size),
            ceil(map_size[1] / cell_size),
        ]

        self._map = np.zeros(cell_map_dim)

        self._cell_map_origin = np.array(
            [cell_map_dim[0] / 2, cell_map_dim[1] / 2]
        ).astype(int)

        self._torch_cell_map_origin = torch.from_numpy(
            self._cell_map_origin
        ).to(self._device, self._dtype)

        x_range = self._cell_size * self._map.shape[0]
        y_range = self._cell_size * self._map.shape[1]

        self.x_lim = [-x_range / 2, x_range / 2]
        self.y_lim = [-y_range / 2, y_range / 2]


        self.circle_obs_list: List[CircleObstacle] = []
        self.rectangle_obs_list: List[RectangleObstacle] = []

        self._torch_map = torch.from_numpy(self._map).to(self._device, self._dtype)


    # obstacle insertion
    def add_circle_obstacle(self, center: np.ndarray, radius: float, velocity: np.ndarray,) -> None:
        assert len(center) == 2
        assert radius > 0

        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.round(center_occ).astype(int)
        radius_occ = ceil(radius / self._cell_size)

        for i in range(-radius_occ, radius_occ + 1):
            for j in range(-radius_occ, radius_occ + 1):
                if i**2 + j**2 <= radius_occ**2:
                    x = np.clip(center_occ[0] + i, 0, self._map.shape[0] - 1)
                    y = np.clip(center_occ[1] + j, 0, self._map.shape[1] - 1)
                    self._map[x, y] = 1

        self.circle_obs_list.append(CircleObstacle(center, radius, velocity))

    def add_rectangle_obstacle(
        self, center: np.ndarray, width: float, height: float, velocity: np.ndarray
    ) -> None:
        assert len(center) == 2
        assert width > 0
        assert height > 0

        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.ceil(center_occ).astype(int)

        w = ceil(width / self._cell_size)
        h = ceil(height / self._cell_size)

        x_init = np.clip(center_occ[0] - ceil(w / 2), 0, self._map.shape[0] - 1)
        x_end = np.clip(center_occ[0] + ceil(w / 2), 0, self._map.shape[0] - 1)
        y_init = np.clip(center_occ[1] - ceil(h / 2), 0, self._map.shape[1] - 1)
        y_end = np.clip(center_occ[1] + ceil(h / 2), 0, self._map.shape[1] - 1)

        self._map[x_init:x_end, y_init:y_end] = 1

        self.rectangle_obs_list.append(RectangleObstacle(center, width, height, velocity))


    # torch / visualization
    def convert_to_torch(self):
        self._torch_map = torch.from_numpy(self._map).to(self._device, self._dtype)
        return self._torch_map

    def render_occupancy(self, ax, cmap="binary"):
        ax.imshow(self._map, cmap=cmap)

    def render(self, ax, zorder=0):
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect("equal")

        for c in self.circle_obs_list:
            ax.add_patch(plt.Circle(c.center, c.radius, color="gray", zorder=zorder))

        for r in self.rectangle_obs_list:
            ax.add_patch(
                plt.Rectangle(
                    r.center - np.array([r.width / 2, r.height / 2]),
                    r.width,
                    r.height,
                    color="gray",
                    zorder=zorder,
                )
            )

    def rebuild_map(self):
        # 1. clear occupancy grid
        self._map.fill(0)

        # 2. circle obstacles
        for obs in self.circle_obs_list:
            self._rasterize_circle_fast(obs.center, obs.radius)

        # 3. rectangle obstacles
        for obs in self.rectangle_obs_list:
            self._rasterize_rectangle_fast(obs.center, obs.width, obs.height)

    def _rasterize_circle_fast(self, center, radius):
        cx = center[0] / self._cell_size + self._cell_map_origin[0]
        cy = center[1] / self._cell_size + self._cell_map_origin[1]

        cx = int(cx + 0.5)
        cy = int(cy + 0.5)

        r = int(radius / self._cell_size)

        x_min = max(cx - r, 0)
        x_max = min(cx + r + 1, self._map.shape[0])
        y_min = max(cy - r, 0)
        y_max = min(cy + r + 1, self._map.shape[1])

        rr = r * r

        for i in range(x_min, x_max):
            dx = i - cx
            dx2 = dx * dx

            for j in range(y_min, y_max):
                dy = j - cy
                if dx2 + dy * dy <= rr:
                    self._map[i, j] = 1

    def _rasterize_rectangle_fast(self, center, width, height):
        cx = center[0] / self._cell_size + self._cell_map_origin[0]
        cy = center[1] / self._cell_size + self._cell_map_origin[1]

        cx = int(cx + 0.5)
        cy = int(cy + 0.5)

        w = int(width / self._cell_size)
        h = int(height / self._cell_size)

        x_min = max(cx - w // 2, 0)
        x_max = min(cx + w // 2 + 1, self._map.shape[0])

        y_min = max(cy - h // 2, 0)
        y_max = min(cy + h // 2 + 1, self._map.shape[1])

        self._map[x_min:x_max, y_min:y_max] = 1



    # for dynamic obstacle environment
    def step(self, dt: float):
        for obs in self.circle_obs_list:
            # Move the obstacle
            obs.center += obs.velocity * dt
            
            # Clamp the obstacle position to ensure it doesn't go out of bounds
            obs.center[0] = np.clip(obs.center[0], self.x_lim[0], self.x_lim[1])
            obs.center[1] = np.clip(obs.center[1], self.y_lim[0], self.y_lim[1])

            # Add reflection effect when the obstacle hits the boundary
            if obs.center[0] == self.x_lim[0] or obs.center[0] == self.x_lim[1]:
                obs.velocity[0] = -obs.velocity[0]  # Reflect the velocity along X-axis

            if obs.center[1] == self.y_lim[0] or obs.center[1] == self.y_lim[1]:
                obs.velocity[1] = -obs.velocity[1]  # Reflect the velocity along Y-axis

        for obs in self.rectangle_obs_list:
            # Move the obstacle
            obs.center += obs.velocity * dt
            
            # Clamp the obstacle position to ensure it doesn't go out of bounds
            obs.center[0] = np.clip(obs.center[0], self.x_lim[0], self.x_lim[1])
            obs.center[1] = np.clip(obs.center[1], self.y_lim[0], self.y_lim[1])

            # Add reflection effect when the obstacle hits the boundary
            if obs.center[0] == self.x_lim[0] or obs.center[0] == self.x_lim[1]:
                obs.velocity[0] = -obs.velocity[0]  # Reflect the velocity along X-axis

            if obs.center[1] == self.y_lim[0] or obs.center[1] == self.y_lim[1]:
                obs.velocity[1] = -obs.velocity[1]  # Reflect the velocity along Y-axis

        self.rebuild_map()


        # Update the map after moving the obstacles
        self._torch_map = torch.from_numpy(self._map).to(self._device, self._dtype)

    def clear_obstacles(self):
        self._map[:] = 0
        self.circle_obs_list = []
        self.rectangle_obs_list = []

