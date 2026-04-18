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


@dataclass
class RectangleObstacle:
    center: np.ndarray
    width: float
    height: float


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

    # -------------------------
    # obstacle insertion
    # -------------------------
    def add_circle_obstacle(self, center: np.ndarray, radius: float) -> None:
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

        self.circle_obs_list.append(CircleObstacle(center, radius))

    def add_rectangle_obstacle(
        self, center: np.ndarray, width: float, height: float
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

        self.rectangle_obs_list.append(RectangleObstacle(center, width, height))

    # -------------------------
    # torch / visualization
    # -------------------------
    def convert_to_torch(self):
        return torch.from_numpy(self._map).to(self._device, self._dtype)

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

    def clear_obstacles(self):
        self._map[:] = 0
        self.circle_obs_list = []
        self.rectangle_obs_list = []

