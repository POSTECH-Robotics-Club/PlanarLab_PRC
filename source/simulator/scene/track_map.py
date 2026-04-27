"""
Lane map (grid + geometry + visualization)
Michikuni Eguchi, 2024 (refactored)
"""

from __future__ import annotations
from source.simulator.scene.obstacle_map import ObstacleMap

from typing import Tuple
import numpy as np
import torch
from math import ceil
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt



class TrackMap(ObstacleMap):
    """
    Track map represented by occupancy grid.

    0 = drivable area
    1 = non-drivable area
    """

    def __init__(
        self,
        lane: np.ndarray,
        right_lane: np.ndarray,
        left_lane: np.ndarray,
        lane_width: float,
        map_size: Tuple[int, int] = (20, 20),
        cell_size: float = 0.01,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        assert lane_width > 0
        assert len(lane.shape) == 2 and lane.shape[1] == 3

        self._device = torch.device("cuda") if torch.cuda.is_available() and device == torch.device("cuda") else torch.device("cpu")
        self._dtype = dtype

        self._cell_size = cell_size
        self._lane_width = lane_width
        self.right_lane = right_lane
        self.left_lane = left_lane

        # grid init
        cell_map_dim = [
            ceil(map_size[0] / cell_size),
            ceil(map_size[1] / cell_size),
        ]

        self._map = np.ones(cell_map_dim)

        self._cell_map_origin = np.array(
            [cell_map_dim[0] // 2, cell_map_dim[1] // 2]
        ).astype(int)

        self._torch_cell_map_origin = torch.from_numpy(
            self._cell_map_origin
        ).to(self._device, self._dtype)

        self.x_lim = [-map_size[0] / 2, map_size[0] / 2]
        self.y_lim = [-map_size[1] / 2, map_size[1] / 2]

        self._populate_lane(lane)


    # core generation
    def _populate_lane(self, lane: np.ndarray):
        """
        Mark lane centerline and expand via distance transform
        """

        # 1. draw centerline
        for p in lane:
            x, y = p[0], p[1]

            cx = int(round(x / self._cell_size)) + self._cell_map_origin[0]
            cy = int(round(y / self._cell_size)) + self._cell_map_origin[1]

            if 0 <= cx < self._map.shape[0] and 0 <= cy < self._map.shape[1]:
                self._map[cx, cy] = 0

        # 2. expand drivable region
        dist = distance_transform_edt(self._map)
        max_dist = (self._lane_width / 2) / self._cell_size
        self._map = np.where(dist <= max_dist, 0, 1)

        self._map_torch = torch.tensor(self._map, device=self._device, dtype=self._dtype)




    # visualization
    def render_occupancy(self, ax, cmap="binary"):
        extent = [self.x_lim[0], self.x_lim[1], self.y_lim[0], self.y_lim[1]]
        ax.imshow(self._map.T, cmap=cmap, origin='lower', extent=extent)


    def render(self, ax, zorder=0):
        """
        Visualize the track map with obstacles and lanes.
        """
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect("equal")

        # Render lane obstacles as circle and rectangle
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

        ax.plot(
            self.right_lane[:, 0], self.right_lane[:, 1], 
            color="black", linestyle="--", zorder=5,
        )
        ax.plot(
            self.left_lane[:, 0], self.left_lane[:, 1], 
            color="black", linestyle="--", zorder=5,
        )