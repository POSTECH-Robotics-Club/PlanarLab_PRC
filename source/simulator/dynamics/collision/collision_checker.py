"""
Collision + range computation module (GPU-ready)
Isaac Lab style separation:
- NO cost
- ONLY geometry / occupancy query
"""

from PIL.TiffImagePlugin import X_RESOLUTION
import torch
import torch.nn.functional as F


class CollisionChecker:
    def __init__(
        self,
        obstacle_map,
        robot_radius: float = 1.0,
        detect_range: float = 4.0,
    ):
        self.map = obstacle_map
        print("robot_radius : ", robot_radius)

        self.robot_radius = robot_radius
        self.detect_range = detect_range

        self.device = obstacle_map._device
        self.dtype = obstacle_map._dtype

        self._map_torch = obstacle_map.convert_to_torch()
        if self._map_torch is None:
            raise ValueError("convert_to_torch() returned None")

        self._inflated_map = None

    # ------------------------------------------------------------
    # build inflated map (robot radius)
    # ------------------------------------------------------------
    def _build_inflated_map(self):
        radius_occ = int(round(self.robot_radius / self.map._cell_size))
        kernel_size = 2 * radius_occ + 1

        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size),
            device=self.device,
            dtype=torch.float32,
        )

        map_tensor = self._map_torch.unsqueeze(0).unsqueeze(0).float()

        inflated = F.conv2d(map_tensor, kernel, padding=radius_occ)

        self._inflated_map = (inflated > 0).squeeze(0).squeeze(0)


    # core: collision + range check
    def compute_collision(self, x: torch.Tensor, initial_state=None):
        """
        Args:
            x:
                (B, 2) or (B, H, 2)
            initial_state:
                (6,) or (B, 6)

        Returns:
            collisions: (B,) or (B, H)
            range_mask: (B,) or (B, H)
        """

        if self._inflated_map is None:
            self._build_inflated_map()

        assert self._inflated_map is not None, "_inflated_map is still None!"

        if x.device != self.device:
            x = x.to(self.device)

    
        # reshape safety
        squeeze_back = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_back = True

        
        # world -> grid
        x_occ = (x / self.map._cell_size) + self.map._torch_cell_map_origin
        x_occ = torch.round(x_occ).long()


        # initial state (range reference)
        if initial_state is None:
            initial_state = x[0, 0]

        if initial_state.dim() == 1:
            init_occ = (
                initial_state[:2] / self.map._cell_size
            ) + self.map._torch_cell_map_origin
        else:
            init_occ = (
                initial_state[..., :2] / self.map._cell_size
            ) + self.map._torch_cell_map_origin


        # out of bounds
        out_of_bound = (
            (x_occ[..., 0] < 0)
            | (x_occ[..., 0] >= self._inflated_map.shape[0])
            | (x_occ[..., 1] < 0)
            | (x_occ[..., 1] >= self._inflated_map.shape[1])
        )

        # clamp indices
        x_occ[..., 0] = torch.clamp(x_occ[..., 0], 0, self._inflated_map.shape[0] - 1)
        x_occ[..., 1] = torch.clamp(x_occ[..., 1], 0, self._inflated_map.shape[1] - 1)


        # collision lookup
        collisions = self._inflated_map[x_occ[..., 0], x_occ[..., 1]].clone()
        collisions = collisions.float()

        collisions[out_of_bound] = 1.0


        # range check (sensor-like constraint)
        range_occ = torch.sqrt(
            (x_occ[..., 0] - init_occ[..., 0]) ** 2
            + (x_occ[..., 1] - init_occ[..., 1]) ** 2
        )

        detect_range_occ = torch.tensor(
            self.detect_range / self.map._cell_size,
            device=self.device,
            dtype=torch.float32,
        )

        out_of_range = range_occ >= detect_range_occ

        # range mask (for visualization / gating only)
        range_mask = collisions.clone()
        range_mask[out_of_range] = 0.0


        # restore shape
        if squeeze_back:
            collisions = collisions.squeeze(1)
            range_mask = range_mask.squeeze(1)


        return collisions, range_mask

    


    # convenience APIs (MPPI / TDMPC friendly)
    def check_point(self, state: torch.Tensor):
        """
        state: (B, 6)
        """
        pos = state[..., :2]   # (x, y) * (batch size)
        return self.compute_collision(pos)[0]

    def check_trajectory(self, states: torch.Tensor):
        """
        states: (B, H, 6)
        """
        pos = states[..., :2]
        return self.compute_collision(pos)[0]

    def check_any(self, states: torch.Tensor):
        """
        returns (B,)
        """
        traj = self.check_trajectory(states)
        return traj.any(dim=-1)