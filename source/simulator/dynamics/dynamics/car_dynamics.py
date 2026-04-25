import torch
import numpy as np


class CarDynamics:
    """
    Approximated bicycle-style vehicle dynamics.

    State:  [x, y, theta, u, v, w]
    Action: [accel, steer]
    """

    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        self.m = torch.tensor(cfg.m, device=self.device, dtype=self.dtype)
        self.lf = torch.tensor(cfg.lf, device=self.device, dtype=self.dtype)
        self.lr = torch.tensor(cfg.lr, device=self.device, dtype=self.dtype)
        self.Iz = torch.tensor(cfg.Iz, device=self.device, dtype=self.dtype)
        self.kf = torch.tensor(cfg.kf, device=self.device, dtype=self.dtype)
        self.kr = torch.tensor(cfg.kr, device=self.device, dtype=self.dtype)

        self.u_min = torch.tensor(cfg.u_min, device=self.device, dtype=self.dtype)
        self.u_max = torch.tensor(cfg.u_max, device=self.device, dtype=self.dtype)
        self.v_max = torch.tensor(cfg.v_max, device=self.device, dtype=self.dtype)
        self.w_max = torch.tensor(cfg.w_max, device=self.device, dtype=self.dtype)

        self.x_lim = torch.tensor(cfg.x_lim, device=self.device, dtype=self.dtype)
        self.y_lim = torch.tensor(cfg.y_lim, device=self.device, dtype=self.dtype)


        self.process_noise_std = cfg.process_noise_std

    # core: nonlinear dynamics (USED BY MPPI / TD-MPC)
    def step(self, state: torch.Tensor, action: torch.Tensor, dt: float):
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 6) [x, y, theta, u, v, w]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [accel, steer]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 6) [x, y, theta, u, v, w]
        """

        x, y, theta, u, v, w = state.unbind(-1)

        accel = torch.clamp(action[..., 0], self.u_min[0], self.u_max[0])
        steer = torch.clamp(action[..., 1], self.u_min[1], self.u_max[1])

        theta = self._wrap(theta)

        # kinematics
        new_x = x + dt * (u * torch.cos(theta) - v * torch.sin(theta))
        new_y = y + dt * (v * torch.cos(theta) + u * torch.sin(theta))
        new_theta = self._wrap(theta + dt * w)

        # boundary clamp
        new_x = torch.clamp(new_x, self.x_lim[0], self.x_lim[1])
        new_y = torch.clamp(new_y, self.y_lim[0], self.y_lim[1])


        # longitudinal
        new_u = u + dt * accel

        # lateral velocity (bicycle approx)
        denom_v = (self.m * u - dt * (self.kf + self.kr))
        new_v = (
            self.m * u * v
            + dt * (self.lf * self.kf - self.lr * self.kr) * w
            - dt * self.kf * steer * u
            - dt * self.m * u * u * w
        ) / denom_v

        # yaw rate
        denom_w = (self.Iz * u - dt * (self.lf**2 * self.kf + self.lr**2 * self.kr))
        new_w = (
            self.Iz * u * w
            + dt * (self.lf * self.kf - self.lr * self.kr) * v
            - dt * self.lf * self.kf * steer * u
        ) / denom_w

        # clamp stability
        new_u = torch.clamp(new_u, -self.v_max, self.v_max)
        # new_v = torch.clamp(new_v, -self.v_max, self.v_max)
        new_w = torch.clamp(new_w, -self.w_max, self.w_max)

        next_state = torch.stack(
            [new_x, new_y, new_theta, new_u, new_v, new_w], dim=-1
        )

        # optional noise (realism)
        if self.process_noise_std > 0:
            next_state = next_state + torch.randn_like(next_state) * self.process_noise_std

        return next_state


    # optional utility
    def _wrap(self, x):
        return (x + torch.pi) % (2 * torch.pi) - torch.pi