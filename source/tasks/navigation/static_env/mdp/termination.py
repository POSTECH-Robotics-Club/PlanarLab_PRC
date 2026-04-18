"""
Kohei Honda, 2023.
Refactored for MPPI-style modular termination (IsaacLab-like)
"""

from __future__ import annotations
import torch


# =========================
# Goal Reached
# =========================

def goal_reached(
    state: torch.Tensor,
    goal: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Check if goal is reached.

    Args:
        state: (batch, state_dim)
        goal: (2,)
        threshold: distance threshold

    Returns:
        done: (batch,) bool tensor
    """
    dist = torch.norm(state[:, :2] - goal, dim=1)
    return dist < threshold


# =========================
# Collision Termination
# =========================

def collision_termination(
    state: torch.Tensor,
    obstacle_map,
) -> torch.Tensor:
    """
    Check collision with obstacles.

    Args:
        state: (batch, traj, state_dim) OR (batch, state_dim)

    Returns:
        done: (batch,) bool tensor
    """
    if state.dim() == 2:
        pos = state[:, :2].unsqueeze(1)  # (batch, 1, 2)
    else:
        pos = state[:, :, :2]  # (batch, traj, 2)

    is_collision, _ = obstacle_map.compute_cost(pos)
    is_collision = is_collision.squeeze(1)

    return is_collision > 0


# =========================
# Out of Bounds
# =========================

def out_of_bounds(
    state: torch.Tensor,
    world_min: torch.Tensor,
    world_max: torch.Tensor,
) -> torch.Tensor:
    """
    Check if robot leaves map.

    Args:
        state: (batch, state_dim)

    Returns:
        done: (batch,)
    """
    pos = state[:, :2]

    lower = pos < world_min
    upper = pos > world_max

    out = torch.any(lower | upper, dim=1)
    return out


# =========================
# Timeout
# =========================

def time_out(
    step_count: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    """
    Check episode timeout.

    Args:
        step_count: (batch,)
        max_steps: int

    Returns:
        done: (batch,)
    """
    return step_count >= max_steps


# =========================
# Combined Termination
# =========================

def compute_termination(
    state: torch.Tensor,
    step_count: torch.Tensor,
    goal: torch.Tensor,
    obstacle_map,
    world_min: torch.Tensor,
    world_max: torch.Tensor,
    max_steps: int,
    goal_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Combine all termination conditions.

    Returns:
        done: (batch,)
    """

    done_goal = goal_reached(state, goal, goal_threshold)
    done_collision = collision_termination(state, obstacle_map)
    done_oob = out_of_bounds(state, world_min, world_max)
    done_timeout = time_out(step_count, max_steps)

    done = done_goal | done_collision | done_oob | done_timeout
    return done