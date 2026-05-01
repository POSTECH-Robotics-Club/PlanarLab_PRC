"""
Random obstacle generator
"""

from ipywidgets.widgets.widget_datetime import validate
import numpy as np


def generate_random_obstacles(
    obstacle_map,
    random_x_range,
    random_y_range,
    num_circle_obs,
    radius_range,
    num_rectangle_obs,
    width_range,
    height_range,
    speed_range,
    max_iteration,
    seed,
):

    rng = np.random.default_rng(seed)

    # clamp to map
    random_x_range = list(random_x_range)
    random_y_range = list(random_y_range)

    random_x_range[0] = max(random_x_range[0], obstacle_map.x_lim[0])
    random_x_range[1] = min(random_x_range[1], obstacle_map.x_lim[1])
    random_y_range[0] = max(random_y_range[0], obstacle_map.y_lim[0])
    random_y_range[1] = min(random_y_range[1], obstacle_map.y_lim[1])

    # circles
    for _ in range(num_circle_obs):
        for _ in range(max_iteration):
            c = np.array([
                rng.uniform(*random_x_range),
                rng.uniform(*random_y_range),
            ])
            r = rng.uniform(*radius_range)
            theta = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(*speed_range)

            v = np.array([
                speed * np.cos(theta),
                speed * np.sin(theta),
            ])

            overlap = False
            for o in obstacle_map.circle_obs_list:
                if np.linalg.norm(o.center - c) <= o.radius + r:
                    overlap = True

            if not overlap:
                obstacle_map.add_circle_obstacle(c, r, v)
                break

    # rectangles
    for _ in range(num_rectangle_obs):
        for _ in range(max_iteration):
            c = np.array([
                rng.uniform(*random_x_range),
                rng.uniform(*random_y_range),
            ])
            w = rng.uniform(*width_range)
            h = rng.uniform(*height_range)
            theta = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(*speed_range)

            v = np.array([
                speed * np.cos(theta),
                speed * np.sin(theta),
            ])

            overlap = False
            for o in obstacle_map.circle_obs_list:
                if np.linalg.norm(o.center - c) <= o.radius + max(w, h) / 2:
                    overlap = True

            if not overlap:
                obstacle_map.add_rectangle_obstacle(c, w, h, v)
                break