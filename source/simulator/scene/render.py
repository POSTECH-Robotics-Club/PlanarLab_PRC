# source/scene/render.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class NavigationRenderer:
    def __init__(self, env):
        """
        Stateful matplotlib renderer.

        Args:
            env: NavigationEnv
        """
        self.env = env

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()

        self._ax.set_xlim(self.env.map.x_lim)
        self._ax.set_ylim(self.env.map.y_lim)
        self._ax.set_aspect("equal")

    def render(
        self,
        state,
        action=None,
        predicted_trajectory=None,
        collisions=None,
        top_samples=None,
    ):
        """
        Render current scene.

        Returns:
            RGB image (H, W, 3)
        """

        ax = self._ax
        ax.cla()

        ax.set_xlim(self.env.map.x_lim)
        ax.set_ylim(self.env.map.y_lim)
        ax.set_aspect("equal")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # -------------------------
        # Obstacle map
        # -------------------------
        self.env.map.render(ax, zorder=10)

        # -------------------------
        # Start & Goal
        # -------------------------
        ax.scatter(*self.env.start.cpu().numpy(), color="red", zorder=10)
        ax.scatter(*self.env.goal.cpu().numpy(), color="orange", zorder=10)

        # -------------------------
        # Robot
        # -------------------------
        x, y = state[0].item(), state[1].item()
        theta = state[2].item()
        v = state[3].item()

        ax.scatter(x, y, color="green", zorder=100)

        ax.quiver(
            x, y,
            v * np.cos(theta),
            v * np.sin(theta),
            color="green",
            zorder=100,
        )

        # Steering direction
        if action is not None:
            steer = action[1].item()

            ax.quiver(
                x, y,
                self.env.L.cpu().item() * np.cos(theta + steer),
                self.env.L.cpu().item() * np.sin(theta + steer),
                color="blue",
                zorder=100,
            )

        # -------------------------
        # Detection range
        # -------------------------
        circle = patches.Circle(
            (x, y),
            radius=self.env.map._detect_range,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            linewidth=0.5,
            zorder=80,
        )
        ax.add_patch(circle)

        # -------------------------
        # Collision visualization
        # -------------------------
        if collisions is not None:
            if collisions[0, 0].item() > 0:
                circle = patches.Circle(
                    (x, y),
                    radius=self.env.map._detect_range / 2,
                    color="red",
                    zorder=150,
                )
                ax.add_patch(circle)

        # -------------------------
        # Top samples
        # -------------------------
        if top_samples is not None:
            samples, weights = top_samples
            samples = samples.cpu().numpy()
            weights = weights.cpu().numpy()

            weights = 0.7 * weights / np.max(weights)
            weights = np.clip(weights, 0.1, 0.7)

            for i in range(samples.shape[0]):
                ax.plot(
                    samples[i, :, 0],
                    samples[i, :, 1],
                    color="lightblue",
                    alpha=weights[i],
                    zorder=1,
                )

        # -------------------------
        # Predicted trajectory
        # -------------------------
        if predicted_trajectory is not None:
            traj = predicted_trajectory[0].cpu().numpy()

            colors = np.array(["darkblue"] * traj.shape[0])

            if collisions is not None:
                col = collisions.cpu().numpy()
                col = np.any(col, axis=0)
                colors[col] = "red"

            ax.scatter(traj[:, 0], traj[:, 1], c=colors, s=3, zorder=2)

        # -------------------------
        # Title
        # -------------------------
        if action is not None:
            ax.set_title(
                f"v: {v:.2f}, accel: {action[0].item():.2f}, steer: {action[1].item():.2f}"
            )

        # -------------------------
        # Convert to RGB image
        # -------------------------
        canvas = FigureCanvas(self._fig)
        canvas.draw()

        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]

        return image