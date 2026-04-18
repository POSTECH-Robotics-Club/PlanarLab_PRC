# source/envs/base_env.py

from typing import Tuple, Any, Dict


class BaseEnv:
    """
    Base interface for all environment wrappers.

    This class defines the standard API that all environments must follow.
    It does NOT implement any environment-specific logic.
    """

    def __init__(self, cfg):
        """
        Initialize the environment with configuration.

        Args:
            cfg: Configuration object containing environment parameters.
        """
        self.cfg = cfg

    def reset(self) -> Any:
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial observation after reset.
        """
        raise NotImplementedError

    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Perform one step in the environment.

        Args:
            action: Action to apply.

        Returns:
            observation: Next observation.
            reward: Scalar reward.
            done: Whether the episode is finished.
            info: Additional debug information.
        """
        raise NotImplementedError

    def render(self):
        """
        Render the environment (optional).
        """
        pass

    def close(self):
        """
        Clean up resources (optional).
        """
        pass