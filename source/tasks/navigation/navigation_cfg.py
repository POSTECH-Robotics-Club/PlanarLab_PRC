"Example code"

from dataclasses import dataclass, field
from typing import Optional

# task configs
from source.tasks.navigation.navigation_cfg import NavigationCfg

# controller configs (optional) -> we must devolpe this class! (Not coded yet)
from source.config.mppi_cfg import MPPIConfig


# -------------------------------------------------
# Full Config (Top-level)
# -------------------------------------------------
@dataclass
class MPPINavigationCfg:
    """
    Top-level configuration for entire system.

    This acts as:
    - experiment config
    - entry-point config
    """

    # -------------------------
    # Task
    # -------------------------
    task: NavigationCfg = field(default_factory=NavigationCfg)

    # -------------------------
    # Controller (Optional)
    # -------------------------
    mppi: Optional[MPPIConfig] = None

    # -------------------------
    # Global settings (optional)
    # -------------------------
    seed: int = 42
    device: str = "cuda"


@dataclass
class TDMPCNavigationCfg:
    """
    Top-level configuration for entire system.

    This acts as:
    - experiment config
    - entry-point config
    """

    # -------------------------
    # Task
    # -------------------------
    task: NavigationCfg = field(default_factory=NavigationCfg)

    # -------------------------
    # Controller (Optional)
    # -------------------------
    tdmpc: Optional[tdmpcConfig] = None

    # -------------------------
    # Global settings (optional)
    # -------------------------
    seed: int = 42
    device: str = "cuda"