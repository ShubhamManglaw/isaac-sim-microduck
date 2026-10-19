from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

ASSET_DIR = Path(__file__).resolve().parent / "assets"
MICRODUCK_USD_PATH = str(ASSET_DIR / "microduck.usd")


@configclass
class MicroduckArticulationCfg(ArticulationCfg):
    """Configuration for the Microduck biped robot articulation in Isaac Lab."""

    # -------------------------------------------------------------
    # BLOCK 1: Spawner (Where is the USD file on disk?)
    # -------------------------------------------------------------
    spawn  : sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=MICRODUCK_USD_PATH,
        activate_contact_sensors=True,
    )

    # -------------------------------------------------------------
    # BLOCK 2: Initial State (How does it spawn into the world?)
    # -------------------------------------------------------------
    # (To be implemented)
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # Left Leg Chain
            ".*left_hip_yaw.*": 0.0,
            ".*left_hip_roll.*": -0.0873,
            ".*left_hip_pitch.*": -0.4579,
            ".*left_knee.*": -0.0049,
            ".*left_ankle.*": 0.4530,
            # Head / Neck Chain
            ".*neck_pitch.*": 0.3491,
            ".*head_pitch.*": 0.3491,
            ".*head_yaw.*": 0.0,
            ".*head_roll.*": 0.0,
            # Right Leg Chain
            ".*right_hip_yaw.*": 0.0,
            ".*right_hip_roll.*": 0.0873,
            ".*right_hip_pitch.*": 0.4579,
            ".*right_knee.*": 0.0049,
            ".*right_ankle.*": -0.4530,
        },
        joint_vel={".*": 0.0},
    )

    # -------------------------------------------------------------
    # BLOCK 3: Actuator Dynamics (How do the motors behave?)
    # -------------------------------------------------------------
    # (To be implemented)
    actuators : dict[str,IdealPDActuatorCfg]={
        "legs_and_head":IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=8.5,
            damping=0.45,
            armature=0.0018,
            effort_limit=0.96,
            velocity_limit=10.0,
        )
    }
    # -------------------------------------------------------------
    # BLOCK 4: Safety / Soft Joint Limits
    # -------------------------------------------------------------
    soft_joint_pos_limit_factor: float = 0.90
