"""Frozen kudada_real legacy values; verified against stage 0 fixtures."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets/kudada/kudada_lingzu.xml"
JOINT_NAMES = tuple(
    f"{side}_leg_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch")
)
FEET = ("left_leg_ankle_pitch_link", "right_leg_ankle_pitch_link")
CONTACT_LINKS = ("base_link", *FEET, "left_leg_knee_pitch_link", "right_leg_knee_pitch_link")
DEFAULT = np.array([0, .05, -.2, .6, -.4, 0, -.05, -.2, .6, -.4], dtype=np.float32)
KP = np.array([15, 18, 20, 20, 12] * 2, dtype=np.float32)
KD = np.array([.5, .5, .7, .7, .35] * 2, dtype=np.float32)
COMMAND_SCALE = np.array([2, 2, .25], dtype=np.float32)
FRAME_SIZE = 44
HISTORY_FRAMES = 21
CRITIC_SIZE = 243
ACTION_SCALE = .25
CLIP = 100.
