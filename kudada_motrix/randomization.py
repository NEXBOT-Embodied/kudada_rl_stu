"""Explicit per-episode randomization for the flat training task."""

import math

from motrix_env_core.config import configclass


@configclass
class DomainRandCfg:
    randomize_payload_mass: bool = False
    payload_mass_range: tuple[float, float] = (-.5, 1.)
    randomize_com_displacement: bool = False
    com_displacement_range: tuple[float, float] = (-.01, .01)
    randomize_link_mass: bool = False
    link_mass_range: tuple[float, float] = (.8, 1.2)
    randomize_inertia: bool = False
    inertia_range: tuple[float, float] = (.8, 1.2)
    randomize_friction: bool = False
    friction_range: tuple[float, float] = (.5, 1.2)
    randomize_restitution: bool = False  # Unsupported by MotrixSim 0.10.1.
    randomize_motor_strength: bool = False
    motor_strength_range: tuple[float, float] = (.9, 1.1)
    randomize_kp: bool = False
    kp_range: tuple[float, float] = (.8, 1.2)
    randomize_kd: bool = False
    kd_range: tuple[float, float] = (.8, 1.2)
    randomize_initial_joint_pos: bool = False
    initial_joint_pos_range: tuple[float, float] = (.8, 1.2)
    randomize_init_joint_offset: bool = False
    init_joint_offset_range: tuple[float, float] = (-.1, .1)
    randomize_motor_zero_offset: bool = False
    motor_zero_offset_range: tuple[float, float] = (-.035, .035)
    randomize_initial_velocity: bool = False
    initial_velocity_range: tuple[float, float] = (-.5, .5)
    delay: bool = False
    delay_substeps_range: tuple[int, int] = (0, 19)  # inclusive, sampled each action
    disturbance: bool = False
    disturbance_range: tuple[float, float] = (-5., 5.)
    disturbance_interval_steps: int = 8
    push_robots: bool = False
    push_interval_seconds: float = 30.
    max_push_vel_xy: float = .5

    @property
    def physical_enabled(self):
        return any((self.randomize_payload_mass, self.randomize_com_displacement,
                    self.randomize_link_mass, self.randomize_inertia, self.randomize_friction))

    def validate(self):
        if self.randomize_restitution:
            raise ValueError("MotrixSim 0.10.1 has no restitution override; this switch cannot be enabled")
        for name, bounds in vars(self).items():
            if name.endswith("_range"):
                if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                    raise ValueError(f"{name} must contain finite, ordered bounds")
        for name in ("link_mass_range", "inertia_range", "motor_strength_range", "kp_range", "kd_range",
                     "initial_joint_pos_range"):
            if getattr(self, name)[0] <= 0:
                raise ValueError(f"{name} must be positive")
        if self.friction_range[0] < 0:
            raise ValueError("friction_range must be nonnegative")
        lo, hi = self.delay_substeps_range
        if not isinstance(lo, int) or not isinstance(hi, int) or not 0 <= lo <= hi < 20:
            raise ValueError("delay_substeps_range must contain integers in [0, 19]")
        if not isinstance(self.disturbance_interval_steps, int) or self.disturbance_interval_steps < 1:
            raise ValueError("disturbance_interval_steps must be a positive integer")
        if not math.isfinite(self.push_interval_seconds) or self.push_interval_seconds < .02:
            raise ValueError("push_interval_seconds must be at least .02")
        if not math.isfinite(self.max_push_vel_xy) or self.max_push_vel_xy < 0:
            raise ValueError("max_push_vel_xy must be finite and nonnegative")


def training_randomization():
    return DomainRandCfg(
        randomize_payload_mass=True, randomize_com_displacement=True,
        randomize_link_mass=True, randomize_inertia=True, randomize_friction=True,
        randomize_motor_strength=True, randomize_kp=True, randomize_kd=True,
        randomize_initial_joint_pos=True, randomize_init_joint_offset=True,
        randomize_motor_zero_offset=True, randomize_initial_velocity=True,
        delay=True, disturbance=True, push_robots=True,
    )
