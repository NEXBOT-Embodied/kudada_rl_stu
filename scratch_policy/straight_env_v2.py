"""Straight-walking PPO rewards with cycle yaw and a fixed nominal cohort.

The actor and controller observation contract remain unchanged. Circular yaw,
the rolling window and cohort membership are training-only state; no original
policy or policy demonstrations are used.
"""

from __future__ import annotations

import math

import numpy as np

from motrix_env_core.config import configclass
from scratch_policy.env import ScratchCfg, ScratchEnv

METHOD = "PPO_straight_v2"


class CircularYawWindow:
    """Causal, per-episode mean yaw rate across a fixed-duration window.

Each step unwraps the shortest circular displacement. The rate uses actual
elapsed samples, and ``full`` distinguishes complete windows from startup.
Resetting some rows does not disturb the other rows or the shared ring cursor.
"""

    def __init__(self, num_envs: int, ctrl_dt: float, window_seconds: float = .5):
        ratio = window_seconds / ctrl_dt
        if num_envs < 1 or not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio)):
            raise ValueError("Yaw window must contain a positive integer number of control steps")
        self.ctrl_dt = float(ctrl_dt)
        self.window_steps = round(ratio)
        self.cursor = 0
        self.previous = np.zeros(num_envs, dtype=np.float64)
        self.unwrapped = np.zeros(num_envs, dtype=np.float64)
        self.history = np.zeros((num_envs, self.window_steps + 1), dtype=np.float64)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.rate = np.zeros(num_envs, dtype=np.float32)

    @property
    def full(self):
        return self.steps == self.window_steps

    def reset(self, env_ids, yaw):
        ids = np.asarray(env_ids, dtype=np.int64)
        values = np.asarray(yaw, dtype=np.float64)
        if values.shape != ids.shape or not np.isfinite(values).all():
            raise ValueError("Reset yaw must contain one finite angle per selected row")
        self.previous[ids] = values
        self.unwrapped[ids] = 0.
        self.history[ids] = 0.
        self.steps[ids] = 0
        self.rate[ids] = 0.

    def update(self, yaw):
        values = np.asarray(yaw, dtype=np.float64)
        if values.shape != self.previous.shape or not np.isfinite(values).all():
            raise ValueError("Yaw must contain one finite angle per environment")
        delta = values - self.previous
        self.unwrapped += np.arctan2(np.sin(delta), np.cos(delta))
        self.previous[:] = values
        self.cursor = (self.cursor + 1) % (self.window_steps + 1)
        self.history[:, self.cursor] = self.unwrapped
        self.steps[:] = np.minimum(self.steps + 1, self.window_steps)
        oldest = (self.cursor - self.steps) % (self.window_steps + 1)
        displacement = self.unwrapped - self.history[np.arange(len(values)), oldest]
        self.rate[:] = displacement / (self.steps * self.ctrl_dt)
        return self.rate


def quaternion_yaw(quaternion):
    """Return yaw from the simulator's xyzw quaternion convention."""
    x, y, z, w = np.asarray(quaternion).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@configclass
class StraightCfgV2(ScratchCfg):
    cycle_yaw_window_seconds: float = .5
    cycle_yaw_rate_cost: float = 120.

    def validate(self):
        super().validate()
        if not math.isclose(self.ctrl_dt, .02) or not math.isclose(self.cycle_yaw_window_seconds, .5):
            raise ValueError("Straight v2 retains 20 ms control and a 0.5 s yaw window")
        if not math.isfinite(self.cycle_yaw_rate_cost) or self.cycle_yaw_rate_cost < 0.:
            raise ValueError("cycle_yaw_rate_cost must be finite and nonnegative")
        if self.heading_command or not self.enforce_attitude_protection or self.only_positive_rewards:
            raise ValueError("Straight v2 requires raw hardware commands, existing protections and signed rewards")
        # This focused version matches the existing mild randomization recipe.
        # Restrict other events so nominal rows cannot receive hidden pushes,
        # motor changes or delayed actions.
        allowed = {"randomize_friction", "randomize_init_joint_offset", "randomize_initial_velocity"}
        for name, value in vars(self.randomization).items():
            if isinstance(value, bool) and value and name not in allowed:
                raise ValueError(f"Straight v2 mild randomization does not enable {name}")
        if not self.randomization.randomize_friction:
            raise ValueError("Straight v2 requires friction overrides; use --randomize or an explicit mild recipe")


class StraightEnvV2(ScratchEnv):
    """Keep every fourth row nominal; preserve the other rows' mild recipe."""

    def __init__(self, cfg, num_envs=128):
        if num_envs < 4 or num_envs % 4:
            raise ValueError("Straight v2 needs a multiple of four environments for the exact 1/4 cohort")
        self.nominal_mask = np.arange(num_envs) % 4 == 0
        self.nominal_ids = np.flatnonzero(self.nominal_mask)
        self.randomized_ids = np.flatnonzero(~self.nominal_mask)
        self.cohort_episode_resets = np.zeros(num_envs, dtype=np.int64)
        self.yaw_window = CircularYawWindow(num_envs, cfg.ctrl_dt, cfg.cycle_yaw_window_seconds)
        super().__init__(cfg, num_envs)

    def cohort_evidence(self):
        return dict(nominal_rule="env_id_mod_4_equals_0", nominal_ids=self.nominal_ids.tolist(),
                    randomized_ids=self.randomized_ids.tolist(), nominal_count=len(self.nominal_ids),
                    randomized_count=len(self.randomized_ids), nominal_fraction=.25,
                    nominal_sliding_friction=.9, nominal_initial_joint_offset=0.,
                    nominal_initial_base_velocity=0., nominal_observation_noise=0.,
                    commands="Both cohorts retain the same standing/walking command sampling",
                    initial_position=[0., 0., float(self.cfg.initial_height)])

    def _sample_noise(self, env_ids):
        # Keep the original RNG sampling order; only discard nominal-row noise.
        super()._sample_noise(env_ids)
        ids = np.asarray(env_ids, dtype=np.int64)
        self.actor_noise[ids[self.nominal_mask[ids]]] = 0.

    def _randomize_episode(self, env_ids):
        super()._randomize_episode(env_ids)
        ids = np.asarray(env_ids, dtype=np.int64)
        nominal = ids[self.nominal_mask[ids]]
        for name, value in (("kp_factors", 1.), ("kd_factors", 1.), ("motor_strength", 1.),
                            ("motor_zero_offsets", 0.), ("init_joint_offsets", 0.),
                            ("initial_joint_factors", 1.)):
            getattr(self, name)[nominal] = value
        for name, buffer in self.physical.items():
            buffer[nominal] = getattr(self.sim, "nominal_" + name)
        self.physical["friction"][nominal, :, 0] = .9

    def reset(self, env_ids):
        ids = np.asarray(env_ids, dtype=np.int64)
        if not len(ids):
            return
        super().reset(ids)
        nominal = ids[self.nominal_mask[ids]]
        if len(nominal):
            # The inherited reset draws initial base velocities for every row.
            # Restore nominal rows, then reapply their physical overrides because
            # the native reset restores model defaults.
            self.reset_write.buffer("q")[nominal] = self.default_q
            self.reset_write.buffer("qvel")[nominal] = 0.
            self.reset_write.buffer("lin_vel")[nominal] = 0.
            self.reset_write.buffer("ang_vel")[nominal] = 0.
            self.reset_write.execute(nominal)
            self.sim.set_physical_properties(nominal, **self.physical)
            self.sim_data.execute(nominal)
            self._derive_state()
            self.last_dof_vel[nominal] = self.sim_data["qvel"][nominal]
            self.prev_world_lin_vel[nominal] = self.sim_data["world_lin_vel"][nominal]
            self.base_lin_acc[nominal] = 0.
            self.position_window[nominal] = self.sim_data["base_pos"][nominal, None]
            self.position_window_steps[nominal] = 0
            self.cycle_world_velocity[nominal] = 0.
            self.cycle_body_velocity[nominal] = 0.
        self.yaw_window.reset(ids, quaternion_yaw(self.sim_data["base_quat"][ids]))
        self.cohort_episode_resets[ids] += 1

    def _reward_terms(self, phase):
        terms = super()._reward_terms(phase)
        rate = self.yaw_window.update(quaternion_yaw(self.sim_data["base_quat"]))
        complete = self.yaw_window.full
        # Replace instantaneous gyro damping with completed-cycle net yaw. Keep
        # the zero-valued old key so inherited configurations remain explicit.
        terms["yaw_rate"] = np.zeros(self.num_envs, dtype=np.float32)
        terms["cycle_yaw_rate"] = np.asarray(
            -self.cfg.cycle_yaw_rate_cost * rate**2 * complete * self.cfg.ctrl_dt, dtype=np.float32)
        self.scratch_reward_diagnostics.update(cycle_yaw_rate=rate.copy(),
            cycle_yaw_window_complete=complete.copy(), nominal_cohort=self.nominal_mask.copy())
        return terms

    def compute_transition(self, state):
        state = super().compute_transition(state)
        self.transition_data.update(cycle_yaw_rate=self.yaw_window.rate.copy(),
            cycle_yaw_window_complete=self.yaw_window.full.copy(), nominal_cohort=self.nominal_mask.copy())
        return state
