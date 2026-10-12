"""Own-policy yaw-command curriculum with training-only path references.

The ordinary five-frame actor and 924-by-11 hardware contract are unchanged.
Nominal rows always receive zero yaw; a subset of randomized walking rows learn
small signed steering commands. Reference coordinates enter rewards/critic only.
"""

from __future__ import annotations

import math

import numpy as np

from motrix_env_core.config import configclass
from kudada_motrix.randomization import DomainRandCfg
from scratch_policy.straight_env_v2 import (
    CircularYawWindow, StraightCfgV2, StraightEnvV2, quaternion_yaw,
)

METHOD = "PPO_yaw_course_v7"
PARENT_CHECKPOINT_SHA256 = "2ef29e15aabd10bdaed5e072aa4666ddd488df6498c399054f062069c1c99e0e"


def yaw_course_randomization_v7():
    return DomainRandCfg(
        randomize_friction=True, friction_range=(.75, 1.05),
        randomize_init_joint_offset=True, init_joint_offset_range=(-.01, .01),
        randomize_initial_velocity=True, initial_velocity_range=(-.01, .01),
    )


class CommandPathReferenceV7:
    """Integrate physical filtered body commands once per executed control.

    Constant-twist integration is exact inside each dt. The yaw target window
    has the same duration, sample count and per-row resets as the actual-yaw
    window. This class neither reads a simulator nor supplies actor inputs.
    """

    def __init__(self, num_envs, ctrl_dt, window_seconds=.5):
        self.ctrl_dt = float(ctrl_dt)
        self.position = np.zeros((num_envs, 2), dtype=np.float64)
        self.heading = np.zeros(num_envs, dtype=np.float64)
        self.yaw_window = CircularYawWindow(num_envs, ctrl_dt, window_seconds)

    def reset(self, env_ids, position, yaw):
        ids = np.asarray(env_ids, dtype=np.int64)
        xy = np.asarray(position, dtype=np.float64)
        angles = np.asarray(yaw, dtype=np.float64)
        if xy.shape != (len(ids), 2) or angles.shape != ids.shape:
            raise ValueError("Reference reset requires one xy position and yaw per selected row")
        if not np.isfinite(xy).all() or not np.isfinite(angles).all():
            raise ValueError("Reference reset state must be finite")
        self.position[ids] = xy
        self.heading[ids] = angles
        self.yaw_window.reset(ids, angles)

    def advance(self, filtered_commands):
        commands = np.asarray(filtered_commands, dtype=np.float64)
        if commands.shape != (len(self.heading), 3) or not np.isfinite(commands).all():
            raise ValueError("Reference requires finite physical [vx,vy,yaw_rad_s] commands")
        half_angle = .5 * commands[:, 2] * self.ctrl_dt
        midpoint_heading = self.heading + half_angle
        interval = self.ctrl_dt * np.sinc(half_angle / np.pi)
        cosine, sine = np.cos(midpoint_heading), np.sin(midpoint_heading)
        self.position[:, 0] += interval * (cosine*commands[:, 0] - sine*commands[:, 1])
        self.position[:, 1] += interval * (sine*commands[:, 0] + cosine*commands[:, 1])
        self.heading += 2.*half_angle
        self.yaw_window.update(self.heading)

    def errors(self, position, yaw):
        return course_reference_errors_v7(position, yaw, self.position, self.heading)


def course_reference_errors_v7(position, yaw, reference_position, reference_heading):
    position = np.asarray(position, dtype=np.float64)
    heading = np.asarray(reference_heading, dtype=np.float64)
    delta = position[:, :2] - np.asarray(reference_position, dtype=np.float64)
    path_error = -np.sin(heading)*delta[:, 0] + np.cos(heading)*delta[:, 1]
    angle_error = np.asarray(yaw, dtype=np.float64) - heading
    heading_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
    return path_error.astype(np.float32), heading_error.astype(np.float32)


@configclass
class YawCourseCfgV7(StraightCfgV2):
    curriculum_speed_start: float = .2
    standing_fraction: float = .35
    randomized_turning_fraction_of_moving: float = .35
    yaw_command_min_rad_s: float = .08
    yaw_command_max_rad_s: float = .16
    yaw_curriculum_updates: int = 25
    randomization: DomainRandCfg = yaw_course_randomization_v7()

    def validate(self):
        super().validate()
        values = (self.randomized_turning_fraction_of_moving,
                  self.yaw_command_min_rad_s, self.yaw_command_max_rad_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Yaw course command values must be finite")
        if not 0. <= self.randomized_turning_fraction_of_moving <= 1.:
            raise ValueError("randomized_turning_fraction_of_moving must be in [0,1]")
        if not 0. < self.yaw_command_min_rad_s <= self.yaw_command_max_rad_s <= .16:
            raise ValueError("Yaw course final command magnitudes must be positive and at most .16 rad/s")
        if type(self.yaw_curriculum_updates) is not int or self.yaw_curriculum_updates < 1:
            raise ValueError("yaw_curriculum_updates must be a positive integer")
        if not np.allclose(self.randomization.friction_range, (.75, 1.05), rtol=0., atol=1.e-8):
            raise ValueError("Yaw course retains randomized friction [.75,1.05]")
        if self.fixed_command is not None:
            raise ValueError("Yaw course training requires sampled commands, not a fixed-command override")


class YawCourseEnvV7(StraightEnvV2):
    def __init__(self, cfg, num_envs=128):
        self.course_stage_updates = 0
        self.course_yaw_gain = 0.
        self.course_base_yaw_commands = np.zeros(num_envs, dtype=np.float32)
        # Columns stand/straight/positive-turn/negative-turn, rows nominal/random.
        # Turn counts use sampled final amplitudes even while curriculum gain=0.
        self.course_command_sample_counts = np.zeros((2, 4), dtype=np.int64)
        self.course_reference = CommandPathReferenceV7(
            num_envs, cfg.ctrl_dt, cfg.cycle_yaw_window_seconds)
        super().__init__(cfg, num_envs)

    def set_curriculum(self, progress):
        result = super().set_curriculum(progress)
        # Friction is the identified factor under study; train its full range
        # throughout this stage while retaining inherited noise/init scheduling.
        self.cfg.randomization.friction_range = (.75, 1.05)
        return dict(result, friction_range=[.75, 1.05])

    def set_course_stage_update(self, stage_updates_completed):
        if type(stage_updates_completed) is not int or stage_updates_completed < 0:
            raise ValueError("Course stage update must be a nonnegative integer")
        self.course_stage_updates = stage_updates_completed
        self.course_yaw_gain = min(1., stage_updates_completed/self.cfg.yaw_curriculum_updates)
        self.commands[:, 2] = self.course_base_yaw_commands * self.course_yaw_gain
        self.commands[self.nominal_mask, 2] = 0.
        return dict(stage_updates_completed=stage_updates_completed,
                    yaw_curriculum_gain=self.course_yaw_gain,
                    final_yaw_magnitude_range_rad_s=[self.cfg.yaw_command_min_rad_s,
                                                    self.cfg.yaw_command_max_rad_s])

    def _sample_commands(self, env_ids):
        ids = np.asarray(env_ids, dtype=np.int64)
        if not len(ids):
            return
        self.commands[ids] = [self.current_training_speed, 0., 0.]
        self.course_base_yaw_commands[ids] = 0.
        standing = self.rng.random(len(ids)) < self.cfg.standing_fraction
        self.commands[ids[standing]] = 0.
        eligible = ids[~standing & ~self.nominal_mask[ids]]
        turning = eligible[self.rng.random(len(eligible)) < self.cfg.randomized_turning_fraction_of_moving]
        magnitude = self.rng.uniform(self.cfg.yaw_command_min_rad_s,
                                     self.cfg.yaw_command_max_rad_s, len(turning))
        signs = np.where(self.rng.random(len(turning)) < .5, -1., 1.)
        self.course_base_yaw_commands[turning] = magnitude*signs
        self.commands[turning, 2] = self.course_base_yaw_commands[turning]*self.course_yaw_gain
        modes = np.where(standing, 0, 1)
        positive = self.course_base_yaw_commands[ids] > 0.
        negative = self.course_base_yaw_commands[ids] < 0.
        modes[positive], modes[negative] = 2, 3
        cohorts = (~self.nominal_mask[ids]).astype(np.int64)
        np.add.at(self.course_command_sample_counts, (cohorts, modes), 1)
        self.command_samples[ids] += 1

    def reset(self, env_ids):
        ids = np.asarray(env_ids, dtype=np.int64)
        if not len(ids):
            return
        super().reset(ids)
        self.course_reference.reset(ids, self.sim_data['base_pos'][ids, :2],
                                    quaternion_yaw(self.sim_data['base_quat'][ids]))

    def _reward_terms(self, phase):
        # C_t is still the filtered command in the observation for the action
        # just executed. compute_observation will filter C_(t+1) only later.
        self.course_reference.advance(self.filtered_commands)
        terms = super()._reward_terms(phase)
        path_error, heading_error = self.course_reference.errors(
            self.sim_data['base_pos'], quaternion_yaw(self.sim_data['base_quat']))
        moving = self.filtered_commands[:, 0] >= .1
        target_rate = self.course_reference.yaw_window.rate
        rate_error = self.yaw_window.rate - target_rate
        complete = self.yaw_window.full & self.course_reference.yaw_window.full
        dt = self.cfg.ctrl_dt
        terms['path'] = np.asarray(-self.cfg.lateral_cost*path_error**2*moving*dt, dtype=np.float32)
        terms['heading'] = np.asarray(-self.cfg.heading_cost*heading_error**2*dt, dtype=np.float32)
        terms['cycle_yaw_rate'] = np.asarray(
            -self.cfg.cycle_yaw_rate_cost*rate_error**2*complete*dt, dtype=np.float32)
        # StraightEnvV2 already zeroes the instantaneous gyro damping term.
        self.scratch_reward_diagnostics.update(
            course_path_error=path_error.copy(), course_heading_error=heading_error.copy(),
            course_target_cycle_yaw_rate=target_rate.copy(), course_cycle_yaw_rate_error=rate_error.copy(),
            course_yaw_curriculum_gain=np.full(self.num_envs, self.course_yaw_gain, dtype=np.float32))
        return terms

    def compute_transition(self, state):
        state = super().compute_transition(state)
        # Save before DirectEnv autoresets, including the terminal reference
        # required for timeout-only critic bootstrap.
        self.transition_data.update(
            course_reference_position=self.course_reference.position.copy(),
            course_reference_heading=self.course_reference.heading.copy(),
            course_filtered_commands=self.filtered_commands.copy(),
            course_path_error=self.scratch_reward_diagnostics['course_path_error'].copy(),
            course_heading_error=self.scratch_reward_diagnostics['course_heading_error'].copy(),
            course_target_cycle_yaw_rate=self.course_reference.yaw_window.rate.copy(),
            course_cycle_yaw_rate_error=self.scratch_reward_diagnostics['course_cycle_yaw_rate_error'].copy())
        return state

    def privileged(self, terminal=False):
        original = super().privileged(terminal=terminal)
        if terminal:
            data = self.transition_data
            path_error, heading_error = course_reference_errors_v7(
                data['position'], data['yaw'], data['course_reference_position'],
                data['course_reference_heading'])
        else:
            path_error, heading_error = self.course_reference.errors(
                self.sim_data['base_pos'], quaternion_yaw(self.sim_data['base_quat']))
        # Retain [N,13] privileged / [N,233] critic widths. Only the existing
        # y/sin(yaw)/cos(yaw) meanings change; the weights can be inherited but
        # general turning-state value-function equality is not asserted.
        original[:, 3] = path_error
        original[:, 5] = np.sin(heading_error)
        original[:, 6] = np.cos(heading_error)
        return original

    def cohort_evidence(self):
        original = super().cohort_evidence()
        original['commands'] = 'Nominal yaw=0; randomized walking rows sample signed yaw commands'
        original['yaw_course'] = self.course_evidence()
        return original

    def course_evidence(self):
        return dict(
            schema='scratch_yaw_course_v7', standing_fraction=self.cfg.standing_fraction,
            randomized_turning_fraction_of_moving=self.cfg.randomized_turning_fraction_of_moving,
            sign_sampling='independent equiprobable positive/negative',
            nominal_yaw_command_rad_s=0., randomized_friction_range=[.75, 1.05],
            final_yaw_magnitude_range_rad_s=[self.cfg.yaw_command_min_rad_s, self.cfg.yaw_command_max_rad_s],
            yaw_curriculum_updates=self.cfg.yaw_curriculum_updates,
            stage_updates_completed=self.course_stage_updates, yaw_curriculum_gain=self.course_yaw_gain,
            command_sample_count_columns=['stand', 'straight', 'positive_turn', 'negative_turn'],
            nominal_command_sample_counts=self.course_command_sample_counts[0].tolist(),
            randomized_command_sample_counts=self.course_command_sample_counts[1].tolist(),
            sample_counts_describe='sampled final-amplitude modes, including zero-gain course startup',
            reference_reset='each physical episode uses measured initial world xy and heading',
            reference_timing='one constant-twist integration of physical filtered C_t after control, before reward',
            command_resampling_resets_reference=False, actor_receives_world_reference=False,
            privileged_width=13, critic_reference_slots=[3, 5, 6],
            terminal_reference='pre-reset transition snapshot for timeout bootstrap')
