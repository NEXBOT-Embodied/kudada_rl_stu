"""From-scratch walking rewards on the exact Tinker deployment interface.

The gait reference is a *training reward only*.  Actions passed to the simulator
always come from the learned policy, through TinkerEnv's controller decoding.
No reference pose, simulator velocity, position, or heading is added to actor
observations.  The asymmetric critic may use the existing privileged state.
"""

import math

import numpy as np

from motrix_env_core.config import configclass
from kudada_motrix.contract import rotate_inverse
from kudada_motrix.randomization import DomainRandCfg
from tinker_policy.env import TinkerCfg, TinkerEnv


def scratch_randomization():
    """Allocate physical overrides initially; curriculum widens their ranges."""
    return DomainRandCfg(
        randomize_friction=True, friction_range=(.88, .92),
        randomize_init_joint_offset=True, init_joint_offset_range=(-.003, .003),
        randomize_initial_velocity=True, initial_velocity_range=(-.003, .003),
    )


@configclass
class ScratchCfg(TinkerCfg):
    """Walking acquisition without a pretrained network or action prior."""

    training_speed: float = .2
    curriculum_speed_start: float = .14
    standing_fraction: float = .25
    command_resampling_seconds: float = 8.
    max_episode_seconds: float = 20.
    enforce_action_bounds: bool = True
    add_noise: bool = True
    noise_level: float = .1
    randomization: DomainRandCfg = scratch_randomization()
    curriculum_randomization: bool = True
    target_height: float = .275
    stance_fraction: float = .6
    foot_origin_ground_height: float = .044
    swing_clearance: float = .025
    hip_pitch_amplitude: float = .18
    knee_swing_amplitude: float = .45
    support_roll_amplitude: float = .045
    gait_reward_scale: float = 1.5
    pose_reward_scale: float = 4.
    clearance_reward_scale: float = .6
    speed_reward_scale: float = 4.
    progress_reward_scale: float = 3.
    progress_window_seconds: float = .5
    wrong_phase_liftoff_scale: float = 1.
    speed_tracking_sigma: float = .10
    lateral_cost: float = 2.
    heading_cost: float = 1.
    survival_reward_scale: float = .3
    enforce_attitude_protection: bool = True

    def validate(self):
        super().validate()
        values = [self.training_speed, self.curriculum_speed_start,
                  self.target_height, self.foot_origin_ground_height,
                  self.swing_clearance, self.hip_pitch_amplitude,
                  self.knee_swing_amplitude, self.support_roll_amplitude,
                  self.gait_reward_scale, self.pose_reward_scale,
                  self.clearance_reward_scale, self.speed_reward_scale,
                  self.progress_reward_scale, self.progress_window_seconds,
                  self.wrong_phase_liftoff_scale,
                  self.speed_tracking_sigma, self.lateral_cost, self.heading_cost,
                  self.survival_reward_scale]
        if not all(math.isfinite(v) and v >= 0. for v in values):
            raise ValueError('Scratch reward and curriculum values must be finite and nonnegative')
        if not .1 < self.curriculum_speed_start <= self.training_speed:
            raise ValueError('Curriculum speed must exceed the hardware phase-freeze threshold of .1 m/s')
        if self.speed_tracking_sigma <= 0. or self.target_height <= 0.:
            raise ValueError('Tracking sigma and target height must be positive')
        if self.progress_window_seconds < self.ctrl_dt or not math.isclose(
                self.progress_window_seconds/self.ctrl_dt,
                round(self.progress_window_seconds/self.ctrl_dt)):
            raise ValueError('Progress window must contain a positive integer number of control steps')
        if not .5 <= self.stance_fraction < .8:
            raise ValueError('stance_fraction must be in [.5,.8)')
        if not 0. <= self.standing_fraction <= 1.:
            raise ValueError('standing_fraction must be in [0,1]')
        if not self.enforce_action_bounds:
            raise ValueError('Scratch deployment training requires physical action bounds')


class ScratchEnv(TinkerEnv):
    """Train a full action policy; retain TinkerEnv's sensor and control contract.

    ``policy_input()`` is chronological [N,924], with a zero-filled initial
    history. ``privileged()`` is [N,13] and is intended for the critic only.
    ``transition_data`` is captured before automatic episode resets, including
    the terminal sensor frame needed for correct time-limit bootstrapping.
    """

    def __init__(self, cfg, num_envs=1):
        self.curriculum_progress = 0.
        self.final_training_speed = float(cfg.training_speed)
        self.current_training_speed = float(cfg.curriculum_speed_start)
        self.final_noise_level = float(cfg.noise_level)
        self.current_noise_level = .5*self.final_noise_level
        self.attitude_protection_time = np.zeros(num_envs, dtype=np.float32)
        self.scratch_reward_diagnostics = {}
        super().__init__(cfg, num_envs)
        self.velocity_window_steps = round(cfg.progress_window_seconds/cfg.ctrl_dt)
        self.position_window = np.zeros((num_envs, self.velocity_window_steps+1, 3), dtype=np.float32)
        self.position_window_steps = np.zeros(num_envs, dtype=np.int64)
        self.position_window_cursor = 0
        self.cycle_world_velocity = np.zeros((num_envs, 3), dtype=np.float32)
        self.cycle_body_velocity = np.zeros((num_envs, 3), dtype=np.float32)
        self.set_curriculum(0.)

    def set_curriculum(self, progress):
        """Advance deterministic difficulty, normally called once per PPO update.

        The final stage samples .2 m/s walking and zero-command standing. Fixed
        evaluation commands are never changed by this method. No simulator-only
        quantity can enter the actor through the curriculum.
        """
        if not math.isfinite(progress) or not 0. <= progress <= 1.:
            raise ValueError('Curriculum progress must be in [0,1]')
        self.curriculum_progress = float(progress)
        self.current_training_speed = (self.cfg.curriculum_speed_start
                                       + progress*(self.final_training_speed-self.cfg.curriculum_speed_start))
        self.current_noise_level = self.final_noise_level*(.5+.5*progress)
        # Noise scales were cached by the base constructor.
        self.noise_scale[:] = 0.
        self.noise_scale[8:11] = .35*.25*self.current_noise_level
        self.noise_scale[11:14] = .07*self.current_noise_level
        self.noise_scale[14:24] = .01*self.current_noise_level
        self.noise_scale[24:34] = 1.5*.05*self.current_noise_level
        if self.cfg.curriculum_randomization:
            rand = self.cfg.randomization
            if rand.randomize_friction:
                rand.friction_range = (.88-.13*progress, .92+.13*progress)
            if rand.randomize_init_joint_offset:
                scale = .003+.007*progress
                rand.init_joint_offset_range = (-scale, scale)
            if rand.randomize_initial_velocity:
                scale = .003+.007*progress
                rand.initial_velocity_range = (-scale, scale)
        # Commands already held in active episodes also follow the curriculum.
        if self.cfg.fixed_command is None:
            moving_ids = self.commands[:, 0] > .1
            self.commands[moving_ids, 0] = self.current_training_speed
        return dict(progress=self.curriculum_progress,
                    training_speed=float(self.current_training_speed),
                    noise_level=float(self.current_noise_level))

    def _sample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.fixed_command is not None:
            self.commands[env_ids] = self.cfg.fixed_command
        else:
            self.commands[env_ids] = [self.current_training_speed, 0., 0.]
            standing = self.rng.random(len(env_ids)) < self.cfg.standing_fraction
            self.commands[env_ids[standing]] = 0.
        self.command_samples[env_ids] += 1

    def training_reference(self):
        """Return a soft analytic pose/contact reference, used only in reward.

        Swing is the final 40 percent of each leg cycle; the opposite leg is
        half a cycle ahead. Ankle compensation keeps the sole approximately
        level. Roll reference shifts weight toward the expected stance leg.
        """
        phase = np.stack((self.phase, (self.phase+.5) % 1.), axis=1)
        stance = phase < self.cfg.stance_fraction
        swing_progress = np.clip((phase-self.cfg.stance_fraction)
                                 /(1.-self.cfg.stance_fraction), 0., 1.)
        swing_wave = np.sin(np.pi*swing_progress)
        moving = self.filtered_commands[:, 0] >= .1
        # A gentle acquisition stage grows into the complete walking reference.
        difficulty = .75+.25*self.curriculum_progress
        hip_delta = -self.cfg.hip_pitch_amplitude*difficulty*np.cos(2*np.pi*phase)
        knee_delta = self.cfg.knee_swing_amplitude*difficulty*swing_wave
        roll_delta = -self.cfg.support_roll_amplitude*np.sin(2*np.pi*self.phase)
        reference = np.broadcast_to(self.default_q, self.actions.shape).copy()
        for leg, offset in enumerate((0, 5)):
            reference[:, offset+1] += roll_delta*moving
            reference[:, offset+2] += hip_delta[:, leg]*moving
            reference[:, offset+3] += knee_delta[:, leg]*moving
            reference[:, offset+4] -= (hip_delta[:, leg]+knee_delta[:, leg])*moving
        reference = np.clip(reference, self.physical_ranges[:, 0]+1.e-4,
                            self.physical_ranges[:, 1]-1.e-4)
        clearance = (self.cfg.foot_origin_ground_height
                     + self.cfg.swing_clearance*difficulty*swing_wave*moving[:, None])
        expected_contact = stance | ~moving[:, None]
        return reference.astype(np.float32), expected_contact, clearance.astype(np.float32)

    def _update_progress_velocity(self, position, quaternion):
        # Measure signed displacement over a complete gait cycle. Instantaneous
        # base velocity can reward rocking with alternating positive/negative
        # velocities while the robot makes no progress. This state remains
        # privileged; it never changes the 44-value hardware actor frame.
        cursor = (self.position_window_cursor+1) % (self.velocity_window_steps+1)
        self.position_window[:, cursor] = position
        self.position_window_steps[:] = np.minimum(self.position_window_steps+1,
                                                  self.velocity_window_steps)
        oldest = (cursor-self.position_window_steps) % (self.velocity_window_steps+1)
        previous_position = self.position_window[np.arange(self.num_envs), oldest]
        self.cycle_world_velocity[:] = ((position-previous_position)
                                       /(self.position_window_steps*self.cfg.ctrl_dt)[:, None])
        self.cycle_body_velocity[:] = rotate_inverse(quaternion, self.cycle_world_velocity)
        self.position_window_cursor = cursor

    def _reward_terms(self, phase):
        # TinkerEnv's hardware phase, rather than the legacy task phase, is the
        # one present in the actor's latest sensor frame.
        data = self.sim_data
        self._update_progress_velocity(data['base_pos'], data['base_quat'])
        velocity = self.base_lin_vel
        gravity = self.projected_gravity
        height = data['base_pos'][:, 2]
        tilt = np.sum(gravity[:, :2]**2, axis=1)
        moving = self.filtered_commands[:, 0] >= .1
        contact = data['contacts'][:, 1:3, 2] > 5.
        reference, expected_contact, clearance = self.training_reference()
        pose_error = ((data['q']-reference)
                      /np.array([.15, .12, .22, .3, .22]*2, dtype=np.float32))
        reference_score = np.exp(-np.mean(pose_error**2, axis=1))
        match = np.mean(contact == expected_contact, axis=1)
        feet_z = data['feet_pos'][:, :, 2]
        swing = (~expected_contact) & moving[:, None]
        clearance_score = (np.exp(-((feet_z-clearance)/.025)**2)*swing*~contact).sum(axis=1)
        incorrect_liftoff = np.sum((~contact) & expected_contact & moving[:, None], axis=1)
        feet_slip = np.sum(np.sum(data['feet_vel'][:, :, :2]**2, axis=2)*contact, axis=1)
        q = data['base_quat']
        yaw = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]),
                         1-2*(q[:, 1]**2+q[:, 2]**2))
        command = self.filtered_commands[:, 0]
        sigma = np.where(moving, self.cfg.speed_tracking_sigma, .05)
        progress_velocity = self.cycle_body_velocity[:, 0]
        instant_planar_speed = np.linalg.norm(velocity[:, :2], axis=1)
        speed_error = np.where(moving, progress_velocity-command, instant_planar_speed)
        healthy = ((height > .18) & (tilt < .36)
                   & (np.linalg.norm(data['contacts'][:, 0, :3], axis=1) <= 1.))
        failure = ((height < .15) | (tilt > .64)
                   | (np.linalg.norm(data['contacts'][:, 0, :3], axis=1) > 1.))
        # At rest pose preference is soft; it cannot outweigh velocity tracking
        # when a walking command is present. No positive-only reward clipping.
        raw = dict(
            speed=self.cfg.speed_reward_scale*np.exp(-(speed_error/sigma)**2),
            forward=self.cfg.progress_reward_scale*np.clip(
                progress_velocity/np.maximum(command, .1), -1., 1.)*moving,
            alive=self.cfg.survival_reward_scale*healthy,
            upright=1.2*np.exp(-tilt/.06),
            height=.6*np.exp(-((height-self.cfg.target_height)/.035)**2),
            gait_contact=self.cfg.gait_reward_scale*(match-self.cfg.stance_fraction)*moving,
            gait_pose=-self.cfg.pose_reward_scale*np.mean(pose_error**2, axis=1)*moving,
            swing_clearance=self.cfg.clearance_reward_scale*clearance_score,
            incorrect_liftoff=-self.cfg.wrong_phase_liftoff_scale*incorrect_liftoff,
            standing_pose=.4*reference_score*~moving,
            stand_support=.4*np.all(contact, axis=1)*~moving,
            no_fly=-.8*~np.any(contact, axis=1),
            lateral_velocity=-8.*velocity[:, 1]**2,
            path=-self.cfg.lateral_cost*data['base_pos'][:, 1]**2*moving,
            heading=-self.cfg.heading_cost*yaw**2,
            roll_pitch_rate=-.10*np.sum(self.base_ang_vel[:, :2]**2, axis=1),
            yaw_rate=-.3*self.base_ang_vel[:, 2]**2,
            vertical=-.3*velocity[:, 2]**2,
            foot_slip=-.08*feet_slip,
            torque=-5.e-5*np.sum(self.torques**2, axis=1),
            joint_velocity=-2.e-4*np.sum(data['qvel']**2, axis=1),
            action_rate=-.01*np.sum((self.actions-self.last_actions)**2, axis=1),
            action_acceleration=-.005*np.sum((self.actions+self.last_last_actions
                                             -2*self.last_actions)**2, axis=1),
            failure=-30.*failure,
            protection=-30.*self.velocity_protection_triggered,
        )
        self.scratch_reward_diagnostics = dict(
            actual_forward_velocity=velocity[:, 0].copy(),
            instant_planar_speed=instant_planar_speed.copy(),
            speed_tracking_error=speed_error.copy(),
            progress_forward_velocity=progress_velocity.copy(),
            reference_pose_error=np.mean(pose_error**2, axis=1),
            gait_contact_match=match.copy(),
            swing_clearance=clearance_score.copy(),
            moving=moving.copy(),
        )
        return {name: np.asarray(value*self.cfg.ctrl_dt, dtype=np.float32)
                for name, value in raw.items()}

    def compute_transition(self, state):
        state = super().compute_transition(state)
        # Reject actions whose resulting execution would stop on the existing
        # hardware controller, rather than merely reporting its watchdog flag.
        protection = self.velocity_protection_triggered.copy()
        outside = self.target_outside_physical_range.copy()
        quat = self.sim_data['base_quat']
        x, y, z, w = quat.T
        roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = np.arcsin(np.clip(2*(w*y-z*x), -1., 1.))
        attitude_exceeded = (np.abs(roll) > np.deg2rad(25.)) | (np.abs(pitch) > np.deg2rad(35.))
        self.attitude_protection_time[:] = np.where(
            attitude_exceeded, self.attitude_protection_time+self.cfg.ctrl_dt, 0.)
        attitude = ((self.attitude_protection_time > .02+1.e-7)
                    & self.cfg.enforce_attitude_protection)
        state.terminated |= protection | outside | attitude
        # The new protection criterion is evaluated after the parent reward.
        attitude_reward = np.asarray(-30.*self.cfg.ctrl_dt*attitude, dtype=np.float32)
        state.reward_terms['attitude_protection'] = attitude_reward
        state.reward += attitude_reward
        state.metrics['termination_velocity_protection'] = protection
        state.metrics['termination_target_range'] = outside
        state.metrics['termination_attitude_protection'] = attitude.copy()
        state.metrics.update(self.scratch_reward_diagnostics)
        self.transition_data['terminated'] = state.terminated.copy()
        self.transition_data['termination_reasons'].update(
            velocity_protection=protection, target_range=outside,
            attitude_protection=attitude.copy())
        self.transition_data['feet_positions'] = self.sim_data['feet_pos'].copy()
        self.transition_data['phase'] = self.phase.copy()
        self.transition_data['cycle_velocity_body'] = self.cycle_body_velocity.copy()
        self.transition_data['cycle_velocity_world'] = self.cycle_world_velocity.copy()
        return state

    def reset(self, env_ids):
        super().reset(env_ids)
        self.attitude_protection_time[env_ids] = 0.
        self.position_window[env_ids] = self.sim_data['base_pos'][env_ids, None]
        self.position_window_steps[env_ids] = 0
        self.cycle_world_velocity[env_ids] = 0.
        self.cycle_body_velocity[env_ids] = 0.

    def privileged(self, terminal=False):
        original = super().privileged(terminal=terminal)
        averaged_velocity = (self.transition_data['cycle_velocity_body'] if terminal
                             else self.cycle_body_velocity)
        return np.concatenate((original, averaged_velocity*2.), axis=1).astype(np.float32)


# Explicit aliases support consumers that use the descriptive class names.
ScratchWalkingConfig = ScratchCfg
ScratchWalkingEnv = ScratchEnv
