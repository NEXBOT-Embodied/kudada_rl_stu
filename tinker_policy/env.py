"""MotrixLab task with the supplied Tinker observation and actuator contract."""

from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import yaml

from motrix_env_core.config import configclass
from motrix_env_core.array.env import NpObs
from kudada_motrix.cfg import KudadaRealTrainingCfg
from kudada_motrix.constants import COMMAND_SCALE, JOINT_NAMES, MODEL
from kudada_motrix.env import KudadaRealTrainingEnv
from kudada_motrix.randomization import DomainRandCfg

ROOT = Path(__file__).resolve().parents[1]


def hardware_contract():
    gait = yaml.safe_load((ROOT/'references/tinker/param_gait.yaml').read_text(encoding='utf-8'))
    robot = yaml.safe_load((ROOT/'references/tinker/param_robot.yaml').read_text(encoding='utf-8'))
    rl, imp, kin = gait['rl_gait'], gait['imp_param'], robot['kin_param']
    keys = [f'{leg}{joint}' for leg in range(2) for joint in range(5)]
    return dict(
        default_q=[rl[f'def_act{i}'] for i in [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]],
        kp=[imp['kp_q'+k]*imp['stiff_q'+k] for k in keys],
        kd=[imp['kd_q'+k] for k in keys],
        action_scale=rl['action_scale'],
        action_gains=rl.get('action_scale_gains', [1.75, 1.0625, 1.0625, 1.0625, 1.4]*2),
        target_ranges=[[np.deg2rad(kin['min_q'+k]), np.deg2rad(kin['max_q'+k])] for k in keys],
        torque_limits=[kin['tau_max'+k] for k in keys],
        gait_period=rl['rl_gait_duty'], inference_dt=rl['onnx_run_dt'],
        action_filter_rate=50., action_clip=20., command_scale=[2., 2., .25],
        command_filter_alpha=.03, command_deadband=.01, startup_history='zeros',
        reserved_inputs=[0., 0., 0.], history='oldest_first', gait_output_feedback=False,
    )


def mild_randomization():
    return DomainRandCfg(randomize_friction=True, friction_range=(.75, 1.05),
                         randomize_init_joint_offset=True, init_joint_offset_range=(-.01, .01),
                         randomize_initial_velocity=True, initial_velocity_range=(-.01, .01))


def deployment_action_bounds(contract=None):
    """Bound raw outputs so the hardware filter stays within both joint ranges."""
    contract = contract or hardware_contract()
    tree = ET.parse(MODEL)
    physical = np.array([
        np.fromstring(tree.find(f".//joint[@name='{name}']").attrib['range'], sep=' ')
        for name in JOINT_NAMES
    ], dtype=np.float64)
    commanded = np.asarray(contract['target_ranges'], dtype=np.float64)
    # Leave a small margin for float32 filtering and target decoding.
    lower_q = np.maximum(physical[:, 0], commanded[:, 0]) + 1.e-5
    upper_q = np.minimum(physical[:, 1], commanded[:, 1]) - 1.e-5
    default = np.asarray(contract['default_q'], dtype=np.float64)
    gain = contract['action_scale'] * np.asarray(contract['action_gains'], dtype=np.float64)
    if np.any(gain <= 0.) or np.any(default < lower_q) or np.any(default > upper_q):
        raise ValueError('Invalid joint bounds or default pose')
    return dict(lower=np.maximum(-20., (lower_q-default)/gain).astype(np.float32).tolist(),
                upper=np.minimum(20., (upper_q-default)/gain).astype(np.float32).tolist())


@configclass
class TinkerCfg(KudadaRealTrainingCfg):
    initial_height: float = .28
    kp_scale: float = 1.
    command_resampling_seconds: float = 8.
    heading_command: bool = False
    full_rewards: bool = False
    only_positive_rewards: bool = False
    add_noise: bool = False
    noise_level: float = .1
    randomization: DomainRandCfg = DomainRandCfg()
    standing_fraction: float = .3
    training_speed: float = .2
    lateral_cost: float = 8.
    heading_cost: float = 2.
    gait_id_feedback: bool = False
    enforce_action_bounds: bool = False


class TinkerEnv(KudadaRealTrainingEnv):
    def __init__(self, cfg, num_envs=1):
        self.contract = hardware_contract()
        self.default_q = np.array(self.contract['default_q'], dtype=np.float32)
        self.kp = np.array(self.contract['kp'], dtype=np.float32)
        self.kd = np.array(self.contract['kd'], dtype=np.float32)
        self.action_gains = np.array(self.contract['action_gains'], dtype=np.float32)
        self.target_ranges = np.array(self.contract['target_ranges'], dtype=np.float32)
        self.hardware_torque_limits = np.array(self.contract['torque_limits'], dtype=np.float32)
        self.action_bounds = deployment_action_bounds(self.contract)
        self.action_lower = np.asarray(self.action_bounds['lower'], dtype=np.float32)
        self.action_upper = np.asarray(self.action_bounds['upper'], dtype=np.float32)
        self.filtered_actions = np.zeros((num_envs, 10), dtype=np.float32)
        self.phase = np.zeros(num_envs, dtype=np.float32)
        self.filtered_commands = np.zeros((num_envs, 3), dtype=np.float32)
        self.gait_ids = np.ones(num_envs, dtype=np.int64)
        self.previous_targets = np.broadcast_to(self.default_q, (num_envs, 10)).copy()
        self.command_velocity_filtered = np.zeros((num_envs, 10), dtype=np.float32)
        self.velocity_protection_timer = np.zeros(num_envs, dtype=np.float32)
        self.velocity_protection_triggered = np.zeros(num_envs, dtype=bool)
        tree = ET.parse(MODEL)
        self.physical_ranges = np.array([
            np.fromstring(tree.find(f".//joint[@name='{name}']").attrib['range'], sep=' ')
            for name in JOINT_NAMES
        ], dtype=np.float32)
        self.transition_data = None
        super().__init__(cfg, num_envs)
        self._action_space = (gym.spaces.Box(self.action_lower, self.action_upper, dtype=np.float32)
                              if cfg.enforce_action_bounds else gym.spaces.Box(-20., 20., (10,), dtype=np.float32))
        self.noise_scale[:8] = 0.
        self.noise_scale[34:] = 0.

    def _sample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.fixed_command is not None:
            self.commands[env_ids] = self.cfg.fixed_command
        else:
            self.commands[env_ids] = [self.cfg.training_speed, 0., 0.]
            self.commands[env_ids[self.rng.random(len(env_ids)) < self.cfg.standing_fraction]] = 0.
        self.command_samples[env_ids] += 1

    def apply_action(self, actions, state):
        if actions.shape != self.actions.shape or not np.isfinite(actions).all():
            raise ValueError('Expected finite [num_envs,10] actions')
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.actions[:] = (np.clip(actions, self.action_lower, self.action_upper)
                           if self.cfg.enforce_action_bounds else np.clip(actions, -20., 20.))
        r = self.cfg.randomization
        self.delay_substeps[:] = (self.rng.integers(*[r.delay_substeps_range[0], r.delay_substeps_range[1]+1],
                                                  self.num_envs) if r.delay else 0)
        return state

    def physics_step(self):
        self._scheduled_events()
        alpha = self.contract['action_filter_rate']*self.cfg.sim.dt
        self.max_feedback_velocity = np.zeros(self.num_envs, dtype=np.float32)
        self.max_command_velocity = np.zeros(self.num_envs, dtype=np.float32)
        self.target_outside_physical_range = np.zeros(self.num_envs, dtype=bool)
        self.torque_saturation_substeps = np.zeros(self.num_envs, dtype=np.int64)
        for substep in range(self.cfg.sim_substeps):
            self.joint_data.execute()
            action = np.where((substep < self.delay_substeps)[:, None], self.last_actions, self.actions)
            self.filtered_actions += alpha*(action-self.filtered_actions)
            self.target_positions[:] = np.clip(
                self.default_q + self.contract['action_scale']*self.action_gains*self.filtered_actions
                + self.motor_zero_offsets,
                self.target_ranges[:, 0], self.target_ranges[:, 1],
            )
            self.torques[:] = np.clip(
                (self.kp*self.kp_factors*(self.target_positions-self.joint_data['q'])
                 -self.kd*self.kd_factors*self.joint_data['qvel'])*self.motor_strength,
                -self.hardware_torque_limits, self.hardware_torque_limits,
            )
            feedback_velocity = np.abs(self.joint_data['qvel'])*57.3
            command_velocity = np.abs(self.target_positions-self.previous_targets)*57.3/self.cfg.sim.dt
            self.command_velocity_filtered += self.cfg.sim.dt/(.1+self.cfg.sim.dt)*(command_velocity-self.command_velocity_filtered)
            exceed = np.any((feedback_velocity > 1000.) | (self.command_velocity_filtered > 1000.), axis=1)
            self.velocity_protection_timer[:] = np.where(exceed, self.velocity_protection_timer+self.cfg.sim.dt, 0.)
            self.velocity_protection_triggered |= self.velocity_protection_timer > .03
            self.max_feedback_velocity = np.maximum(self.max_feedback_velocity, feedback_velocity.max(axis=1))
            self.max_command_velocity = np.maximum(self.max_command_velocity, self.command_velocity_filtered.max(axis=1))
            self.target_outside_physical_range |= np.any((self.target_positions < self.physical_ranges[:, 0])
                                                        |(self.target_positions > self.physical_ranges[:, 1]),axis=1)
            self.torque_saturation_substeps += np.any(np.abs(self.torques) >= .999*self.hardware_torque_limits,axis=1)
            self.previous_targets[:] = self.target_positions
            self.control_write.execute()
            self.sim.step(1)
        self.physics_steps += self.cfg.sim_substeps

    def _frame_observations(self, phase):
        frame = np.concatenate((
            np.zeros((self.num_envs, 3), dtype=np.float32), self.filtered_commands*COMMAND_SCALE,
            np.stack((np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase)), axis=1),
            self.base_ang_vel*.25, self.projected_gravity,
            self.sim_data['q']-self.default_q, self.sim_data['qvel']*.05, self.actions,
        ), axis=1)
        if self.cfg.add_noise:
            frame += self.actor_noise
        frame = np.clip(frame, -100., 100.).astype(np.float32)
        critic = np.zeros((self.num_envs, 243), dtype=np.float32)
        critic[:, :44] = frame
        critic[:, 44:47] = self.base_lin_vel*2.
        return frame, critic

    def compute_observation(self, state):
        cmd = np.where(np.abs(self.commands) < self.contract['command_deadband'], 0., self.commands)
        self.filtered_commands += self.contract['command_filter_alpha']*(cmd-self.filtered_commands)
        advanced_phase = (self.phase+self.cfg.ctrl_dt/self.contract['gait_period']) % 1.
        freeze = np.all(np.abs(self.filtered_commands) < .1, axis=1) & (advanced_phase < .1)
        self.phase[:] = np.where(freeze, 0., advanced_phase)
        actor, critic = self._frame_observations(self.phase)
        self.history[:, 1:] = self.history[:, :-1].copy()
        self.history[:, 0] = actor
        self._fresh_history[:] = False
        return state.replace(obs=NpObs(policy=self.history.reshape(self.num_envs, -1).copy(), value=critic))

    def reset(self, env_ids):
        super().reset(env_ids)
        self.reset_write.buffer('q')[env_ids] = self.default_q+self.init_joint_offsets[env_ids]
        self.reset_write.execute(env_ids)
        if self.cfg.randomization.physical_enabled:
            self.sim.set_physical_properties(env_ids, **self.physical)
        self.sim_data.execute(env_ids)
        self._derive_state()
        self.phase[env_ids] = 0.
        self.filtered_commands[env_ids] = 0.
        self.filtered_actions[env_ids] = 0.
        self.previous_targets[env_ids] = self.default_q
        self.command_velocity_filtered[env_ids] = 0.
        self.velocity_protection_timer[env_ids] = 0.
        self.velocity_protection_triggered[env_ids] = False
        self.target_positions[env_ids] = self.default_q
        self.prev_world_lin_vel[env_ids] = self.sim_data['world_lin_vel'][env_ids]

    def _reward_terms(self, phase):
        velocity = self.sim_data['world_lin_vel']
        q = self.sim_data['base_quat']
        yaw = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]),
                         1-2*(q[:, 1]**2+q[:, 2]**2))
        height = self.sim_data['base_pos'][:, 2]
        tilt = np.sum(self.projected_gravity[:, :2]**2, axis=1)
        moving = self.filtered_commands[:, 0] >= .1
        contact = self.sim_data['contacts'][:, 1:3, 2] > 5.
        failure = (height < .15) | (tilt > .64) | (np.linalg.norm(self.sim_data['contacts'][:, 0, :3], axis=1) > 1.)
        raw = dict(
            speed=4.*np.exp(-((velocity[:, 0]-self.filtered_commands[:, 0])/.055)**2),
            side=-8.*velocity[:, 1]**2,
            path=-self.cfg.lateral_cost*self.sim_data['base_pos'][:, 1]**2*moving,
            yaw=-self.cfg.heading_cost*yaw**2,
            upright=1.5*np.exp(-tilt/.06),
            height=.5*np.exp(-((height-.275)/.04)**2),
            stand_support=.5*np.all(contact, axis=1)*~moving,
            roll_pitch_rate=-.1*np.sum(self.base_ang_vel[:, :2]**2, axis=1),
            yaw_rate=-.25*self.base_ang_vel[:, 2]**2,
            vertical=-.5*velocity[:, 2]**2,
            torque=-1.e-4*np.sum(self.torques**2, axis=1),
            smooth=-.005*np.sum((self.actions-self.last_actions)**2, axis=1),
            failure=-50.*failure,
        )
        return {k: np.asarray(v*self.cfg.ctrl_dt, dtype=np.float32) for k, v in raw.items()}

    def compute_transition(self, state):
        state = super().compute_transition(state)
        q = self.sim_data['base_quat']
        yaw = np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2))
        self.transition_data = dict(
            position=self.sim_data['base_pos'].copy(), velocity_world=self.sim_data['world_lin_vel'].copy(),
            velocity_body=self.base_lin_vel.copy(), projected_gravity=self.projected_gravity.copy(),
            yaw=yaw.copy(), feet_force_z=self.sim_data['contacts'][:, 1:3, 2].copy(),
            terminated=state.terminated.copy(), truncated=(state.episode_steps+1 >= self.cfg.max_episode_steps),
            termination_reasons={k.removeprefix('termination_'): v.copy() for k, v in state.metrics.items()
                                 if k.startswith('termination_')}, q=self.sim_data['q'].copy(),
            max_feedback_velocity_deg_s=self.max_feedback_velocity.copy(),
            max_command_velocity_deg_s=self.max_command_velocity.copy(),
            velocity_protection_triggered=self.velocity_protection_triggered.copy(),
            target_outside_physical_range=self.target_outside_physical_range.copy(),
            torque_saturation_substeps=self.torque_saturation_substeps.copy(),
        )
        # Capture next actor input before any terminal physical reset.
        cmd = np.where(np.abs(self.commands) < self.contract['command_deadband'], 0., self.commands)
        nextcmd = self.filtered_commands+self.contract['command_filter_alpha']*(cmd-self.filtered_commands)
        advanced_phase = (self.phase+self.cfg.ctrl_dt/self.contract['gait_period']) % 1.
        nextphase = np.where(np.all(np.abs(nextcmd) < .1, axis=1)&(advanced_phase < .1), 0., advanced_phase)
        frame, _ = self._frame_observations(self.phase)
        frame[:, 3:6] = nextcmd*COMMAND_SCALE
        frame[:, 6:8] = np.stack((np.sin(2*np.pi*nextphase), np.cos(2*np.pi*nextphase)), axis=1)
        self.transition_data['next_frame'] = frame.copy()
        return state

    def policy_input(self):
        return self.history[:, ::-1].reshape(self.num_envs, 924).copy()

    def privileged(self, terminal=False):
        data = self.transition_data if terminal else dict(
            position=self.sim_data['base_pos'], velocity_world=self.sim_data['world_lin_vel'],
            projected_gravity=self.projected_gravity,
            yaw=np.arctan2(2*(self.sim_data['base_quat'][:, 3]*self.sim_data['base_quat'][:, 2]
                              +self.sim_data['base_quat'][:, 0]*self.sim_data['base_quat'][:, 1]),
                           1-2*(self.sim_data['base_quat'][:, 1]**2+self.sim_data['base_quat'][:, 2]**2)),
        )
        return np.concatenate((data['velocity_world']*2., data['position'][:, 1:3],
                               np.sin(data['yaw'])[:, None], np.cos(data['yaw'])[:, None],
                               data['projected_gravity']), axis=1).astype(np.float32)

    def set_gait_ids(self, ids):
        # The provided control_task ignores output[10]; preserve it only for diagnostics.
        self.gait_ids[:] = ids
