"""Kudada flat task using MotrixLab's DirectEnv lifecycle and sim boundary."""

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim.model import ActuatorType
from motrix_env_core.sim.read import (
    BatchLinkNetContactForceQuery, BatchLinkPositionQuery, BatchLinkLinearVelocityQuery,
    JointPositionQuery, JointVelocityQuery,
    LinkAngularVelocityQuery, LinkLinearVelocityQuery, LinkPositionQuery, LinkQuaternionQuery,
)
from motrix_env_core.sim.write import (
    BodyAngularVelocityWrite, BodyLinearVelocityWrite, BodyPositionWrite, BodyRotationWrite,
    CtrlTargetsWrite, JointPositionWrite, JointVelocityWrite,
)

from kudada_motrix.cfg import KudadaRealFlatCfg
from kudada_motrix.constants import (
    CLIP, CONTACT_LINKS, CRITIC_SIZE, DEFAULT, FEET, FRAME_SIZE, HISTORY_FRAMES, JOINT_NAMES, KD, KP,
)
from kudada_motrix.contract import decode_actions, observations, rotate_inverse, termination_reasons
from kudada_motrix.rewards import EXTRA_REWARD_SCALES, extra_reward_terms

REWARD_SCALES = {
    "tracking_lin_vel": 1., "tracking_ang_vel": .3, "orientation": -2.,
    "base_height": -3., "lin_vel_z": -.2, "ang_vel_xy": -.15,
    "torques": -.0005, "action_smoothness": -.01, "gait_tracking": 1.,
    "no_fly": -.3, "collision": -1., "stand_still": -.02,
}


@registry.env("kudada-real-flat")
class KudadaRealFlatEnv(DirectEnv[KudadaRealFlatCfg]):
    frame_size = FRAME_SIZE
    history_frames = HISTORY_FRAMES
    critic_size = CRITIC_SIZE

    def __init__(self, cfg: KudadaRealFlatCfg, num_envs=1, backend=None):
        cfg.validate()
        super().__init__(cfg, num_envs, backend=backend)
        self.rng = np.random.default_rng(cfg.seed)
        if (cfg.randomization.physical_enabled or cfg.randomization.disturbance or cfg.randomization.push_robots) and not hasattr(self.sim, "apply_base_force"):
            raise ValueError("Physical randomization requires sim='kudada-motrixsim' or the flat-training preset")
        self.model = self.sim.compile_model({})
        actuators = {a.name: a for a in self.model.actuators}
        selected = [actuators[n] for n in JOINT_NAMES]
        if self.num_actuators != 10 or any(a.target_name != n or a.actuator_type != ActuatorType.MOTOR
                                          for a, n in zip(selected, JOINT_NAMES)):
            raise ValueError("Expected ten named torque motors matching the frozen joint order")
        self.torque_limits = np.array([a.ctrl_range for a in selected], dtype=np.float32)
        self.joint_data = self.sim.compile_reads({
            "q": JointPositionQuery(joints=JOINT_NAMES), "qvel": JointVelocityQuery(joints=JOINT_NAMES),
        })
        self.sim_data = self.sim.compile_reads({
            "q": JointPositionQuery(joints=JOINT_NAMES), "qvel": JointVelocityQuery(joints=JOINT_NAMES),
            "base_pos": LinkPositionQuery(link="base_link"),
            "base_quat": LinkQuaternionQuery(link="base_link"),
            "world_lin_vel": LinkLinearVelocityQuery(link="base_link"),
            "world_ang_vel": LinkAngularVelocityQuery(link="base_link"),
            "contacts": BatchLinkNetContactForceQuery(links=CONTACT_LINKS),
            "feet_pos": BatchLinkPositionQuery(links=FEET),
            "feet_vel": BatchLinkLinearVelocityQuery(links=FEET),
        })
        self.control_write = self.sim.compile_writes({"torque": CtrlTargetsWrite(JOINT_NAMES)}, forward_kinematics=False)
        self.reset_write = self.sim.compile_writes({
            "q": JointPositionWrite(JOINT_NAMES), "qvel": JointVelocityWrite(JOINT_NAMES),
            "pos": BodyPositionWrite(("base_link",)), "quat": BodyRotationWrite(("base_link",)),
            "lin_vel": BodyLinearVelocityWrite(("base_link",)),
            "ang_vel": BodyAngularVelocityWrite(("base_link",)),
        }, reset=True)
        self.push_write = self.sim.compile_writes({"lin_vel": BodyLinearVelocityWrite(("base_link",))},
                                                  forward_kinematics=True)
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-CLIP, CLIP, (self.frame_size * self.history_frames,), dtype=np.float32),
            value=gym.spaces.Box(-CLIP, CLIP, (self.critic_size,), dtype=np.float32),
        )
        self._action_space = gym.spaces.Box(-CLIP, CLIP, (10,), dtype=np.float32)
        self.actions = np.zeros((num_envs, 10), dtype=np.float32)
        self.last_actions = np.zeros_like(self.actions)
        self.last_last_actions = np.zeros_like(self.actions)
        self.target_positions = np.broadcast_to(DEFAULT, self.actions.shape).copy()
        self.history = np.zeros((num_envs, self.history_frames, self.frame_size), dtype=np.float32)
        self._fresh_history = np.ones(num_envs, dtype=bool)
        self.commands = np.zeros((num_envs, 3), dtype=np.float32)
        self.heading_targets = np.zeros(num_envs, dtype=np.float32)
        self.command_samples = np.zeros(num_envs, dtype=np.int64)
        self.command_interval = round(cfg.command_resampling_seconds / cfg.ctrl_dt)
        self.terrain_heights = np.zeros((num_envs, 187), dtype=np.float32)
        self.disturbance = np.zeros((num_envs, 3), dtype=np.float32)
        self._gravity_world = np.broadcast_to(np.array([0, 0, -1], dtype=np.float32), (num_envs, 3))
        self.torques = self.control_write.buffer("torque")
        self.physics_steps = 0
        self.common_step_counter = 0
        self.disturbance_events = 0
        self.push_events = 0
        self.last_dof_vel = np.zeros_like(self.actions)
        self.prev_world_lin_vel = np.zeros((num_envs, 3), dtype=np.float32)
        self.base_lin_acc = np.zeros_like(self.prev_world_lin_vel)
        self.feet_air_time = np.zeros((num_envs, 2), dtype=np.float32)
        self.last_contacts = np.zeros((num_envs, 2), dtype=bool)
        self.kp_factors = np.ones((num_envs, 1), dtype=np.float32)
        self.kd_factors = np.ones_like(self.kp_factors)
        self.motor_strength = np.ones_like(self.kp_factors)
        self.motor_zero_offsets = np.zeros_like(self.actions)
        self.init_joint_offsets = np.zeros_like(self.actions)
        self.initial_joint_factors = np.ones_like(self.actions)
        self.delay_substeps = np.zeros(num_envs, dtype=np.int64)
        self.noise_scale = np.zeros(self.frame_size, dtype=np.float32)
        self.noise_scale[8:11] = .35*.25
        self.noise_scale[11:14] = .07
        self.noise_scale[14:24] = .01
        self.noise_scale[24:34] = 1.5*.05
        self.noise_scale *= cfg.noise_level
        self.actor_noise = np.zeros((num_envs, self.frame_size), dtype=np.float32)
        self.reward_scales = {**REWARD_SCALES, **(EXTRA_REWARD_SCALES if cfg.full_rewards else {})}
        if cfg.randomization.physical_enabled:
            if self.sim.nominal_mass[self.sim.base_index]+cfg.randomization.payload_mass_range[0] <= 0:
                raise ValueError("payload_mass_range can produce a nonpositive base mass")
            self.physical = {"mass": np.broadcast_to(self.sim.nominal_mass, (num_envs, len(self.sim.link_names))).copy(),
                             "com": np.broadcast_to(self.sim.nominal_com, (num_envs, *self.sim.nominal_com.shape)).copy(),
                             "inertia": np.broadcast_to(self.sim.nominal_inertia, (num_envs, *self.sim.nominal_inertia.shape)).copy(),
                             "friction": np.broadcast_to(self.sim.nominal_friction, (num_envs, *self.sim.nominal_friction.shape)).copy()}
        self.terminal_env_ids = np.empty(0, dtype=np.int64)
        self.terminal_policy_obs = np.empty((0, self.frame_size * self.history_frames), dtype=np.float32)
        self.terminal_value_obs = np.empty((0, self.critic_size), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _derive_state(self):
        q = self.sim_data["base_quat"]
        self.base_lin_vel = rotate_inverse(q, self.sim_data["world_lin_vel"])
        self.base_ang_vel = rotate_inverse(q, self.sim_data["world_ang_vel"])
        self.projected_gravity = rotate_inverse(q, self._gravity_world)

    def _update_heading_commands(self, env_ids=None):
        if self.cfg.fixed_command is not None or not self.cfg.heading_command:
            return
        quat = self.sim_data["base_quat"] if env_ids is None else self.sim_data["base_quat"][env_ids]
        x, y, z, w = quat.T
        heading = np.arctan2(2*(x*y+w*z), 1-2*(y*y+z*z))
        target = self.heading_targets if env_ids is None else self.heading_targets[env_ids]
        error = (target-heading) % (2*np.pi)
        error = np.where(error > np.pi, error-2*np.pi, error)
        if env_ids is None:
            self.commands[:, 2] = np.clip(.5*error, -2., 2.)
        else:
            self.commands[env_ids, 2] = np.clip(.5*error, -2., 2.)

    def _sample_commands(self, env_ids):
        if env_ids.size == 0:
            return
        if self.cfg.fixed_command is not None:
            self.commands[env_ids] = self.cfg.fixed_command
        else:
            rows = len(env_ids)
            self.commands[env_ids, 0] = self.rng.uniform(*self._command_x_bounds(), rows)
            self.commands[env_ids, 1] = self.rng.uniform(-.15, .15, rows)
            if self.cfg.heading_command:
                self.heading_targets[env_ids] = self.rng.uniform(-3.14, 3.14, rows)
            else:
                self.commands[env_ids, 2] = self.rng.uniform(-.3, .3, rows)
            small = np.linalg.norm(self.commands[env_ids, :2], axis=1) <= .1
            self.commands[env_ids[small], :2] = 0.
            self._update_heading_commands(env_ids)
        self.command_samples[env_ids] += 1

    def _command_x_bounds(self):
        return (-.3, .3)

    def _height_for_reward(self):
        return self.sim_data["base_pos"][:, 2]

    def _height_for_termination(self):
        return self.sim_data["base_pos"][:, 2]

    def _reset_positions(self, env_ids):
        return np.broadcast_to([0., 0., self.cfg.initial_height], (len(env_ids), 3))

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState):
        if actions.shape != self.actions.shape or not np.isfinite(actions).all():
            raise ValueError(f"actions must be finite and have shape {self.actions.shape}")
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.actions[:], self.target_positions[:] = decode_actions(actions)
        rand = self.cfg.randomization
        self.delay_substeps[:] = self.rng.integers(rand.delay_substeps_range[0], rand.delay_substeps_range[1]+1,
                                                  self.num_envs) if rand.delay else 0
        return state

    def _sample_noise(self, env_ids):
        if self.cfg.add_noise:
            self.actor_noise[env_ids] = self.rng.uniform(-1., 1., (len(env_ids), self.frame_size))*self.noise_scale
        else:
            self.actor_noise[env_ids] = 0.

    def _randomize_episode(self, env_ids):
        rand = self.cfg.randomization
        rows = len(env_ids)
        for flag, bounds, buffer, nominal in (
            (rand.randomize_kp, rand.kp_range, self.kp_factors, 1.),
            (rand.randomize_kd, rand.kd_range, self.kd_factors, 1.),
            (rand.randomize_motor_strength, rand.motor_strength_range, self.motor_strength, 1.),
            (rand.randomize_motor_zero_offset, rand.motor_zero_offset_range, self.motor_zero_offsets, 0.),
            (rand.randomize_init_joint_offset, rand.init_joint_offset_range, self.init_joint_offsets, 0.),
            (rand.randomize_initial_joint_pos, rand.initial_joint_pos_range, self.initial_joint_factors, 1.),
        ):
            buffer[env_ids] = self.rng.uniform(*bounds, (rows, buffer.shape[1])) if flag else nominal
        if not rand.physical_enabled:
            if hasattr(self, "physical"):
                for name, buffer in self.physical.items():
                    buffer[env_ids] = getattr(self.sim, "nominal_"+name)
            return
        sim, p = self.sim, self.physical
        mass = np.broadcast_to(sim.nominal_mass, (rows, len(sim.link_names))).copy()
        base = sim.base_index
        if rand.randomize_payload_mass:
            mass[:, base] += self.rng.uniform(*rand.payload_mass_range, rows)
        if rand.randomize_link_mass:
            links = np.arange(len(sim.link_names)) != base
            mass[:, links] *= self.rng.uniform(*rand.link_mass_range, (rows, int(links.sum())))
        p["mass"][env_ids] = mass
        com = np.broadcast_to(sim.nominal_com, (rows, *sim.nominal_com.shape)).copy()
        if rand.randomize_com_displacement:
            com[:, base] += self.rng.uniform(*rand.com_displacement_range, (rows, 3))
        p["com"][env_ids] = com
        inertia_factor = mass/sim.nominal_mass
        if rand.randomize_inertia:
            inertia_factor *= self.rng.uniform(*rand.inertia_range, inertia_factor.shape)
        p["inertia"][env_ids] = sim.nominal_inertia*inertia_factor[:, :, None]
        friction = np.broadcast_to(sim.nominal_friction, (rows, *sim.nominal_friction.shape)).copy()
        if rand.randomize_friction:
            # Randomize sliding coefficient; preserve nominal spin/roll coefficients.
            friction[:, :, 0] = self.rng.uniform(*rand.friction_range, (rows, 1))
        p["friction"][env_ids] = friction
        # Reset restores native overrides, so physical writes occur after reset_write.

    def _scheduled_events(self):
        rand = self.cfg.randomization
        self.common_step_counter += 1
        self.disturbance[:] = 0.
        if rand.disturbance and self.common_step_counter % rand.disturbance_interval_steps == 0:
            self.disturbance[:] = self.rng.uniform(*rand.disturbance_range, self.disturbance.shape)
            self.disturbance_events += 1
        if rand.push_robots and self.common_step_counter % round(rand.push_interval_seconds/self.cfg.ctrl_dt) == 0:
            velocity = self.sim_data["world_lin_vel"].copy()
            velocity[:, :2] = self.rng.uniform(-rand.max_push_vel_xy, rand.max_push_vel_xy, (self.num_envs, 2))
            self.push_write.buffer("lin_vel")[:] = velocity[:, None]
            self.push_write.execute()
            # The pinned neutral compiler skips FK for velocity-only writes.
            self.sim.refresh_kinematics()
            self.push_events += 1

    def physics_step(self):
        self._scheduled_events()
        # The target is held for 20 ms; PD is recomputed every 1 ms.
        for substep in range(self.cfg.sim_substeps):
            self.joint_data.execute()
            action = np.where((substep < self.delay_substeps)[:, None], self.last_actions, self.actions)
            target = DEFAULT+.25*action+self.motor_zero_offsets
            self.torques[:] = np.clip(
                (KP*self.cfg.kp_scale*self.kp_factors*(target-self.joint_data["q"])
                 -KD*self.kd_factors*self.joint_data["qvel"])*self.motor_strength,
                self.torque_limits[:, 0], self.torque_limits[:, 1],
            )
            self.control_write.execute()
            # One 1 ms pulse per event, applied at the base COM in its local frame.
            if substep == 0 and self.cfg.randomization.disturbance and np.any(self.disturbance):
                self.sim.apply_base_force(self.disturbance)
            self.sim.step(1)
        self.physics_steps += self.cfg.sim_substeps

    def _frame_observations(self, phase):
        return observations(
            q=self.sim_data["q"], qvel=self.sim_data["qvel"], commands=self.commands,
            phase=phase, actions=self.actions, base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel, projected_gravity=self.projected_gravity,
            base_height=self.sim_data["base_pos"][:, 2],
            foot_forces=self.sim_data["contacts"][:, 1:3, :3],
            terrain_heights=self.terrain_heights, disturbance=self.disturbance,
            actor_noise=self.actor_noise if self.cfg.add_noise else None,
        )

    def _reward_terms(self, phase):
        contacts = self.sim_data["contacts"][:, 1:3, 2] > 5.
        expected = np.stack((phase < .6, (phase+.5) % 1. < .6), axis=1)
        match = np.mean(contacts == expected, axis=1)
        moving = np.linalg.norm(self.commands[:, :2], axis=1) > .1
        raw = {
            "tracking_lin_vel": np.exp(-np.sum((self.commands[:, :2]-self.base_lin_vel[:, :2])**2, axis=1)/2.5),
            "tracking_ang_vel": np.exp(-(self.commands[:, 2]-self.base_ang_vel[:, 2])**2/2.5),
            "orientation": np.sum(self.projected_gravity[:, :2]**2, axis=1),
            "base_height": (self._height_for_reward()-.27)**2,
            "lin_vel_z": np.minimum(self.base_lin_vel[:, 2]**2, .1),
            "ang_vel_xy": np.sum(self.base_ang_vel[:, :2]**2, axis=1),
            "torques": np.sum(self.torques**2, axis=1),
            "action_smoothness": (np.sum((self.last_actions-self.actions)**2, axis=1)
                                  + np.sum((self.actions+self.last_last_actions-2*self.last_actions)**2, axis=1)
                                  + .05*np.sum(np.abs(self.actions), axis=1)),
            "gait_tracking": match*moving,
            "no_fly": ~np.any(contacts, axis=1),
            "collision": np.sum(np.linalg.norm(self.sim_data["contacts"][:, 3:, :3], axis=2) > .1, axis=1),
            "stand_still": np.sum(np.abs(self.sim_data["q"]-DEFAULT), axis=1)*(np.linalg.norm(self.commands[:, :2], axis=1) < .1),
        }
        if self.cfg.full_rewards:
            raw.update(extra_reward_terms(self, phase))
        return {name: np.asarray(value*self.reward_scales[name]*self.cfg.ctrl_dt, dtype=np.float32)
                for name, value in raw.items()}

    def compute_transition(self, state: ArrayEnvState):
        self.sim_data.execute()
        self._derive_state()
        self.base_lin_acc[:] = rotate_inverse(self.sim_data["base_quat"],
                                             (self.sim_data["world_lin_vel"]-self.prev_world_lin_vel)/self.cfg.ctrl_dt)
        next_steps = state.episode_steps+1
        self._sample_commands(np.flatnonzero(next_steps % self.command_interval == 0))
        self._update_heading_commands()
        phase = ((next_steps*self.cfg.ctrl_dt*2.) % 1.).astype(np.float32)
        reasons = termination_reasons(self._height_for_termination(), self.projected_gravity,
                                      self.sim_data["contacts"][:, 0, :3])
        state.terminated = np.logical_or.reduce(list(reasons.values()))
        state.reward_terms = self._reward_terms(phase)
        for name, scale in self.cfg.reward_multipliers.items():
            if name not in state.reward_terms:
                raise ValueError(f"Unknown reward term: {name}")
            state.reward_terms[name] *= scale
        self.last_dof_vel[:] = self.sim_data["qvel"]
        self.prev_world_lin_vel[:] = self.sim_data["world_lin_vel"]
        self._sample_noise(np.arange(self.num_envs))
        state.reward = sum(state.reward_terms.values())
        if self.cfg.only_positive_rewards:
            state.reward = np.maximum(state.reward, 0.)
        state.metrics = {
            "base_height": self.sim_data["base_pos"][:, 2].copy(),
            "tilt_xy": np.linalg.norm(self.projected_gravity[:, :2], axis=1),
            "foot_contact_count": np.sum(self.sim_data["contacts"][:, 1:3, 2] > 5., axis=1),
            "foot_force_z_max": np.max(self.sim_data["contacts"][:, 1:3, 2], axis=1).copy(),
            "command_speed": np.linalg.norm(self.commands, axis=1),
            "disturbance_norm": np.linalg.norm(self.disturbance, axis=1),
            **{f"termination_{k}": v.copy() for k, v in reasons.items()},
        }
        # Capture pre-reset terminal observations for the later HIM adapter.
        done = state.terminated | (next_steps >= self.cfg.max_episode_steps)
        self.terminal_env_ids = np.flatnonzero(done)
        if self.terminal_env_ids.size:
            actor, critic = self._frame_observations(phase)
            ids = self.terminal_env_ids
            self.terminal_policy_obs = np.concatenate((actor[ids, None], self.history[ids, :-1]), axis=1).reshape(len(ids), -1)
            self.terminal_value_obs = critic[ids].copy()
        else:
            self.terminal_policy_obs = np.empty((0, self.frame_size*self.history_frames), dtype=np.float32)
            self.terminal_value_obs = np.empty((0, self.critic_size), dtype=np.float32)
        return state

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self._randomize_episode(env_ids)
        self.reset_write.buffer("q")[env_ids] = DEFAULT*self.initial_joint_factors[env_ids]+self.init_joint_offsets[env_ids]
        self.reset_write.buffer("qvel")[env_ids] = 0.
        self.reset_write.buffer("pos")[env_ids] = self._reset_positions(env_ids)[:, None]
        self.reset_write.buffer("quat")[env_ids] = [0., 0., 0., 1.]
        rand = self.cfg.randomization
        velocity = self.rng.uniform(*rand.initial_velocity_range, (len(env_ids), 6)) if rand.randomize_initial_velocity else np.zeros((len(env_ids), 6))
        self.reset_write.buffer("lin_vel")[env_ids] = velocity[:, None, :3]
        self.reset_write.buffer("ang_vel")[env_ids] = velocity[:, None, 3:]
        self.reset_write.execute(env_ids)
        if rand.physical_enabled:
            self.sim.set_physical_properties(env_ids, **self.physical)
        for buffer in (self.actions, self.last_actions, self.last_last_actions, self.disturbance, self.torques):
            buffer[env_ids] = 0.
        self.target_positions[env_ids] = DEFAULT
        self.history[env_ids] = 0.
        self._fresh_history[env_ids] = True
        self.sim_data.execute(env_ids)
        self._derive_state()
        self.last_dof_vel[env_ids] = self.sim_data["qvel"][env_ids]
        self.prev_world_lin_vel[env_ids] = self.sim_data["world_lin_vel"][env_ids]
        self.base_lin_acc[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.delay_substeps[env_ids] = 0
        self._sample_noise(env_ids)
        self._sample_commands(env_ids)

    def compute_observation(self, state: ArrayEnvState):
        phase = ((state.episode_steps*self.cfg.ctrl_dt*2.) % 1.).astype(np.float32)
        actor, critic = self._frame_observations(phase)
        self.history[:, 1:] = self.history[:, :-1].copy()
        self.history[:, 0] = actor
        self.history[self._fresh_history] = actor[self._fresh_history, None]
        self._fresh_history[:] = False
        return state.replace(obs=NpObs(policy=self.history.reshape(self.num_envs, -1).copy(), value=critic))


@registry.env("kudada-real-flat-training")
class KudadaRealTrainingEnv(KudadaRealFlatEnv):
    """Training preset with a project backend that provides physical overrides."""

    def __init__(self, cfg, num_envs=1, backend=None):
        from kudada_motrix import backend as _backend  # noqa: F401 backend registration

        # The pinned registry injects its default "motrixsim" even when sim=None.
        # This task extends that backend for unsupported neutral write operations.
        if backend in (None, "motrixsim"):
            backend = "kudada-motrixsim"
        if backend != "kudada-motrixsim":
            raise ValueError("The Kudada training preset requires sim='kudada-motrixsim'")
        super().__init__(cfg, num_envs, backend)
