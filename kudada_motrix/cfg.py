"""The flat, deterministic first step of the kudada_real migration."""

import math

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SystemCameraCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.direct.env import DirectEnvCfg

from kudada_motrix.constants import MODEL
from kudada_motrix.randomization import DomainRandCfg, training_randomization
from kudada_motrix.terrain import KudadaTerrainCfg


@registry.envcfg("kudada-real-flat")
@configclass
class KudadaRealFlatCfg(DirectEnvCfg):
    """Kudada real observation contract on flat ground, with original PD gains.

    zh_CN: Kudada 实机观测布局的平地最小环境。
    """

    scene: SceneCfg = SceneCfg(
        file=str(MODEL),
        system_camera=SystemCameraCfg(lookat=(0., 0., .28), distance=1.25, elevation=-15., azimuth=125.),
    )
    sim: SimCfg = SimCfg(dt=.001)
    ctrl_dt: float = .02
    max_episode_seconds: float = 20.
    render_spacing: float = 1.
    seed: int = 42
    initial_height: float = .30
    kp_scale: float = 1.  # 4 is a separate diagnostic setting, not a training default.
    command_resampling_seconds: float = 20.
    heading_command: bool = True
    fixed_command: tuple[float, float, float] | None = None
    only_positive_rewards: bool = True
    reward_multipliers: dict[str, float] = {}
    full_rewards: bool = False
    add_noise: bool = False
    noise_level: float = 1.
    randomization: DomainRandCfg = DomainRandCfg()

    def validate(self):
        super().validate()
        self.randomization.validate()
        if not all(math.isfinite(v) for v in self.reward_multipliers.values()):
            raise ValueError("Reward multipliers must be finite")
        if not math.isfinite(self.noise_level) or self.noise_level < 0:
            raise ValueError("noise_level must be finite and nonnegative")
        if not math.isclose(self.sim.dt, .001) or not math.isclose(self.ctrl_dt, .02):
            raise ValueError("The frozen control contract requires dt=.001 and ctrl_dt=.02")
        if not math.isfinite(self.kp_scale) or self.kp_scale <= 0:
            raise ValueError("kp_scale must be finite and positive")
        if not math.isfinite(self.initial_height) or self.initial_height <= 0:
            raise ValueError("initial_height must be finite and positive")
        if not math.isfinite(self.command_resampling_seconds) or self.command_resampling_seconds < self.ctrl_dt:
            raise ValueError("command_resampling_seconds must be at least one control step")
        if self.max_episode_seconds is None or not math.isfinite(self.max_episode_seconds) or self.max_episode_seconds < self.ctrl_dt:
            raise ValueError("max_episode_seconds must be at least one control step")
        if self.fixed_command is not None:
            if len(self.fixed_command) != 3 or not all(math.isfinite(v) for v in self.fixed_command):
                raise ValueError("fixed_command must contain three finite values")


@registry.envcfg("kudada-real-flat-training")
@configclass
class KudadaRealTrainingCfg(KudadaRealFlatCfg):
    """All 25 legacy rewards, observation noise and supported randomization."""

    full_rewards: bool = True
    add_noise: bool = True
    randomization: DomainRandCfg = training_randomization()


@registry.envcfg("kudada-terrain")
@configclass
class KudadaCfg(KudadaRealTrainingCfg):
    """Ordinary Kudada: 40 x 10 actor history, complete 239-value critic."""

    terrain: KudadaTerrainCfg = KudadaTerrainCfg()
    command_curriculum: bool = True
    max_command_curriculum: float = .8
    initial_command_x_range: tuple[float, float] = (-.3, .3)

    def validate(self):
        super().validate()
        self.terrain.validate()
        if not math.isfinite(self.max_command_curriculum) or self.max_command_curriculum <= 0:
            raise ValueError("max_command_curriculum must be finite and positive")
        lo, hi = self.initial_command_x_range
        if not all(math.isfinite(v) for v in (lo, hi)) or not lo <= 0 <= hi or lo == hi:
            raise ValueError("initial_command_x_range must straddle zero with ordered finite bounds")
