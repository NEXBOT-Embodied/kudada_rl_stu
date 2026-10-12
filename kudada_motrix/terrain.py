"""Seeded Kudada curriculum height fields using the native scene asset API.

The source atlas indexes [x, y]; native HField buffers index [y, x].
Geometry generation is a NumPy migration, not the unavailable Isaac Gym binary.
"""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.asset import MaterialCfg, ProceduralHFieldAssetCfg, TerrainGeneratorCfg
from motrix_env_core.config.scene.base import SceneAssetsCfg, SceneCfg, SceneObjsCfg, RobotCfg, SceneVisualCfg, SystemCameraCfg
from motrix_env_core.config.scene.geometry import HFieldTerrainCfg
from motrix_env_core.config.scene.light import LightCfg
from motrix_env_core.config.scene.mjcf import MjcfFileCfg

from kudada_motrix.constants import MODEL


@configclass
class KudadaTerrainCfg:
    num_rows: int = 10
    num_cols: int = 20
    terrain_length: float = 8.
    terrain_width: float = 8.
    horizontal_scale: float = .1
    vertical_scale: float = .005
    border_size: float = 25.
    terrain_proportions: tuple[float, ...] = (.4, .25, .15, .15, .05)
    slope_max: float = .18
    step_height_min: float = .005
    step_height_max: float = .01
    step_width: float = .2
    obstacle_height_min: float = .005
    obstacle_height_max: float = .01
    platform_size: float = 3.
    max_init_terrain_level: int = 3
    curriculum: bool = True
    curriculum_distance_ratio: float = .25
    spawn_xy_range: float = 1.

    def validate(self):
        for name in ("num_rows", "num_cols"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("terrain_length", "terrain_width", "horizontal_scale", "vertical_scale", "step_width"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("border_size", "slope_max", "step_height_min", "step_height_max",
                     "obstacle_height_min", "obstacle_height_max", "spawn_xy_range"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.platform_size < min(self.terrain_length, self.terrain_width):
            raise ValueError("platform_size must fit inside a tile")
        if not 0 <= self.max_init_terrain_level < self.num_rows:
            raise ValueError("max_init_terrain_level must be a valid terrain row")
        if not 0 < self.curriculum_distance_ratio <= 1:
            raise ValueError("curriculum_distance_ratio must be in (0, 1]")
        if len(self.terrain_proportions) != 5 or min(self.terrain_proportions) < 0 or not np.isclose(sum(self.terrain_proportions), 1):
            raise ValueError("terrain_proportions must contain five nonnegative fractions totaling one")
        if self.step_height_min > self.step_height_max or self.obstacle_height_min > self.obstacle_height_max:
            raise ValueError("terrain height ranges must be ordered")
        for value in (self.terrain_length, self.terrain_width, self.border_size, self.step_width):
            if not np.isclose(value/self.horizontal_scale, round(value/self.horizontal_scale)):
                raise ValueError("tile, border and step dimensions must align to horizontal_scale")


class KudadaTerrainMap:
    def __init__(self, cfg, seed):
        cfg.validate()
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        nx, ny = round(cfg.terrain_length/cfg.horizontal_scale), round(cfg.terrain_width/cfg.horizontal_scale)
        border = round(cfg.border_size/cfg.horizontal_scale)
        self.raw = np.zeros((cfg.num_rows*nx+2*border, cfg.num_cols*ny+2*border), dtype=np.int16)
        self.origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        self.column_types = []
        proportions = np.cumsum(cfg.terrain_proportions)
        for col in range(cfg.num_cols):
            choice = col/cfg.num_cols+.001
            kind = int(np.searchsorted(proportions, choice, side="right"))
            self.column_types.append(("smooth_slope", "rough_slope", "stairs_down", "stairs_up", "obstacles")[kind])
            for row in range(cfg.num_rows):
                tile = self._tile(cfg, rng, nx, ny, kind, choice, row/cfg.num_rows)
                x0, y0 = border+row*nx, border+col*ny
                self.raw[x0:x0+nx, y0:y0+ny] = tile
                # Match the source origin: maximum in the central 2 m square.
                rx, ry = round(1./cfg.horizontal_scale), round(1./cfg.horizontal_scale)
                z = float(np.max(tile[nx//2-rx:nx//2+rx, ny//2-ry:ny//2+ry]))*cfg.vertical_scale
                self.origins[row, col] = [(x0+nx/2)*cfg.horizontal_scale,
                                          (y0+ny/2)*cfg.horizontal_scale, z]
        self.size = ((self.raw.shape[0]-1)*cfg.horizontal_scale, (self.raw.shape[1]-1)*cfg.horizontal_scale)
        self.origin_xy = -np.array(self.size, dtype=np.float32)/2
        # Generated native geometry is centered at z=0. Translate the entire
        # atlas vertically instead of introducing a negative HField sample.
        self.z_offset = -float(self.raw.min())*cfg.vertical_scale
        self.heights = (self.raw.astype(np.float32)*cfg.vertical_scale+self.z_offset).astype(np.float32)
        self.height_scale = max(float(self.heights.max()), cfg.vertical_scale)
        self.origins[:, :, :2] += self.origin_xy
        self.origins[:, :, 2] += self.z_offset
        self.column_types = tuple(self.column_types)

    @staticmethod
    def _tile(c, rng, nx, ny, kind, choice, difficulty):
        x, y = np.arange(nx), np.arange(ny)
        if kind in (0, 1):
            slope = difficulty*c.slope_max
            if kind == 0 and choice < c.terrain_proportions[0]/2:
                slope *= -1
            peak = int(slope*c.horizontal_scale*nx/2/c.vertical_scale)
            rx = 1-np.abs(nx//2-x)/(nx//2)
            ry = 1-np.abs(ny//2-y)/(ny//2)
            tile = (peak*rx[:, None]*ry[None]).astype(np.int16)
            half = int(c.platform_size/c.horizontal_scale/2)
            level = tile[nx//2-half, ny//2-half]
            tile = np.clip(tile, min(0, level), max(0, level))
            if kind == 1:
                # Coarse 20 cm ternary noise, interpolated to 10 cm and quantized.
                coarse = rng.integers(-1, 2, (max(nx//2, 2), max(ny//2, 2)))
                xx, yy = np.linspace(0, coarse.shape[0]-1, nx), np.linspace(0, coarse.shape[1]-1, ny)
                ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
                jx, jy = np.minimum(ix+1, coarse.shape[0]-1), np.minimum(iy+1, coarse.shape[1]-1)
                fx, fy = (xx-ix)[:, None], (yy-iy)[None]
                noise = (coarse[ix[:, None], iy]*(1-fx)*(1-fy)+coarse[jx[:, None], iy]*fx*(1-fy)
                         +coarse[ix[:, None], jy]*(1-fx)*fy+coarse[jx[:, None], jy]*fx*fy)
                tile += np.rint(noise*.005/c.vertical_scale).astype(np.int16)
            return tile
        if kind in (2, 3):
            step = int((c.step_height_min+difficulty*(c.step_height_max-c.step_height_min))/c.vertical_scale)
            if kind == 2:
                step *= -1
            width = max(round(c.step_width/c.horizontal_scale), 1)
            edge = np.minimum(np.minimum(x, nx-1-x)[:, None], np.minimum(y, ny-1-y)[None])
            cap = int(np.ceil((min(nx, ny)-c.platform_size/c.horizontal_scale)/(2*width)))
            return (np.minimum(edge//width, cap)*step).astype(np.int16)
        units = int((c.obstacle_height_min+difficulty*(c.obstacle_height_max-c.obstacle_height_min))/c.vertical_scale)
        tile = np.zeros((nx, ny), dtype=np.int16)
        lo, hi = max(round(1/c.horizontal_scale), 1), max(round(2/c.horizontal_scale), 2)
        for _ in range(20):  # Static geometry generation, never an environment loop.
            sx, sy = rng.integers(lo, hi, 2)
            sx, sy = min(int(sx), nx-1), min(int(sy), ny-1)
            x0, y0 = rng.integers(0, nx-sx), rng.integers(0, ny-sy)
            tile[x0:x0+sx, y0:y0+sy] = rng.choice([-units, -(units//2), units//2, units])
        half = int(c.platform_size/c.horizontal_scale/2)
        tile[nx//2-half:nx//2+half, ny//2-half:ny//2+half] = 0
        return tile

    def sample_conservative(self, xy):
        """Legacy scan: minimum of three neighboring [x, y] grid cells."""
        index = ((xy-self.origin_xy)/self.cfg.horizontal_scale).astype(np.int64)
        ix = np.clip(index[..., 0], 0, self.raw.shape[0]-2)
        iy = np.clip(index[..., 1], 0, self.raw.shape[1]-2)
        return np.minimum(np.minimum(self.heights[ix, iy], self.heights[ix+1, iy]), self.heights[ix, iy+1])


@configclass
class KudadaTerrainGeneratorCfg(TerrainGeneratorCfg):
    terrain: KudadaTerrainCfg = KudadaTerrainCfg()

    def generate(self, size, shape):
        terrain = KudadaTerrainMap(self.terrain, self.seed)
        # MotrixSim's runtime buffer has rows along y and columns along x.
        return np.ascontiguousarray(terrain.heights.T/terrain.height_scale)


@configclass
class KudadaTerrainAssetsCfg(SceneAssetsCfg):
    kudada_heightfield: ProceduralHFieldAssetCfg
    kudada_ground_material: MaterialCfg = MaterialCfg(color=(.35, .45, .55, 1.), roughness=.8)


@configclass(kw_only=True)
class KudadaTerrainObjsCfg(SceneObjsCfg):
    ground: HFieldTerrainCfg
    sun: LightCfg = LightCfg(illuminance=5000.)


def terrain_scene(terrain, seed):
    generator = KudadaTerrainGeneratorCfg(terrain=terrain.cfg, seed=seed, height_scale=terrain.height_scale)
    return SceneCfg(
        assets=KudadaTerrainAssetsCfg(kudada_heightfield=ProceduralHFieldAssetCfg(
            generator=generator, size=terrain.size, shape=terrain.raw.T.shape)),
        objs=KudadaTerrainObjsCfg(
            robot=RobotCfg(model=MjcfFileCfg(file=str(MODEL)), base_link_name="base_link"),
            ground=HFieldTerrainCfg(hfield="kudada_heightfield", material="kudada_ground_material", friction=(.8, .2, .2))),
        visual=SceneVisualCfg(ambient_light_color=(.3, .3, .3), ambient_light_brightness=500.),
        system_camera=SystemCameraCfg(distance=1.25, elevation=-15., azimuth=125.),
    )
