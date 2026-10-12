"""Native camera following, kept at the project simulator boundary."""
from motrix_env_motrixsim.renderer import MotrixSimRenderer


class FollowRenderer(MotrixSimRenderer):
    def __init__(self, backend, config, *, num_envs, render_spacing, system_camera):
        super().__init__(backend._model, lambda: backend._data, config,
                         num_envs=num_envs, render_spacing=render_spacing, system_camera=system_camera)
        self.distance = config.camera_distance or system_camera.distance
        self.azimuth = config.camera_azimuth if config.camera_azimuth is not None else system_camera.azimuth
        self.elevation = config.camera_elevation if config.camera_elevation is not None else system_camera.elevation

    def follow(self, position):
        """Accept a NumPy world position; alter only the view, never physics."""
        self._render.system_camera.set_view([float(position[0]), float(position[1]), .25],
                                            self.distance, self.elevation, self.azimuth)
