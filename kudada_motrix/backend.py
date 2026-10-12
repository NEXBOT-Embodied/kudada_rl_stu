"""Project backend extensions for native Kudada mass, inertia and force events.

Native handles stay here; the environment exchanges NumPy arrays only.
"""

import numpy as np

from motrix_env_core.sim.registry import register_sim_backend
from motrix_env_motrixsim.runtime import MotrixSimBackend


class KudadaMotrixBackend(MotrixSimBackend):
    name = "kudada-motrixsim"

    def __init__(self, scene, sim, num_envs):
        super().__init__(scene, sim, num_envs)
        self._links = tuple(self._model.get_body("base_link").links)
        self.link_names = tuple(link.name for link in self._links)
        self.base_index = self.link_names.index("base_link")
        self._base = self._links[self.base_index]
        # Original MJCF collision geoms are unnamed. Use native handles here.
        self._collision_geoms = tuple(g for g in self._model.geoms if not g.visual_only
                                      and (g.collision_group or g.collision_affinity))
        self.collision_geom_indices = tuple(g.index for g in self._collision_geoms)
        self.nominal_mass = np.array([link.mass for link in self._links], dtype=np.float32)
        self.nominal_com = np.array([link.center_of_mass for link in self._links], dtype=np.float32)
        self.nominal_inertia = np.stack([link.get_inertia_override(self._data)[0] for link in self._links])
        self.nominal_friction = np.array([g.friction for g in self._collision_geoms], dtype=np.float32)

    def set_physical_properties(self, env_ids, *, mass, com, inertia, friction):
        # Data slicing is a native shared view. Only selected instances are written.
        env_ids = np.sort(env_ids)
        mask = np.zeros(self._num_envs, dtype=bool)
        mask[env_ids] = True
        data = self._data[mask]
        for i, link in enumerate(self._links):
            link.set_mass_override(data, np.ascontiguousarray(mass[env_ids, i]))
            link.set_center_of_mass_override(data, np.ascontiguousarray(com[env_ids, i]))
            link.set_inertia_override(data, np.ascontiguousarray(inertia[env_ids, i]))
        for i, geom in enumerate(self._collision_geoms):
            geom.set_friction_override(data, np.ascontiguousarray(friction[env_ids, i]))
        self._model.forward_kinematic(data)

    def physical_properties(self):
        """Effective native values, for verification and experiment snapshots."""
        return {
            "mass": np.stack([link.get_mass_override(self._data) for link in self._links], axis=1),
            "com": np.stack([link.get_center_of_mass_override(self._data) for link in self._links], axis=1),
            "inertia": np.stack([link.get_inertia_override(self._data) for link in self._links], axis=1),
            "friction": np.stack([g.get_friction_override(self._data) for g in self._collision_geoms], axis=1),
        }

    def apply_base_force(self, force):
        """Apply a body-local force at the base COM for the next physics step."""
        self._base.add_external_force(self._data, np.ascontiguousarray(force, dtype=np.float32), local=True)

    def refresh_kinematics(self):
        """Refresh link velocity channels after a velocity-only write."""
        self._model.forward_kinematic(self._data)


register_sim_backend("kudada-motrixsim", lambda: KudadaMotrixBackend)
