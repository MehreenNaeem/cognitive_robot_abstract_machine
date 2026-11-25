from __future__ import annotations

import mujoco
import numpy as np
import pytest

from llm_planner.read_usd import (
    APARTMENT_USD_PATH,
    COLLISION_PROXY_GEOM_GROUP,
    UsdToMujocoConverter,
    WorldSpaceMesh,
    read_apartment,
)


@pytest.fixture(scope="module")
def apartment_model() -> mujoco.MjModel:
    return read_apartment()


class TestWorldSpaceMeshFlatness:
    def test_coplanar_mesh_is_flat(self):
        mesh = WorldSpaceMesh(
            name="flat",
            vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
        )
        assert mesh.is_flat

    def test_volumetric_mesh_is_not_flat(self):
        mesh = WorldSpaceMesh(
            name="tetrahedron",
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
            ),
            faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
        )
        assert not mesh.is_flat


class TestApartmentConversion:
    def test_all_meshes_become_geoms(self, apartment_model: mujoco.MjModel):
        assert apartment_model.nmesh > 0
        assert apartment_model.ngeom == apartment_model.nmesh
        assert apartment_model.nbody > 1

    def test_model_can_step(self, apartment_model: mujoco.MjModel):
        data = mujoco.MjData(apartment_model)
        mujoco.mj_step(apartment_model, data)

    def test_materials_and_textures_are_converted(
        self, apartment_model: mujoco.MjModel
    ):
        assert apartment_model.nmat > 0
        assert apartment_model.ntex > 0
        material_ids = apartment_model.geom_matid
        assert np.any(material_ids >= 0), "Some geoms should carry a material"

    def test_meshes_with_face_subsets_are_split_per_material(self):
        converter = UsdToMujocoConverter(usd_path=APARTMENT_USD_PATH)
        converter.convert()
        window_meshes = [
            mesh for mesh in converter.meshes if "SM_Window_01_Section" in mesh.name
        ]
        assert len(window_meshes) >= 3
        appearance_names = {mesh.appearance.name for mesh in window_meshes}
        assert len(appearance_names) >= 2

    def test_textured_meshes_carry_texture_coordinates(self):
        converter = UsdToMujocoConverter(usd_path=APARTMENT_USD_PATH)
        converter.convert()
        floor = next(mesh for mesh in converter.meshes if "SM_Floor" in mesh.name)
        assert floor.appearance is not None
        assert floor.appearance.texture_path is not None
        assert floor.appearance.texture_path.exists()
        assert floor.texture_coordinates is not None
        assert floor.texture_coordinate_faces.shape == floor.faces.shape

    def test_collision_proxies_are_invisible(self, apartment_model: mujoco.MjModel):
        proxy_geom_ids = [
            geom_id
            for geom_id in range(apartment_model.ngeom)
            if "UCX_" in mujoco.mj_id2name(apartment_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        ]
        assert len(proxy_geom_ids) > 0
        for geom_id in proxy_geom_ids:
            assert apartment_model.geom_group[geom_id] == COLLISION_PROXY_GEOM_GROUP
            assert apartment_model.geom_rgba[geom_id][3] == 0.0

    def test_every_scene_joint_gets_an_actuated_mujoco_joint(
        self, apartment_model: mujoco.MjModel
    ):
        assert apartment_model.njnt > 0
        assert apartment_model.nu == apartment_model.njnt

    def test_hinge_ranges_are_in_radians(self, apartment_model: mujoco.MjModel):
        hinge_joint_ids = [
            joint_id
            for joint_id in range(apartment_model.njnt)
            if apartment_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        assert len(hinge_joint_ids) > 0
        for joint_id in hinge_joint_ids:
            assert apartment_model.jnt_range[joint_id][1] <= np.pi + 1e-6

    def test_commanded_joints_reach_their_upper_limits(
        self, apartment_model: mujoco.MjModel
    ):
        data = mujoco.MjData(apartment_model)
        data.ctrl[:] = apartment_model.actuator_ctrlrange[:, 1]
        for _ in range(3000):
            mujoco.mj_step(apartment_model, data)
        for joint_id in range(apartment_model.njnt):
            position = data.qpos[apartment_model.jnt_qposadr[joint_id]]
            upper_limit = apartment_model.jnt_range[joint_id][1]
            tolerance = 0.05 * max(abs(upper_limit), 0.1)
            assert abs(position - upper_limit) < tolerance

    def test_geometry_is_expressed_in_world_coordinates(self):
        converter = UsdToMujocoConverter(usd_path=APARTMENT_USD_PATH)
        converter.convert()
        all_vertices = np.vstack([mesh.vertices for mesh in converter.meshes])
        extents = all_vertices.max(axis=0) - all_vertices.min(axis=0)
        assert np.all(extents > 1.0), "Apartment should span more than a meter per axis"