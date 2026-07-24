from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import imageio
import mujoco
import numpy as np


class UngroundedTaskError(Exception):
    """Raised when an :func:`interpret_task` result could not be grounded to any joint."""

from physics_simulators.mujoco_simulator import MujocoSimulator


class NoFramesCapturedError(Exception):
    """Raised when a :class:`MotionRecorder` is asked to save a video before any frame was captured."""
def minimum_jerk(value):
    """
    Smooth fifth-order interpolation.

    Position, velocity and acceleration are smooth at both
    the beginning and end of the movement.
    """
    value = float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )

    return (
        10.0 * value**3
        - 15.0 * value**4
        + 6.0 * value**5
    )

@dataclass
class MotionRecorder:
    """
    Captures MuJoCo simulation frames while a joint is driven toward a target
    value, and saves them as an .mp4 video.

    :param simulator: The running MuJoCo simulator to capture frames from.
    :param width: Frame width in pixels.
    :param height: Frame height in pixels.
    :param fps: Frames per second of the output video.
    """

    simulator: MujocoSimulator
    width: int = 640
    height: int = 480
    fps: int = 60
    camera_lookat: Optional[np.ndarray] = None
    """World point [x, y, z] the camera looks at. Falls back to the model's bounding-box center if not given."""
    camera_distance: Optional[float] = None
    """Distance from `camera_lookat` to the camera. Falls back to a value that fits the whole model if not given."""
    camera_azimuth: Optional[float] = None
    """Horizontal camera angle in degrees. Falls back to the MuJoCo default (90) if not given."""
    camera_elevation: Optional[float] = None
    """Vertical camera angle in degrees, negative looks downward. Falls back to the MuJoCo default (-45) if not given."""
    hide_collision_geoms: bool = True
    """Whether to hide group-0 geoms (the collision primitives in robosuite/robocasa-style models) so only the visual mesh is rendered."""
    group_num: int = 0
    """Geom group to hide when `hide_collision_geoms` is set."""
    hide_sites: bool = True
    """Whether to hide the site group given by `site_group_num` (e.g. frame markers and attachment sites) in the rendered frames."""
    site_group_num: int = 0
    """Site group to hide when `hide_sites` is set."""

    _camera: mujoco.MjvCamera = field(init=False, repr=False)
    """
    Free camera used to render every frame. Defaults are built from the
    model's own bounding statistics via :func:`mujoco.mjv_defaultFreeCamera`,
    since the scene defines no named camera to render from; any of
    `camera_lookat`/`camera_distance`/`camera_azimuth`/`camera_elevation`
    passed in override the corresponding default.
    """

    _scene_option: mujoco.MjvOption = field(init=False, repr=False)
    """Rendering options for every frame; hides group-0 (collision) geoms when `hide_collision_geoms` is set."""

    _frames: list = field(init=False, repr=False, default_factory=list)
    """RGB frames captured so far, in the order they were recorded."""

    def __post_init__(self):
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.simulator._mj_model, self._camera)
        if self.camera_lookat is not None:
            self._camera.lookat[:] = self.camera_lookat
        if self.camera_distance is not None:
            self._camera.distance = self.camera_distance
        if self.camera_azimuth is not None:
            self._camera.azimuth = self.camera_azimuth
        if self.camera_elevation is not None:
            self._camera.elevation = self.camera_elevation

        self._scene_option = mujoco.MjvOption()
        if self.hide_collision_geoms:
            self._scene_option.geomgroup[self.group_num] = 0
        if self.hide_sites:
            self._scene_option.sitegroup[self.site_group_num] = 0

    def capture_frame(self) -> None:
        """Renders the simulator's current state from the free camera and stores it as a frame."""
        with mujoco.Renderer(self.simulator._mj_model, self.height, self.width) as renderer:
            renderer.update_scene(
                self.simulator._mj_data, camera=self._camera, scene_option=self._scene_option
            )
            self._frames.append(renderer.render())

    def record_joint_motion(
        self,
        joint_name: str,
        target_value: float,
        number_of_frames: int = 60,
    ) -> None:
        """
        Moves `joint_name` from its current value to `target_value` in
        `number_of_frames` linear increments, capturing one frame per
        increment.

        :param joint_name: Name of the joint to move.
        :param target_value: Joint position to move toward.
        :param number_of_frames: Number of in-between frames to capture along the motion.
        """
        start_value = self.simulator.get_joint_value(joint_name).result
        for step in range(1, number_of_frames + 1):
            fraction = step / number_of_frames
            smooth_value = minimum_jerk(
                fraction
            )
            interpolated_value = start_value + smooth_value * (target_value - start_value)
            self.simulator.set_joint_value(joint_name, interpolated_value)
            mujoco.mj_kinematics(self.simulator._mj_model, self.simulator._mj_data)
            self.simulator.render()
            self.capture_frame()

    def save(self, output_path: str) -> None:
        """
        Writes every captured frame to `output_path` as an .mp4 video, in
        capture order.

        :param output_path: Filesystem path the video should be written to.
        :raises NoFramesCapturedError: If no frame was captured yet.
        """
        if not self._frames:
            raise NoFramesCapturedError("No frames were captured to write to a video.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with imageio.get_writer(
            str(output_path),
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in self._frames:
                writer.append_data(frame)


def build_joint_recorder(simulator: MujocoSimulator, joint_name: str) -> MotionRecorder:
    """
    Creates a :class:`MotionRecorder` whose free camera is framed on `joint_name`.

    :param simulator: The running MuJoCo simulator to record from.
    :param joint_name: Name of the joint whose parent body's parent the camera should look at.
    :return: A recorder ready to capture motions around the joint.
    """
    model = simulator._mj_model
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    parent_body_id = model.body_parentid[model.jnt_bodyid[joint_id]]
    parent_body_position = np.array(simulator._mj_data.xpos[parent_body_id])
    return MotionRecorder(
        simulator=simulator,
        camera_lookat=parent_body_position,
        # point the camera looks at
        camera_distance=1.5,
        # how far back the camera sits
        camera_azimuth=150,
        # rotate around the vertical axis
        camera_elevation=-20,
        # tilt down toward the scene
    )


def record_task_result(
    task_result: dict,
    simulator: MujocoSimulator,
    output_path: str,
    number_of_frames: int = 30,
    fps: int = 30,
) -> None:
    """
    Drives the joint named in `task_result` to its target value while
    recording the motion, and saves it to `output_path` as an .mp4 video.

    :param task_result: Dict as returned by :func:`interpretator.interpret_task`, must contain "joint_name" and "target_value".
    :param simulator: The running MuJoCo simulator to record from.
    :param output_path: Filesystem path the video should be written to.
    :param number_of_frames: Number of in-between frames to capture along the motion.
    :param fps: Frames per second of the output video.
    :raises UngroundedTaskError: If `task_result` could not be grounded to any joint.
    """
    joint_name = task_result["joint_name"]
    if joint_name is None:
        raise UngroundedTaskError(task_result["reasoning"])
    recorder = build_joint_recorder(simulator, joint_name)
    recorder.record_joint_motion(joint_name, task_result["target_value"], number_of_frames=number_of_frames)
    recorder.save(output_path)
    print('Saved video at:', output_path)