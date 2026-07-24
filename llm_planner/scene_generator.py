import argparse

import imageio
import mujoco
import numpy as np

from physics_simulators.mujoco_simulator import MujocoSimulator

cam =  mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat = np.array([0.0, 0.0, 0.5])  # point the camera looks at
cam.distance = 3.0                       # distance from lookat point
cam.azimuth = 90.0                       # horizontal angle (degrees)
cam.elevation = -30.0                    # vertical angle (degrees)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="robocasa.mp4",
        help="Output video filename.",
    )

    parser.add_argument(
        "--camera",
        default=cam,
        help="Camera name inside the XML.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--hold-start",
        type=float,
        default=0.7,
        help="Seconds to show the initial pose.",
    )

    parser.add_argument(
        "--hold-end",
        type=float,
        default=1.0,
        help="Seconds to show the final pose.",
    )

    return parser.parse_args()

def render_frame(
    renderer,
    data,
    camera_name,
):
    renderer.update_scene(
        data,
        camera=camera_name,
    )

    return renderer.render().copy()

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
def set_joint_position(simulator: MujocoSimulator,joint_name:str, joint_value):
    joint_id = mujoco.mj_name2id(
            m=simulator._mj_model, type=mujoco.mjtObj.mjOBJ_JOINT, name=joint_name
        )
    qpos_address = int(
        simulator._mj_model.jnt_qposadr[joint_id]
    )
    simulator._mj_data.qpos[qpos_address] = joint_value
    simulator._mj_data.qvel[:] = 0.0
    mujoco.mj_forward(
        simulator._mj_model,
        simulator._mj_data,
    )


plan = {'duration_seconds': 2.0}

def generate_video(simulator: MujocoSimulator,
    joint_info,
    camera_name,
    duration,
    output_path,
    fps,
    width,
    height,
    hold_start,
    hold_end,
):
    motion_frames = max(
        2,
        round(
            duration * fps
        ),
    )

    start_hold_frames = max(
        0,
        round(
            hold_start * fps
        ),
    )

    end_hold_frames = max(
        0,
        round(
            hold_end * fps
        ),
    )

    simulator._mj_model.vis.global_.offwidth = max(
        width, simulator._mj_model.vis.global_.offwidth
    )
    simulator._mj_model.vis.global_.offheight = max(
        height, simulator._mj_model.vis.global_.offheight
    )

    renderer = mujoco.Renderer(
        simulator._mj_model,
        height=height,
        width=width,
    )

    try:
        with imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:

            set_joint_position(simulator,joint_info['joint_name'],joint_info['current_value'])

            # Hold the initial position.
            for _ in range(start_hold_frames):
                frame = render_frame(
                    renderer=renderer,
                    data=simulator._mj_data,
                    camera_name=camera_name,
                )

                writer.append_data(frame)

            # Smoothly interpolate from start to target.
            for frame_index in range(motion_frames):
                interpolation_time = (
                    frame_index
                    / (motion_frames - 1)
                )

                smooth_value = minimum_jerk(
                    interpolation_time
                )

                qpos_value = (joint_info['current_value'] + (joint_info['target-value'] -
                                                              joint_info['current_value']) * smooth_value)
                set_joint_position(simulator,joint_info['joint_name'],qpos_value)

                frame = render_frame(
                    renderer=renderer,
                    data=simulator._mj_data,
                    camera_name=camera_name,
                )

                writer.append_data(frame)

            # Make sure the exact final pose is shown.
            set_joint_position(simulator,joint_info['name'],joint_info['target_value'])
            for _ in range(end_hold_frames):
                frame = render_frame(
                    renderer=renderer,
                    data=simulator._mj_data,
                    camera_name=camera_name,
                )

                writer.append_data(frame)

    finally:
        renderer.close()
