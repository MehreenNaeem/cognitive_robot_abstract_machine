
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import mujoco
from capture_motion import MotionRecorder, build_joint_recorder, record_task_result
from task_reasoning import ground_task, TaskHistory
from physics_simulators.base_simulator import SimulatorCallbackResult
from physics_simulators.mujoco_simulator import MujocoSimulator



class UngroundedTaskError(Exception):
    """Raised when an :func:`interpret_task` result could not be grounded to any joint."""


def interpret_task(
    path: str,
    task: str,
    url: str,
    user_id: str,
    simulator: MujocoSimulator,
    history: Optional[TaskHistory] = None,
) -> dict:
    """
    Grounds `task` to a URDF joint by delegating to :func:`ground_task`.

    :param path: Path to the URDF/XML file describing the scene.
    :param task: Natural language instruction, e.g. "Open the cabinet3".
    :param url: Chat-completions endpoint to send the grounding prompt to.
    :param user_id: Authorization header value identifying the requesting user.
    :param simulator: The running simulator to read live joint values from.
    :param history: Previously executed tasks, shown to the LLM so it knows
        which tasks are already done.
    :return: A dict with "joint_name", "limits", "current_value", "target_value" and "reasoning".
    """
    return ground_task(path, task, url, user_id, simulator, history)

def initialize_Mujoco(file_path: str):
    """Initialize Mujoco simulator."""
    sim = MujocoSimulator(_headless=False, file_path=file_path)
    sim.start(simulate_in_thread=True)
    sim.pause()
    joint_names = sim.get_all_joint_names().result
    sim.set_joints_values({name: 0.0 for name in joint_names})
    mujoco.mj_kinematics(sim._mj_model, sim._mj_data)
    sim.render()
    return sim



def update_joint_state_Mujoco(
    task_result: dict, simulator: MujocoSimulator, velocity: float = 0.001
) -> SimulatorCallbackResult:
    """
    Applies the joint decision from :func:`interpret_task` to a running MuJoCo simulation.

    :param task_result: Dict returned by :func:`interpret_task`, must contain "joint_name" and "target_value".
    :param simulator: The running :class:`~physics_simulators.mujoco_simulator.MujocoSimulator` to update.
    :param velocity: The velocity to set on the joint alongside its target value.
    :return: The :class:`SimulatorCallbackResult` from setting the joint velocity.
    :raises UngroundedTaskError: If `task_result` could not be grounded to any joint.
    """
    joint_name = task_result["joint_name"]
    if joint_name is None:
        raise UngroundedTaskError(task_result["reasoning"])
    simulator.set_joint_value(joint_name, task_result["target_value"])
    mujoco.mj_kinematics(simulator._mj_model, simulator._mj_data)
    simulator.render()
    return simulator.set_joint_velocity(joint_name, velocity)

def collect_joint_decision_feedback(result: dict) -> Optional[str]:
    """
    Asks the user, via the interactive CLI, whether `result`'s joint decision
    correctly satisfied the task, and if not, collects a free-text correction.

    :param result: The dict returned by :func:`interpret_task` for the task just executed.
    :return: The user's correction, or ``None`` if they confirmed the decision was correct.
    """
    prompt = f"Was setting {result['joint_name']} to {result['target_value']} correct? [y/N]> "
    is_correct = input(prompt).strip().lower()
    if is_correct in {"y", "yes"}:
        return None
    return input("What should have happened instead?> ").strip() or None


def run_interactive_session(file_path: str, url: str, user_id: str) -> None:
    """
    Repeatedly prompts the user for a natural-language task, grounds and
    executes each one on a live MuJoCo simulation, and stops when the user
    asks to exit.

    :param file_path: Path to the URDF/XML scene file to simulate.
    :param url: Chat-completions endpoint to send grounding prompts to.
    :param user_id: Authorization header value identifying the requesting user.
    """
    exit_commands = frozenset({"exit", "quit", "q"})
    simulator = initialize_Mujoco(file_path)
    history = TaskHistory()

    print("Enter a task to execute (e.g. 'Open the dishwasher door'), or 'exit' to quit.")
    while True:
        task_description = input("Task> ").strip()
        if not task_description or task_description.lower() in exit_commands:
            print("Exiting.")
            break

        try:
            result = interpret_task(
                file_path, task_description, url=url, user_id=user_id,
                simulator=simulator, history=history,
            )
            update_joint_state_Mujoco(result, simulator)
        except UngroundedTaskError as error:
            print(f"Could not ground task {task_description!r}: {error}")
            continue

        print(f"Done: Joint({result['joint_name']}) {result['reasoning']}")
        feedback = collect_joint_decision_feedback(result)
        history.record(task_description, result, feedback=feedback)


def run_single_task_session(file_path: str, url: str, user_id: str, task: str) -> dict:
    """
    Grounds and executes a single natural-language task on a live MuJoCo
    simulation, without task history or interactive user feedback.

    :param file_path: Path to the URDF/XML scene file to simulate.
    :param url: Chat-completions endpoint to send the grounding prompt to.
    :param user_id: Authorization header value identifying the requesting user.
    :param task: Natural language instruction, e.g. "Open the dishwasher door".
    :return: The dict returned by :func:`interpret_task` for the executed task.
    :raises UngroundedTaskError: If `task` could not be grounded to any joint.
    """
    simulator = initialize_Mujoco(file_path)
    result = interpret_task(file_path, task, url=url, user_id=user_id, simulator=simulator)
    update_joint_state_Mujoco(result, simulator)
    return result

def run_single_task_with_recorder(file_path: str, url: str, user_id: str, task: str) -> dict:
    simulator = initialize_Mujoco(file_path)
    result = interpret_task(file_path, task, url=url, user_id=user_id, simulator=simulator)
    file_name = f'{task.replace(" ", "_")}.mp4'
    record_task_result(result, simulator, file_name)
    print(result)

def run_multiple_task_with_recorder(
    file_path: str,
    url: str,
    user_id: str,
    single_video: bool = False,
    combined_video_path: str = "all_tasks.mp4",
) -> list[dict]:
    """
    Repeatedly prompts the user for natural-language tasks, grounds and
    executes each one on a live MuJoCo simulation while recording the motion,
    and stops when the user asks to exit.

    :param file_path: Path to the URDF/XML scene file to simulate.
    :param url: Chat-completions endpoint to send grounding prompts to.
    :param user_id: Authorization header value identifying the requesting user.
    :param single_video: If ``True``, all task motions are recorded into one
        video at `combined_video_path`; otherwise each task is saved to its
        own video named after the task.
    :param combined_video_path: Output path of the combined video, only used
        when `single_video` is set.
    :return: The :func:`interpret_task` results of every executed task, in order.
    """
    exit_commands = frozenset({"exit", "quit", "q"})
    simulator = initialize_Mujoco(file_path)
    history = TaskHistory()
    combined_recorder: Optional[MotionRecorder] = None
    results: list[dict] = []

    print("Enter a task to execute (e.g. 'Open the dishwasher door'), or 'exit' to quit.")
    while True:
        task_description = input("Task> ").strip()
        if not task_description or task_description.lower() in exit_commands:
            print("Exiting.")
            break

        try:
            result = interpret_task(
                file_path, task_description, url=url, user_id=user_id,
                simulator=simulator, history=history,
            )
            if single_video:
                joint_name = result["joint_name"]
                if joint_name is None:
                    raise UngroundedTaskError(result["reasoning"])
                if combined_recorder is None:
                    combined_recorder = build_joint_recorder(simulator, joint_name)
                combined_recorder.record_joint_motion(joint_name, result["target_value"])
            else:
                file_name = f'{task_description.replace(" ", "_")}.mp4'
                record_task_result(result, simulator, file_name)
        except UngroundedTaskError as error:
            print(f"Could not ground task {task_description!r}: {error}")
            continue

        print(f"Done: Joint({result['joint_name']}) {result['reasoning']}")
        history.record(task_description, result)
        results.append(result)

    if combined_recorder is not None:
        combined_recorder.save(combined_video_path)
        print('Saved video at:', combined_video_path)
    return results


