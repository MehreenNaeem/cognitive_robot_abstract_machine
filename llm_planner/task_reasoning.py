"""
task_grounding.py

Grounds natural-language manipulation tasks to scene description.

Pipeline
--------
1. Parse the URDF and build a registry of every joint: name, type, limits,
   parent/child link
2. Filter down to joints that are actually *actuatable* (revolute / prismatic)
3. Feed the task + the filtered joint registry to an LLM agent, which
   returns which joint to move and where.
4. The `llm_agent` function prompts the model and
   parses its structured JSON reply back into a joint decision.


"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Dict, List, Optional
from chatbot import chat_with_model, get_response
from physics_simulators.mujoco_simulator import MujocoSimulator
import spacy
from spacy.language import Language


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class JointInfo:
    name: str
    type: str
    parent: str
    child: str
    lower: Optional[float]
    upper: Optional[float]
    is_mimic: bool
    mimic_of: Optional[str]
    current_value: float = 0.0  # assume closed / zeroed unless tracked elsewhere

    def as_llm_context(self) -> dict:
        """Trimmed view of a joint, this is what gets shown to the LLM."""
        return {
            "joint_name": self.name,
            "type": self.type,
            "parent": self.parent,
            "limits": {"lower": self.lower, "upper": self.upper},
            "current_value": self.current_value,
        }

    def update_current_value(self, simulator: Optional[MujocoSimulator]) -> None:
        """
        Refreshes :attr:`current_value` from `simulator`'s live joint value.

        Leaves :attr:`current_value` unchanged if `simulator` is not a
        :class:`MujocoSimulator`.

        :param simulator: The simulator to read this joint's current value from.
        """
        value = get_joint_current_value(self.name, simulator)
        if value is None:
            return
        self.current_value = value


@dataclass
class ExecutedTask:
    """A previously executed task and the joint decision it produced."""
    task: str
    joint_name: str
    target_value: float
    reasoning: str = ""
    feedback: Optional[str] = None
    """User-supplied correction if this task's joint decision was wrong, else ``None``."""

    def as_llm_context(self) -> dict:
        """Trimmed view of this executed task, this is what gets shown to the LLM."""
        context = {
            "task": self.task,
            "joint_name": self.joint_name,
            "target_value": self.target_value,
        }
        if self.feedback is not None:
            context["feedback"] = self.feedback
        return context


@dataclass
class TaskHistory:
    """Buffer of tasks executed so far, so the LLM can see what is already done."""
    entries: List[ExecutedTask] = field(default_factory=list)

    def record(self, task: str, result: dict, feedback: Optional[str] = None) -> None:
        """
        Appends `result`'s joint decision for `task`, if `task` was grounded.

        :param task: The natural language instruction that produced `result`.
        :param result: The dict returned by :func:`ground_task` for `task`.
        :param feedback: User-supplied correction if the joint decision was wrong.
        """
        joint_name = result["joint_name"]
        if joint_name is None:
            return
        self.entries.append(
            ExecutedTask(
                task=task,
                joint_name=joint_name,
                target_value=result["target_value"],
                reasoning=result.get("reasoning", ""),
                feedback=feedback,
            )
        )

    def as_llm_context(self) -> List[dict]:
        """Trimmed view of every executed task, this is what gets shown to the LLM."""
        return [entry.as_llm_context() for entry in self.entries]

# --------------------------------------------------------------------------- #
# URDF parsing / grounding
# --------------------------------------------------------------------------- #

def find_grandparent(joint_el: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Optional[ET.Element]:
    """Find the body that encloses the body in which ``joint`` is nested.

    In MuJoCo-style XML a joint has no explicit ``<parent>`` element; instead
    it sits inside a ``body``, and that body's own enclosing ``body`` is the
    joint's parent link. This walks the element tree up two levels via
    ``parent_map`` instead of searching ``joint_el``'s children by tag.

    :param joint_el: The ``joint`` element to search from.
    :param parent_map: Maps each element to its direct parent element.
    :return: The grandparent ``body`` element, or ``None`` if not found.
    """
    body_el = parent_map.get(joint_el)
    if body_el is None:
        return None
    return body_el # parent_map.get(body_el)

def get_joint_current_value(name: str, simulator: Optional[MujocoSimulator]) -> Optional[float]:
    """
    Reads a joint's current value from a running MuJoCo simulation.

    :param name: The name of the joint to query.
    :param simulator: The simulator to query; only :class:`MujocoSimulator` is supported.
    :return: The joint's current position, or ``None`` if `simulator` is not a :class:`MujocoSimulator`.
    """
    if not isinstance(simulator, MujocoSimulator):
        return None
    return simulator.get_joint_value(name).result


def get_joint_limit(joint_el: ET.Element) -> tuple[Optional[float], Optional[float]]:
    """Find a joint's lower/upper limits.

    Limits may be given directly as a ``range`` or ``limit`` attribute on
    ``joint_el`` (MuJoCo style, e.g. ``range="0 1.57"``), or as ``lower``/
    ``upper`` attributes on a ``limit`` or ``range`` child element (URDF
    style).

    :param joint_el: The ``joint`` element to search.
    :return: A ``(lower, upper)`` tuple; either side may be ``None`` if not found.
    """
    range_value = joint_el.get("range") or joint_el.get("limit")
    if range_value is not None:
        lower_str, upper_str = range_value.split()
        return float(lower_str), float(upper_str)

    limit_el = joint_el.find("limit")
    if limit_el is None:
        limit_el = joint_el.find("range")
    if limit_el is None:
        return None, None

    lower = float(limit_el.get("lower")) if limit_el.get("lower") else None
    upper = float(limit_el.get("upper")) if limit_el.get("upper") else None
    return lower, upper

def parse_urdf_joints(urdf_path: str) -> Dict[str, JointInfo]:
    """Parse a URDF file and return every joint keyed by name."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    joints: Dict[str, JointInfo] = {}

    for joint_el in root.iter("joint"):
        name = joint_el.get("name")
        jtype = joint_el.get("type")

        parent_el = joint_el.find("parent")
        if parent_el is None:
            parent_el = find_grandparent(joint_el, parent_map)
        child_el = joint_el.find("child")
        parent = (parent_el.get("link") or parent_el.get("name")) if parent_el is not None else None
        child = child_el.get("link") if child_el is not None else None

        lower, upper = get_joint_limit(joint_el)

        mimic_el = joint_el.find("mimic")
        is_mimic = mimic_el is not None
        mimic_of = mimic_el.get("joint") if is_mimic else None

        joints[name] = JointInfo(
            name=name,
            type=jtype,
            parent=parent,
            child=child,
            lower=lower,
            upper=upper,
            is_mimic=is_mimic,
            mimic_of=mimic_of,
        )


    return joints


def actuatable_joints(joints: Dict[str, JointInfo]) -> Dict[str, JointInfo]:
    """
    Keep only joints that can meaningfully be commanded directly:
    revolute / prismatic / continuous, and not a mimic follower.
    Fixed joints and mimic joints are excluded (mimic joints move
    automatically once their driving joint moves).
    """
    actuatable_types = {"revolute", "prismatic", "continuous","hinge", "slide"}
    return {
        name: j
        for name, j in joints.items()
        if j.type in actuatable_types and not j.is_mimic
    }


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """
    Loads and caches the spaCy pipeline used to compare tasks against joints.

    :param model_name: Name of the installed spaCy pipeline to load.
    :return: The loaded spaCy pipeline.
    """
    return spacy.load(model_name)


def _humanize_identifier(identifier: Optional[str]) -> str:
    """
    Turns a snake_case/camelCase URDF identifier into space-separated words
    so it reads like natural language, e.g. 'microwave_door_joint' ->
    'microwave door joint'.

    :param identifier: A joint or link identifier, possibly ``None``.
    :return: The lower-cased, space-separated words of `identifier`.
    """
    if not identifier:
        return ""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", identifier)
    spaced = re.sub(r"[_\-]+", " ", spaced)
    return spaced.lower()


def _joint_description(joint: JointInfo) -> str:
    """
    Builds the natural-language description of `joint` that gets compared
    against the task instruction, from its name, parent link and type.

    :param joint: The joint to describe.
    :return: A space-separated string of `joint`'s name, parent and type.
    """
    parts = (_humanize_identifier(joint.name), _humanize_identifier(joint.parent))
    return " ".join(part for part in parts if part)


def candidates_filter(
    task: str,
    commandable: Dict[str, JointInfo],
    top_n: int = 50,
    nlp: Optional[Language] = None,
) -> Dict[str, JointInfo]:
    """
    Further narrows `commandable` down to the joints whose name, parent and
    type best match `task`, ranked by spaCy semantic similarity. This keeps
    the context passed to the LLM small and relevant instead of dumping the
    whole scene graph every time.

    :param task: Natural language instruction, e.g. "Open the microwave door".
    :param commandable: Candidate joints to rank and filter, keyed by name.
    :param top_n: Maximum number of best-matching joints to keep.
    :param nlp: spaCy pipeline used for similarity scoring; the default
        pipeline is loaded and cached when not given.
    :return: The `top_n` best-matching joints, keyed by name, ordered from
        best to worst match. Returns `commandable` unchanged if it is empty.
    """
    if not commandable:
        return commandable

    pipeline = nlp or _load_spacy_model()
    task_doc = pipeline(task.lower())

    scored_joints = sorted(
        commandable.values(),
        key=lambda joint: task_doc.similarity(pipeline(_joint_description(joint))),
        reverse=True,
    )
    return {joint.name: joint for joint in scored_joints[:top_n]}


# --------------------------------------------------------------------------- #
# LLM agent
# --------------------------------------------------------------------------- #

class LLMResponseError(Exception):
    """Raised when the LLM's response cannot be parsed into the expected joint-selection schema."""


_JSON_DECODER = json.JSONDecoder()


def _build_grounding_prompt(task: str, joint_context: List[dict], history_context: List[dict]) -> str:
    """Builds the prompt that asks the LLM to ground `task` to one of `joint_context`'s joints."""
    history_section = (
        "Previously completed tasks (JSON). An entry with a \"feedback\" field "
        "means the user said that joint decision was wrong; treat its "
        "\"feedback\" as a correction that takes priority over the original "
        "\"reasoning\" when grounding similar tasks now.\n"
        f"{json.dumps(history_context, indent=2)}\n\n"
        if history_context
        else ""
    )
    return (
        "You are grounding a natural-language manipulation task to a single "
        "controllable joint in a robot's scene description.\n\n"
        "A joint's closed/rest state is always value 0.0; \"current_value\" "
        "0.0 means the joint is fully closed. For an \"open\" task, choose a "
        "target_value from \"upper\" or \"lower_value\" that is anything other than 0.0, regardless of whether "
        "it is negative or positive.\n\n"
        f"{history_section}"
        f"Task: \"{task}\"\n\n"
        "Candidate joints (JSON):\n"
        f"{json.dumps(joint_context, indent=2)}\n\n"
        "Choose exactly one joint from the candidates above and a target "
        "value within its limits that satisfies the task, taking into "
        "account which tasks have already been completed. Respond with a "
        "single JSON object and nothing else, in this exact shape:\n"
        '{"joint_name": "<one of the candidate joint_name values>", '
        '"target_value": <float>, "reasoning": "<short explanation>"}'
    )


def _extract_json_object(text: str) -> dict:
    """Parses the first JSON object found in an LLM response, raising :class:`LLMResponseError` if none is found.

    Uses :meth:`json.JSONDecoder.raw_decode` instead of matching up to the
    last ``}`` in `text`, so trailing prose after the JSON object (which may
    itself contain stray braces) is not swept into the parsed object.
    """
    start = text.find("{")
    if start == -1:
        raise LLMResponseError(f"No JSON object found in LLM response: {text!r}")
    try:
        parsed_object, _ = _JSON_DECODER.raw_decode(text, start)
    except json.JSONDecodeError as error:
        raise LLMResponseError(f"No JSON object found in LLM response: {text!r}") from error
    return parsed_object


def _joint_by_name(joint_context: List[dict], joint_name: str) -> dict:
    """Looks up a joint's context entry by name, raising :class:`LLMResponseError` if the LLM hallucinated it."""
    for joint in joint_context:
        if joint["joint_name"] == joint_name:
            return joint
    raise LLMResponseError(f"LLM selected a joint that is not among the candidates: {joint_name!r}")


def llm_agent(
    task: str,
    joint_context: List[dict],
    url: str,
    user_id: str,
    max_attempts: int = 3,
    history: Optional[TaskHistory] = None,
) -> dict:
    """
    Grounds `task` to one of `joint_context`'s joints by prompting an LLM.

    Retries the query up to `max_attempts` times if the LLM selects a joint
    that is not among the candidates.

    :param task: Natural language instruction, e.g. "Open the cabinet3".
    :param joint_context: Candidate joints, each shaped like
        {"joint_name": ..., "type": ..., "limits": {"lower": ..., "upper": ...}, "current_value": ...}.
    :param url: Chat-completions endpoint to send the grounding prompt to.
    :param user_id: Authorization header value identifying the requesting user.
    :param max_attempts: Maximum number of times to query the LLM before giving up.
    :param history: Previously executed tasks, shown to the LLM so it knows
        which tasks are already done.
    :return: A dict with "joint_name", "limits", "current_value", "target_value" and "reasoning".
    """
    if not joint_context:
        return {
            "joint_name": None,
            "limits": None,
            "current_value": None,
            "target_value": None,
            "reasoning": "No candidate joints were found for this task.",
        }

    history_context = history.as_llm_context() if history else []
    prompt = _build_grounding_prompt(task, joint_context, history_context)
    for attempt in range(1, max_attempts + 1):
        response = chat_with_model(prompt, url, user_id)
        content = get_response(response["choices"][0]["message"]["content"])
        decision = _extract_json_object(content)
        try:
            chosen = _joint_by_name(joint_context, decision["joint_name"])
        except LLMResponseError:
            if attempt == max_attempts:
                raise
            continue

        return {
            "joint_name": chosen["joint_name"],
            "limits": chosen["limits"],
            "current_value": chosen["current_value"],
            "target_value": decision["target_value"],
            "reasoning": decision.get("reasoning", ""),
        }
    raise LLMResponseError(
        f"LLM failed to select a candidate joint after {max_attempts} attempts"
    )



def ground_task(
    urdf_path: str,
    task: str,
    url: str,
    user_id: str,
    simulator: Optional[MujocoSimulator] = None,
    history: Optional[TaskHistory] = None,
) -> dict:
    all_joints = parse_urdf_joints(urdf_path)
    commandable = actuatable_joints(all_joints)
    if len(commandable) > 50:
        commandable = candidates_filter(task, commandable)

    for joint in commandable.values():
        joint.update_current_value(simulator)

    context = [j.as_llm_context() for j in commandable.values()]
    result = llm_agent(task, context, url, user_id, history=history)
    return result

