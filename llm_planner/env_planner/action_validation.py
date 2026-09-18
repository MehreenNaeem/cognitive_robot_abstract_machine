"""
Checking an action's pre- or post-condition without performing it.

An action's ``pre_condition``/``post_condition`` (see
:meth:`~coraplex.robot_plans.actions.base.ActionDescription.pre_condition` and
:meth:`~coraplex.robot_plans.actions.base.ActionDescription.post_condition`) build an
Entity Query Language expression from the action's own field values; they do not say
whether that expression holds. :func:`check_pre_condition` and
:func:`check_post_condition` build it and evaluate it - the same check
:meth:`~coraplex.robot_plans.actions.base.ActionDescription.perform` runs before and
after executing the action - so a grounded action can be probed without performing it,
before and after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Tuple, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from krrood.entity_query_language.factories import (
    evaluate_condition,
    get_false_statements,
)
from llm_planner.env_reasoner.action_constraints import ConditionKind

CHECKABLE_ACTIONS: Tuple[Type[ActionDescription], ...] = (
    NavigateAction,
    OpenAction,
    CloseAction,
    PickUpAction,
    PlaceAction,
    MixingAction,
    CuttingAction,
    PouringAction,
)
"""
The action types :func:`check_pre_condition` and :func:`check_post_condition` are
written and tested for.
"""


class ActionNotCheckable(TypeError):
    """
    Raised when a condition is checked for an action outside :data:`CHECKABLE_ACTIONS`.
    """

    def __init__(self, action: ActionDescription) -> None:
        checkable = ", ".join(action_type.__name__ for action_type in CHECKABLE_ACTIONS)
        super().__init__(
            f"{type(action).__name__} is not one of the actions this module checks "
            f"conditions for: {checkable}."
        )
        self.action = action
        """
        The action whose type is not supported.
        """


@dataclass(frozen=True)
class ConditionCheck:
    """
    Whether one action's pre- or post-condition holds right now, and why not when it
    does not.
    """

    action: Type[ActionDescription]
    """
    The action class the condition was checked for.
    """

    kind: ConditionKind
    """
    Whether this is the action's pre- or post-condition.
    """

    satisfied: bool
    """
    Whether the condition holds.
    """

    false_statement_names: List[str] = field(default_factory=list)
    """
    Names of the Entity Query Language predicates that made the condition false, where
    the expression can attribute one.

    Always empty when the condition holds. Also empty when it is a plain
    ``True``/``False`` rather than a predicate expression - several of
    :data:`CHECKABLE_ACTIONS` declare no condition of their own for one of the two kinds
    and inherit the unconditional default - and can be empty even for an unsatisfied
    predicate expression whose failing part is not itself a named predicate (e.g. a
    plain Python function wrapped into the condition). Treat this as best-effort detail,
    not an exhaustive account of why the condition failed.
    """


def _check_condition(
    action: ActionDescription, context: Context, kind: ConditionKind
) -> ConditionCheck:
    """
    Build and evaluate one action's pre- or post-condition.

    :param action: The action to check, already built with its arguments.
    :param context: The context to check it in - the world and robot the condition
        reads.
    :param kind: Which of the action's two conditions to check.
    :return: Whether it holds, and which predicates made it false if not.
    :raises ActionNotCheckable: If ``action`` is not one of :data:`CHECKABLE_ACTIONS`.
    """
    if not isinstance(action, CHECKABLE_ACTIONS):
        raise ActionNotCheckable(action)
    condition_method = (
        action.pre_condition if kind is ConditionKind.PRE else action.post_condition
    )
    condition = condition_method(
        action.bound_variables, context, action.designator_parameter
    )
    satisfied = evaluate_condition(condition)
    false_statement_names = (
        []
        if satisfied or isinstance(condition, bool)
        else [statement._name_ for statement in get_false_statements(condition)]
    )
    return ConditionCheck(
        action=type(action),
        kind=kind,
        satisfied=satisfied,
        false_statement_names=false_statement_names,
    )


def check_pre_condition(action: ActionDescription, context: Context) -> ConditionCheck:
    """
    Check whether an action's pre-condition holds right now.

    :param action: The action to check, already built with its arguments.
    :param context: The context to check it in - the world and robot the condition
        reads.
    :return: Whether the condition holds, and which predicates made it false if not.
    :raises ActionNotCheckable: If ``action`` is not one of :data:`CHECKABLE_ACTIONS`.
    """
    return _check_condition(action, context, ConditionKind.PRE)


def check_post_condition(action: ActionDescription, context: Context) -> ConditionCheck:
    """
    Check whether an action's post-condition holds right now.

    Reads the same way before the action has run as after it: the condition is built
    from the action's field values and the world's current state, whatever that state
    is. Checked before performing, it answers "has this already happened" rather than
    "did performing the action work" - it is :meth:`ActionDescription.perform`'s check
    after execution that gives the latter its meaning.

    :param action: The action to check, already built with its arguments.
    :param context: The context to check it in - the world and robot the condition
        reads.
    :return: Whether the condition holds, and which predicates made it false if not.
    :raises ActionNotCheckable: If ``action`` is not one of :data:`CHECKABLE_ACTIONS`.
    """
    return _check_condition(action, context, ConditionKind.POST)
