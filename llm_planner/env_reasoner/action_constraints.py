from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, fields, MISSING
from enum import StrEnum

from typing_extensions import Any, Callable, Dict, List, Type

from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.actions.composite.tool_based import (
    MixingAction,
    CuttingAction,
    PouringAction,
)
from krrood.entity_query_language.predicate import Predicate


@dataclass
class ActionParameter:
    """
    Describes a single constructor parameter of an action.
    """

    name: str
    """
    Name of the parameter.
    """

    type_annotation: str
    """
    Type annotation of the parameter.
    """

    required: bool
    """
    Whether the parameter has to be provided by the caller.
    """

    default: Any = None
    """
    Default value of the parameter, if it is optional.
    """

    description: str = ""
    """
    Docstring of the field, if available.
    """


@dataclass
class ActionSignature:
    """
    The full parameter signature of an action, split into required and optional parameters.
    """

    action: Type[ActionDescription]
    """
    The action class this signature belongs to.
    """

    parameters: List[ActionParameter]
    """
    All constructor parameters of the action.
    """

    @property
    def required_parameters(self) -> List[ActionParameter]:
        """
        Parameters that must be provided to instantiate the action.
        """
        return [parameter for parameter in self.parameters if parameter.required]

    @property
    def optional_parameters(self) -> List[ActionParameter]:
        """
        Parameters that fall back to a default value when omitted.
        """
        return [parameter for parameter in self.parameters if not parameter.required]

    @classmethod
    def from_action(cls, action: Type[ActionDescription]) -> ActionSignature:
        """
        Build the signature of an action by inspecting its dataclass fields.
        """
        parameters = [
            ActionParameter(
                name=field.name,
                type_annotation=str(field.type),
                required=field.default is MISSING and field.default_factory is MISSING,
                default=None if field.default is MISSING else field.default,
            )
            for field in fields(action)
            if field.init
        ]
        return cls(action=action, parameters=parameters)


def action_signatures(
    actions: List[Type[ActionDescription]],
) -> Dict[str, ActionSignature]:
    """
    Map each action class name to its constructor signature.
    """
    return {action.__name__: ActionSignature.from_action(action) for action in actions}


AVAILABLE_ACTIONS: List[Type[ActionDescription]] = [
    NavigateAction,
    LookAtAction,
    DetectAction,
    OpenAction,
    CloseAction,
    PickUpAction,
    PlaceAction,
    MoveTorsoAction,
    ParkArmsAction,
    MixingAction,
    CuttingAction,
    PouringAction,
]

#: Actions that move the robot or its sensors rather than acting on a target object.
LOCOMOTION_ACTIONS: List[Type[ActionDescription]] = [
    NavigateAction,
    LookAtAction,
    DetectAction,
    MoveTorsoAction,
    ParkArmsAction,
]

INTERACTIVE_ACTION : List[Type[ActionDescription]] = [
    OpenAction,
    CloseAction,
    PickUpAction,
    PlaceAction,
    CuttingAction,
    PouringAction,
]

One_OBJECT_ACTION : List[Type[ActionDescription]] = [
    PickUpAction,
    CloseAction,
    OpenAction,
    NavigateAction,
]
TWO_OBJECT_ACTION : List[Type[ActionDescription]] = [
    PlaceAction,
    CuttingAction,
    PouringAction,
]

def get_required_parameters_name(
    action: Type[ActionDescription],
):
    for name, signature in action_signatures([action]).items():
        return signature.required_parameters


def get_optional_parameters_name(
    action: Type[ActionDescription],
):
    for name, signature in action_signatures([action]).items():
        return signature.optional_parameters


def get_action_constraints_names(
    action: Type[ActionDescription],
):
    return [a.name for a in get_required_parameters_name(action)]


# %% pre- and post-conditions


class ConditionKind(StrEnum):
    """
    Whether a condition has to hold before an action runs or after it has run.
    """

    PRE = "pre"
    """
    Must hold before the action is performed.
    """

    POST = "post"
    """
    Must hold once the action has been performed.
    """


def _condition_method(
    action: Type[ActionDescription], kind: ConditionKind
) -> Callable[..., Any]:
    """
    Return the ``pre_condition`` or ``post_condition`` callable of an action.

    :param action: The action class to read the condition from.
    :param kind: Which of the two conditions to return.
    :return: The (possibly inherited) static method building the EQL condition.
    """
    if kind is ConditionKind.PRE:
        return action.pre_condition
    return action.post_condition


def _referenced_predicates(condition_method: Callable[..., Any]) -> List[str]:
    """
    Names of the EQL predicate classes a condition method constructs.

    The method body is read statically, so no world, robot or context is needed; a
    predicate reached only through a helper function is not seen.

    :param condition_method: A ``pre_condition``/``post_condition`` static method.
    :return: The distinct :class:`~krrood.entity_query_language.predicate.Predicate`
        subclass names it names directly, sorted.
    """
    source = textwrap.dedent(inspect.getsource(condition_method))
    scope = condition_method.__globals__
    predicate_names = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and isinstance(scope.get(node.func.id), type)
        and issubclass(scope[node.func.id], Predicate)
    }
    return sorted(predicate_names)


@dataclass
class ActionCondition:
    """
    A pre- or post-condition an action declares as an Entity Query Language expression.
    """

    action: Type[ActionDescription]
    """
    The action class this condition belongs to.
    """

    kind: ConditionKind
    """
    Whether the condition is checked before or after the action.
    """

    description: str
    """
    The condition method's docstring, i.e. the constraint stated in prose.
    """

    predicates: List[str]
    """
    Names of the EQL predicate classes the condition builds directly.
    """

    is_declared: bool
    """
    Whether the action defines this condition itself; ``False`` means it inherits the
    always-true default from :class:`~coraplex.robot_plans.actions.base.ActionDescription`.
    """


def action_condition(
    action: Type[ActionDescription], kind: ConditionKind
) -> ActionCondition:
    """
    Describe one condition of an action by reusing its EQL condition method.

    :param action: The action class to inspect.
    :param kind: Which condition to describe.
    :return: The condition's prose, referenced predicates and whether it is declared.
    """
    condition_method = _condition_method(action, kind)
    is_declared = condition_method is not _condition_method(ActionDescription, kind)
    return ActionCondition(
        action=action,
        kind=kind,
        description=inspect.getdoc(condition_method) or "",
        predicates=_referenced_predicates(condition_method) if is_declared else [],
        is_declared=is_declared,
    )


def action_conditions(
    actions: List[Type[ActionDescription]],
) -> Dict[str, List[ActionCondition]]:
    """
    Map each action class name to its pre- and post-condition, in that order.
    """
    return {
        action.__name__: [
            action_condition(action, ConditionKind.PRE),
            action_condition(action, ConditionKind.POST),
        ]
        for action in actions
    }



#Usage Example:
if __name__ == "__main__":
    for name, signature in action_signatures(AVAILABLE_ACTIONS).items():
        print(f"\n{name}")
        for required_parameter in signature.required_parameters:
            print(
                f"  required: {required_parameter.name}: "
                f"{required_parameter.type_annotation}"
            )
        for optional_parameter in signature.optional_parameters:
            print(
                f"  optional: {optional_parameter.name}: "
                f"{optional_parameter.type_annotation} "
                f"= {optional_parameter.default!r}"
            )
        for condition in action_conditions([signature.action])[name]:
            if not condition.is_declared:
                continue
            print(f"  {condition.kind.value}-condition: {condition.description}")
            if condition.predicates:
                print(f"    predicates: {', '.join(condition.predicates)}")
