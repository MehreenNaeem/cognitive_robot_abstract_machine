from __future__ import annotations

from dataclasses import dataclass, fields, MISSING

from typing_extensions import Any, Dict, List, Type

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

def get_required_parameters_name(
    action: Type[ActionDescription],):
    for name, signature in action_signatures([action]).items():
        return signature.required_parameters

def get_optional_parameters_name(
    action: Type[ActionDescription], ):
    for name, signature in action_signatures([action]).items():
        return signature.optional_parameters

def get_action_constraints_names(
    action: Type[ActionDescription],
):
    return [a.name for a in get_required_parameters_name(action)]


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