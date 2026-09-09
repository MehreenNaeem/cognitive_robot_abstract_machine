from __future__ import annotations
from semantic_digital_twin.world import World

from dataclasses import dataclass
from enum import StrEnum

from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
)
from llm_planner.env_reasoner.action_constraints import AVAILABLE_ACTIONS
from semantic_digital_twin.semantic_annotations.mixins import (
    HasDoors,
    HasHandle,
    IsStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bread,
    Cabinet,
    CookingContainer,
    CounterTop,
    Cuttlery,
    Drawer,
    Food,
    FoodContainer,
    Fruit,
    Furniture,
    Milk,
    Oven,
    Shelf,
    Sink,
    Table,
    TomatoSoup,
    ToolWithHandle,
    Vegetable,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


class ObjectAffordance(StrEnum):
    """
    A manipulation capability a semantic annotation type affords.
    """

    GRASPABLE = "graspable"
    CUTABLE = "cutable"
    POURABLE = "pourable"
    FILLABLE = "fillable"
    OPENABLE = "openable"
    CLOSABLE = "closable"
    RECEPTACLE = "receptacle"


@dataclass(frozen=True)
class AffordanceRule:
    """
    Grants a fixed set of affordances to every semantic annotation type that is a
    subclass of :attr:`annotation_type`.
    """

    annotation_type: type[SemanticAnnotation]
    """
    Semantic annotation type, or mixin, this rule matches, including its subclasses.
    """

    affordances: frozenset[ObjectAffordance]
    """
    Affordances granted to every matching semantic annotation type.
    """


#: Food subtypes whose loaf/block shape can be sliced, in addition to :class:`Fruit`
#: and :class:`Vegetable`, which are already covered by their own rule below.
ADDITIONAL_CUTABLE_FOOD_TYPES: frozenset[type[Food]] = frozenset({Bread})

#: Food subtypes that are liquid or semi-liquid and can be poured out of their
#: container.
POURABLE_FOOD_TYPES: frozenset[type[Food]] = frozenset({Milk, TomatoSoup})

#: Rules matched from most to least specific; a type's affordances are the union of
#: every rule whose :attr:`~AffordanceRule.annotation_type` it is a subclass of.
AFFORDANCE_RULES: list[AffordanceRule] = [
    AffordanceRule(
        Fruit, frozenset({ObjectAffordance.GRASPABLE, ObjectAffordance.CUTABLE})
    ),
    AffordanceRule(
        Vegetable, frozenset({ObjectAffordance.GRASPABLE, ObjectAffordance.CUTABLE})
    ),
    AffordanceRule(Food, frozenset({ObjectAffordance.GRASPABLE})),
    AffordanceRule(FoodContainer, frozenset({ObjectAffordance.GRASPABLE})),
    AffordanceRule(Furniture, frozenset({ObjectAffordance.RECEPTACLE})),
    AffordanceRule(Cabinet, frozenset({ObjectAffordance.RECEPTACLE})),
    AffordanceRule(Drawer, frozenset({ObjectAffordance.RECEPTACLE})),
    AffordanceRule(CounterTop, frozenset({ObjectAffordance.RECEPTACLE})),
    AffordanceRule(Shelf, frozenset({ObjectAffordance.RECEPTACLE})),
    AffordanceRule(Sink, frozenset({ObjectAffordance.RECEPTACLE})),
    AffordanceRule(Oven, frozenset({ObjectAffordance.RECEPTACLE})),
    AffordanceRule(Cuttlery, frozenset({ObjectAffordance.GRASPABLE})),
    AffordanceRule(
        HasHandle, frozenset({ObjectAffordance.OPENABLE, ObjectAffordance.CLOSABLE})
    ),
    AffordanceRule(
        HasDoors, frozenset({ObjectAffordance.OPENABLE, ObjectAffordance.CLOSABLE})
    ),
]


class ObjectAffordanceReasoner:
    """
    Determines the manipulation affordances of a semantic annotation type.

    Matches the type against :data:`AFFORDANCE_RULES` by subclass, then refines the
    generic :attr:`ObjectAffordance.CUTABLE`/:attr:`ObjectAffordance.POURABLE`
    affordances for :class:`Food` subtypes that are not already covered by the
    :class:`Fruit`/:class:`Vegetable` rule, using :data:`ADDITIONAL_CUTABLE_FOOD_TYPES`
    and :data:`POURABLE_FOOD_TYPES`.
    """

    rules: list[AffordanceRule] = AFFORDANCE_RULES

    def affordances_for(
        self, annotation_type: type[SemanticAnnotation]
    ) -> frozenset[ObjectAffordance]:
        """
        Compute every affordance that applies to ``annotation_type``.

        :param annotation_type: A semantic annotation class from
            :mod:`semantic_digital_twin.semantic_annotations.semantic_annotations`.
        :return: The union of affordances granted by every matching rule.
        """
        affordances: set[ObjectAffordance] = set()
        for rule in self.rules:
            if issubclass(annotation_type, rule.annotation_type):
                affordances.update(rule.affordances)

        if annotation_type in ADDITIONAL_CUTABLE_FOOD_TYPES:
            affordances.add(ObjectAffordance.CUTABLE)
        if annotation_type in POURABLE_FOOD_TYPES:
            affordances.add(ObjectAffordance.POURABLE)

        return frozenset(affordances)


@dataclass(frozen=True)
class ActionAffordanceRequirement:
    """
    States the affordances a semantic annotation type must provide for :attr:`action` to
    be performable on it.
    """

    action: type[ActionDescription]
    """
    Action class this requirement applies to.
    """

    required_affordances: frozenset[ObjectAffordance]
    """
    Affordances an object must provide, all of them, for :attr:`action` to be
    performable on it.
    """


#: Object-directed actions from :data:`~llm_planner.env_reasoner.action_constraints.AVAILABLE_ACTIONS`,
#: matched to the affordances they require of their target object.
ACTION_AFFORDANCE_REQUIREMENTS: list[ActionAffordanceRequirement] = [
    ActionAffordanceRequirement(OpenAction, frozenset({ObjectAffordance.OPENABLE})),
    ActionAffordanceRequirement(CloseAction, frozenset({ObjectAffordance.CLOSABLE})),
    ActionAffordanceRequirement(PickUpAction, frozenset({ObjectAffordance.GRASPABLE})),
    ActionAffordanceRequirement(PlaceAction, frozenset({ObjectAffordance.GRASPABLE})),
    ActionAffordanceRequirement(CuttingAction, frozenset({ObjectAffordance.CUTABLE})),
    ActionAffordanceRequirement(PouringAction, frozenset({ObjectAffordance.POURABLE})),
]


@dataclass(frozen=True)
class ActionParameterTypeRequirement:
    """
    Requires a named parameter of :attr:`action` to be a subclass of one of
    :attr:`accepted_types`.

    This captures constraints the affordance model cannot express, such as a tool that
    has to be held by its handle, or a place target that has to be a storage space.
    """

    action: type[ActionDescription]
    """
    Action class this requirement applies to.
    """

    parameter: str
    """
    Name of the action parameter the requirement constrains.
    """

    accepted_types: frozenset[type[SemanticAnnotation]]
    """
    Semantic annotation types the parameter's object may be a subclass of; a subclass
    of any one of them satisfies the requirement.
    """


#: Per-parameter semantic type constraints, matched to actions in
#: :data:`~llm_planner.env_reasoner.action_constraints.AVAILABLE_ACTIONS`.
ACTION_PARAMETER_TYPE_REQUIREMENTS: list[ActionParameterTypeRequirement] = [
    ActionParameterTypeRequirement(MixingAction, "tool", frozenset({ToolWithHandle})),
    ActionParameterTypeRequirement(
        MixingAction, "container", frozenset({CookingContainer})
    ),
    ActionParameterTypeRequirement(
        PlaceAction, "target_location", frozenset({IsStorageSpace, Table})
    ),
]


def actions_available_for(
    reasoner: ObjectAffordanceReasoner, annotation_type: type[SemanticAnnotation]
) -> list[type[ActionDescription]]:
    """
    Determine which available actions can be performed with a semantic annotation type.

    An action is available when ``annotation_type`` provides every affordance in a
    matching :data:`ACTION_AFFORDANCE_REQUIREMENTS` entry, or is a subclass of one of
    the accepted types in a matching :data:`ACTION_PARAMETER_TYPE_REQUIREMENTS` entry.
    Only actions in
    :data:`~llm_planner.env_reasoner.action_constraints.AVAILABLE_ACTIONS` are
    considered.

    :param reasoner: Reasoner computing the affordances of ``annotation_type``.
    :param annotation_type: A semantic annotation class to check actions against.
    :return: The matching actions, affordance matches first, without duplicates.
    """
    object_affordances = reasoner.affordances_for(annotation_type)
    available: list[type[ActionDescription]] = []
    for requirement in ACTION_AFFORDANCE_REQUIREMENTS:
        if (
            requirement.action in AVAILABLE_ACTIONS
            and requirement.action not in available
            and requirement.required_affordances <= object_affordances
        ):
            available.append(requirement.action)
    for requirement in ACTION_PARAMETER_TYPE_REQUIREMENTS:
        if (
            requirement.action in AVAILABLE_ACTIONS
            and requirement.action not in available
            and issubclass(annotation_type, tuple(requirement.accepted_types))
        ):
            available.append(requirement.action)
    return available


'''
#Usage Example:
if __name__ == "__main__":
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Apple,
        Door,
        Drawer,
        TunaCan,
        Spoon,
        Cabinet,
    )

    reasoner = ObjectAffordanceReasoner()
    for demo_type in [Apple, Bread, Milk, TunaCan, Drawer, Door, Spoon, Cabinet]:
        demo_affordances = reasoner.affordances_for(demo_type)
        print(f"{demo_type.__name__}: {sorted(a.value for a in demo_affordances)}")
'''