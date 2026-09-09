from __future__ import annotations

from dataclasses import dataclass

from coraplex.robot_plans.actions.base import ActionDescription
from llm_planner.env_reasoner.action_constraints import (
    ActionCondition,
    ConditionKind,
    action_condition,
    get_action_constraints_names,
)
from llm_planner.env_reasoner.object_constraints import (
    ACTION_AFFORDANCE_REQUIREMENTS,
    ACTION_PARAMETER_TYPE_REQUIREMENTS,
    ActionParameterTypeRequirement,
    ObjectAffordance,
    ObjectAffordanceReasoner,
    actions_available_for,
)
from llm_planner.env_reasoner.object_graphs import (
    mixin_bases,
    semantic_annotation_graph,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasMechanicalJoint,
    HasRootBody,
    HasCaseAsRootBody,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Furniture
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

#: Affordances each object-directed action requires of its target, keyed by action class.
REQUIRED_AFFORDANCES_BY_ACTION: dict[
    type[ActionDescription], frozenset[ObjectAffordance]
] = {
    requirement.action: requirement.required_affordances
    for requirement in ACTION_AFFORDANCE_REQUIREMENTS
}


# %% one doable action
"""
        For a semantic-annotation object type → every action doable on it, with the constraints each action carries.
        
        object type ──(object_constraints)──► affordances ──► doable actions
        │
        (action_constraints)◄────────────────┘
        required params + pre/post conditions
"""


@dataclass
class DoableAction:
    """
    An action that can be performed on a semantic annotation type, with the constraints
    that reaching a valid plan for it imposes.
    """

    action: type[ActionDescription]
    """
    The action class that can be performed on the object.
    """

    required_affordances: frozenset[ObjectAffordance]
    """
    Affordances the object provides that make this action applicable.
    """

    required_parameters: list[str]
    """
    Names of the action parameters the caller has to supply.
    """

    pre_condition: ActionCondition
    """
    The action's precondition, read from its Entity Query Language condition method.
    """

    post_condition: ActionCondition
    """
    The action's postcondition, read from its Entity Query Language condition method.
    """

    parameter_type_requirements: list[ActionParameterTypeRequirement]
    """
    Semantic type each named action parameter must satisfy, e.g. ``MixingAction`` needs
    a ``ToolWithHandle`` tool and a ``CookingContainer`` container, and ``PlaceAction``
    needs an ``IsStorageSpace`` or ``Table`` target location.
    """


# %% doable actions for one object


@dataclass
class ObjectActionAffordances:
    """
    Every action that can be performed on one semantic annotation type.
    """

    annotation_type: type[SemanticAnnotation]
    """
    The semantic annotation class the object is an instance of.
    """

    affordances: frozenset[ObjectAffordance]
    """
    The manipulation affordances the object provides.
    """

    mixins: list[type]
    """
    The object's base classes beyond its place in the subclass graph, e.g.
    :class:`~semantic_digital_twin.semantic_annotations.mixins.IsPerceivable`.
    """

    doable_actions: list[DoableAction]
    """
    The actions applicable to the object, one entry per action.
    """


#: Bases every annotation type carries as structural plumbing rather than as a
#: distinguishing property, so the object-properties listing leaves them out.
HIDDEN_PROPERTY_BASES: list[type] = [HasRootBody,HasCaseAsRootBody, HasMechanicalJoint, Furniture]


def _mixins_of(annotation_type: type[SemanticAnnotation]) -> list[type]:
    """
    The base classes an object's properties are read from.

    :param annotation_type: The object type to inspect.
    :return: Every direct base class except those in :data:`HIDDEN_PROPERTY_BASES`,
        e.g. ``[FoodContainer, IsPerceivable]`` for :class:`Milk`.
    """
    return mixin_bases(annotation_type, HIDDEN_PROPERTY_BASES)


def _doable_actions(
    annotation_type: type[SemanticAnnotation], reasoner: ObjectAffordanceReasoner
) -> list[DoableAction]:
    """
    Build the :class:`DoableAction` list for a semantic annotation type.

    :param annotation_type: The object type to check actions against.
    :param reasoner: Reasoner computing the object's affordances.
    :return: One :class:`DoableAction` per applicable action.
    """
    object_affordances = reasoner.affordances_for(annotation_type)
    return [
        DoableAction(
            action=action,
            required_affordances=REQUIRED_AFFORDANCES_BY_ACTION.get(action, frozenset())
            & object_affordances,
            required_parameters=get_action_constraints_names(action),
            pre_condition=action_condition(action, ConditionKind.PRE),
            post_condition=action_condition(action, ConditionKind.POST),
            parameter_type_requirements=[
                requirement
                for requirement in ACTION_PARAMETER_TYPE_REQUIREMENTS
                if requirement.action is action
            ],
        )
        for action in actions_available_for(reasoner, annotation_type)
    ]


def doable_actions_for(
    annotation_type: type[SemanticAnnotation],
) -> ObjectActionAffordances:
    """
    Determine everything that can be done with one semantic annotation type.

    Combines the object's affordances
    (:class:`~llm_planner.env_reasoner.object_constraints.ObjectAffordanceReasoner`), the
    actions those affordances unlock
    (:func:`~llm_planner.env_reasoner.object_constraints.actions_available_for`) and each
    action's parameters and pre/post conditions
    (:mod:`llm_planner.env_reasoner.action_constraints`).

    :param annotation_type: A semantic annotation class, e.g. :class:`Milk`.
    :return: The object's affordances, mixins and doable actions.
    """
    reasoner = ObjectAffordanceReasoner()
    return ObjectActionAffordances(
        annotation_type=annotation_type,
        affordances=reasoner.affordances_for(annotation_type),
        mixins=_mixins_of(annotation_type),
        doable_actions=_doable_actions(annotation_type, reasoner),
    )


# %% doable actions for every object under a root

def object_action_affordances(
    root_annotation_type: type,
) -> dict[str, ObjectActionAffordances]:
    """
    Map every concrete object in a subclass graph to its doable actions.

    Walks the subclass graph of ``root_annotation_type``
    (:func:`~llm_planner.env_reasoner.object_graphs.semantic_annotation_graph`), then
    describes each leaf type with :func:`doable_actions_for`.

    :param root_annotation_type: A semantic annotation class or mixin whose concrete
        subclasses are the objects of interest, e.g. ``FoodContainer``.
    :return: One :class:`ObjectActionAffordances` per concrete subclass, keyed by class
        name.
    """
    graph = semantic_annotation_graph(root_annotation_type)
    return {
        leaf.__name__: doable_actions_for(leaf) for leaf in graph.concrete_annotations()
    }


def get_object_action_list(
    semantic_annotation_type: type[SemanticAnnotation],
) -> set[str]:
    """
    Names of the actions that can be performed on a semantic annotation type.

    :param semantic_annotation_type: A semantic annotation class, e.g. :class:`Milk`.
    :return: The class name of every applicable action.
    """
    return {
        doable.action.__name__
        for doable in doable_actions_for(semantic_annotation_type).doable_actions
    }


'''
#Usage Example:

if __name__ == "__main__":
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Furniture,
        Food,
    )

    for root in (Furniture, Food):
        for object_name, affordances in object_action_affordances(root).items():
            print(f"\n{object_name}")
            print(f"  affordances: {sorted(a.value for a in affordances.affordances)}")
            print(f"  also is: {[mixin.__name__ for mixin in affordances.mixins]}")
            for doable in affordances.doable_actions:
                print(
                    f"  {doable.action.__name__}"
                    f"  needs {doable.required_parameters}"
                    f"  for {sorted(a.value for a in doable.required_affordances)}"
                )
                for requirement in doable.parameter_type_requirements:
                    accepted = " or ".join(
                        sorted(
                            accepted_type.__name__
                            for accepted_type in requirement.accepted_types
                        )
                    )
                    print(f"    {requirement.parameter} must be {accepted}")
                if doable.pre_condition.is_declared:
                    print(f"    pre:  {doable.pre_condition.description}")
                if doable.post_condition.is_declared:
                    print(f"    post: {doable.post_condition.description}")
'''