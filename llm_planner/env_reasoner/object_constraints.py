from __future__ import annotations
from semantic_digital_twin.world import World

from dataclasses import dataclass
from enum import StrEnum

from semantic_digital_twin.semantic_annotations.mixins import HasHandle
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bread,
    Cabinet,
    CounterTop,
    Cuttlery,
    Drawer,
    Food,
    Fruit,
    Furniture,
    Milk,
    Oven,
    Shelf,
    Sink,
    TomatoSoup,
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


if __name__ == "__main__":
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Apple,
        Door,
        Drawer,
        TunaCan,
        Spoon,
    )

    reasoner = ObjectAffordanceReasoner()
    for demo_type in [Apple, Bread, Milk, TunaCan, Drawer, Door,Spoon]:
        demo_affordances = reasoner.affordances_for(demo_type)
        print(f"{demo_type.__name__}: {sorted(a.value for a in demo_affordances)}")


def objects_in_environment(world: World) -> set[str]:
    """
    Collect the distinct semantic annotation types present in ``world``.

    :param world: The world whose semantic annotations are inspected.
    :return: The upper-case name of every distinct semantic annotation type
        found in :attr:`World.semantic_annotations`, e.g. ``{"DRAWER", "CABINET"}``.
    """
    return {
        type(annotation).__name__.upper() for annotation in world.semantic_annotations
    }
