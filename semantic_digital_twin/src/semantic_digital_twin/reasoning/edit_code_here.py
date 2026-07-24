from collections import UserDict
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, ClassVar, Dict, List, Optional, Type, Union

from krrood.entity_query_language.factories import variable, entity, inference, contains
from krrood.ripple_down_rules.rdr import GeneralRDR
from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.reasoning.reasoner import CaseRDRs, CaseReasoner, ReasoningResult
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.world_description.connections import PrismaticConnection


def world_semantic_annotations_of_type_drawer(case: World) -> List[Drawer]:
    """Get possible value(s) for World.semantic_annotations  of type Drawer."""
    prismatic_connection = variable(PrismaticConnection, case.connections)
    return (entity(inference(Drawer)(root=prismatic_connection.child))
            .where(contains(prismatic_connection.child.name.name.lower(), "drawer"),
                   InsideOf(prismatic_connection.child, prismatic_connection.parent).compute_containment_ratio() > 0.7,
                   )
            .tolist()
            )