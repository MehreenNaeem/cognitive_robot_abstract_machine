from semantic_digital_twin.semantic_annotations.semantic_annotations import Cabinet, Door, Drawer, Fridge, Handle, Wardrobe
from krrood.ripple_down_rules.helpers import get_an_updated_case_copy, update_case_and_conclusions_with_rule_output
from krrood.ripple_down_rules.utils import copy_case, make_set
from typing_extensions import Optional, Set, Union
from krrood.ripple_down_rules.datastructures.case import Case, create_case
from semantic_digital_twin.world import World
from .world_semantic_annotations_mcrdr_defs import *


attribute_name = 'semantic_annotations'
conclusion_type = (Fridge, Handle, Wardrobe, set, list, Drawer, Door, Cabinet,)
mutually_exclusive = False
name = 'semantic_annotations'
case_type = World
case_name = 'World'


def classify(case: World, **kwargs) -> Set[Union[Fridge, Handle, Wardrobe, Drawer, Door, Cabinet]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_90574698325129464513441443063592862114(case):
        update_case_and_conclusions_with_rule_output(case, conclusions, conclusion_90574698325129464513441443063592862114(case),attribute_name, conclusion_type, mutually_exclusive)

    if conditions_331345798360792447350644865254855982739(case):
        update_case_and_conclusions_with_rule_output(case, conclusions, conclusion_331345798360792447350644865254855982739(case),attribute_name, conclusion_type, mutually_exclusive)

    if conditions_35528769484583703815352905256802298589(case):
        update_case_and_conclusions_with_rule_output(case, conclusions, conclusion_35528769484583703815352905256802298589(case),attribute_name, conclusion_type, mutually_exclusive)

    if conditions_59112619694893607910753808758642808601(case):
        update_case_and_conclusions_with_rule_output(case, conclusions, conclusion_59112619694893607910753808758642808601(case),attribute_name, conclusion_type, mutually_exclusive)

    if conditions_10840634078579061471470540436169882059(case):
        update_case_and_conclusions_with_rule_output(case, conclusions, conclusion_10840634078579061471470540436169882059(case),attribute_name, conclusion_type, mutually_exclusive)

    if conditions_99828403881738252604561834486102211484(case):
        update_case_and_conclusions_with_rule_output(case, conclusions, conclusion_99828403881738252604561834486102211484(case),attribute_name, conclusion_type, mutually_exclusive)

    return conclusions
