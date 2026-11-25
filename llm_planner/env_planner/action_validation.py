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

import inspect
from dataclasses import dataclass, field

from typing_extensions import Iterable, List, Optional, Tuple, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
    TruthValueOperator,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import evaluate_condition
from llm_planner.env_reasoner.action_constraints import ConditionKind

CHECKABLE_ACTIONS: Tuple[Type[ActionDescription], ...] = (
    OpenAction,
    CloseAction,
    PickUpAction,
    PlaceAction,
    MixingAction,
    CuttingAction,
    PouringAction,
)


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
class ConditionStatement:
    """
    One statement of an action's condition, whether it holds, and the statements it
    combines.

    An ``and_``/``or_``/``not_`` of other statements carries them as its
    :attr:`operands`, so a condition that failed can be read down to the predicate that
    refused it.
    """

    name: str
    """
    The name of what the statement checks - the predicate, or the operator combining
    other statements.
    """

    description: str
    """
    What the statement checks, read from the opening paragraph of the documentation of
    the predicate it stands for. Empty for an operator and for an undocumented
    predicate.
    """

    satisfied: bool
    """
    Whether this statement holds.
    """

    operands: List[ConditionStatement] = field(default_factory=list)
    """
    The statements this one combines, in the order the condition names them. Empty for
    a predicate.
    """

    expression: Optional[SymbolicExpression] = field(
        default=None, compare=False, repr=False
    )
    """
    The condition expression this statement was read from, which tells a statement
    reached through two branches of a condition from two statements that read alike.
    """

    @classmethod
    def from_expression(
        cls, expression: SymbolicExpression, satisfied: bool
    ) -> ConditionStatement:
        """
        Read a condition expression as a statement and the statements it combines.

        :param expression: The condition expression to read.
        :param satisfied: Whether that expression holds, already evaluated.
        :return: The statement tree, each operand evaluated on its own.
        """
        subject = _subject_of(expression)
        return cls(
            name=expression._name_ if subject is None else subject.__name__,
            description=_description_of(subject),
            satisfied=satisfied,
            operands=[
                cls.from_expression(operand, evaluate_condition(operand))
                for operand in _operands_of(expression)
            ],
            expression=expression,
        )

    @property
    def innermost_failures(self) -> List[ConditionStatement]:
        """
        :return: The failed statements that no failed operand accounts for - the
            statements to read as the reason, rather than the operators above them.
            Empty where this statement holds.
        """
        if self.satisfied:
            return []
        failed_operands = _without_repeats(
            failure
            for operand in self.operands
            for failure in operand.innermost_failures
        )
        return failed_operands or [self]

    @property
    def summary(self) -> str:
        """
        :return: The statement's name, followed by what it checks where its subject is
            documented.
        """
        if not self.description:
            return self.name
        return f"{self.name}: {self.description}"

    def report(self, depth: int = 0) -> str:
        """
        :param depth: How far the statement is indented, in levels of four spaces.
        :return: The statement and everything below it, one line each, saying whether
            it holds.
        """
        heading = (
            f"{'    ' * depth}{self.name}: {'holds' if self.satisfied else 'fails'}"
        )
        return "\n".join(
            [heading, *(operand.report(depth + 1) for operand in self.operands)]
        )


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

    statement: Optional[ConditionStatement] = None
    """
    The condition read as a statement tree, each of its statements evaluated on its
    own. None when the condition holds, and when it is a plain ``True``/``False``
    rather than an expression - several of :data:`CHECKABLE_ACTIONS` declare no
    condition of their own for one of the two kinds and inherit the unconditional
    default.
    """

    @property
    def false_statements(self) -> List[ConditionStatement]:
        """
        :return: The statements that made the condition false, each the innermost one
            its branch of the condition failed on.
        """
        if self.statement is None:
            return []
        return self.statement.innermost_failures

    @property
    def false_statement_names(self) -> List[str]:
        """
        :return: The name of every statement that made the condition false.
        """
        return [statement.name for statement in self.false_statements]


def _without_repeats(
    statements: Iterable[ConditionStatement],
) -> List[ConditionStatement]:
    """
    The statements in the order they were found, each expression reported once.

    :param statements: The statements to read.
    :return: Those statements, leaving out a statement read from an expression an
        earlier one was already read from.
    """
    first_of_each: dict[int, ConditionStatement] = {}
    for statement in statements:
        first_of_each.setdefault(_identity_of(statement.expression), statement)
    return list(first_of_each.values())


def _identity_of(expression: Optional[SymbolicExpression]) -> int:
    """
    What a statement is told apart by: the predicate a condition names, rather than
    the node it was reached through, since naming one predicate twice wraps it in a
    :class:`~krrood.entity_query_language.core.variable.Literal` of its own each time.

    :param expression: The expression the statement was read from.
    :return: The identity of what it checks.
    """
    if isinstance(expression, Literal):
        return id(expression._value_)
    return id(expression)


def _subject_of(expression: SymbolicExpression) -> Optional[type]:
    """
    The type a condition expression checks with, e.g. the predicate class an expression
    stands for.

    :param expression: The expression to read.
    :return: That type, or None for an operator combining other statements.
    """
    if isinstance(expression, Literal):
        return type(expression._value_)
    if isinstance(expression, Selectable):
        return expression._type_
    return None


def _description_of(subject: Optional[type]) -> str:
    """
    What a statement's subject is documented to check.

    :param subject: The type the statement checks with, if it has one.
    :return: The opening paragraph of that type's documentation, as one line, or an
        empty string where there is no documented subject.
    """
    if subject is None:
        return ""
    documentation = inspect.getdoc(subject)
    if documentation is None:
        return ""
    return " ".join(documentation.split("\n\n")[0].split())


def _operands_of(expression: SymbolicExpression) -> List[SymbolicExpression]:
    """
    The statements an expression combines.

    :param expression: The expression to read.
    :return: Its operands, or an empty list where it checks something itself rather
        than combining other statements.
    """
    if not isinstance(expression, TruthValueOperator):
        return []
    return [child for child in expression._children_ if child is not expression]


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
    statement = (
        None
        if satisfied or isinstance(condition, bool)
        else ConditionStatement.from_expression(condition, satisfied)
    )
    return ConditionCheck(
        action=type(action), kind=kind, satisfied=satisfied, statement=statement
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
