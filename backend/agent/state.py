import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    plan_justification: str
    completed_steps: Annotated[list[str], operator.add]
    research_notes: Annotated[list[str], operator.add]
    final_draft: str
