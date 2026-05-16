import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    thinking: str = Field(
        description="A brief technical analysis of the topic's complexity and why these specific steps are necessary."
    )
    steps: list[str] = Field(
        description="A list of 3-5 technical steps to research the topic. Use 5 steps for complex legal/technical queries."
    )


class SourceVerifiedNote(BaseModel):
    summary_narrative: str = Field(description="The 500-word technical deep-dive.")
    sources: list[str] = Field(description="The exact, unmodified URLs used.")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]
    plan_justification: str
    completed_steps: Annotated[list[str], operator.add]
    research_notes: Annotated[list[SourceVerifiedNote], operator.add]
    final_draft: str
