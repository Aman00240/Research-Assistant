from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from backend.agent.state import AgentState


class ResearchPlan(BaseModel):
    steps: list[str] = Field(
        description="A list of 3-5 technical steps to research the topic"
    )


async def planner_node(state: AgentState, llm):
    planner_llm = llm.with_structured_output(ResearchPlan)

    system_promt = SystemMessage(
        content=(
            """You are a Research Planner. Break down the user's request into 3-5
        distinct, actionable research steps. Each step must be something a
        search engine or a web reader can execute."""
        )
    )

    messages = [system_promt] + state["messages"]

    response = await planner_llm.ainvoke(messages)

    return {"plan": response.steps}
