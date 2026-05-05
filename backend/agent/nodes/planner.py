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
            """You are a Senior Research Architect. Your job is to break a user's request "
                into exactly 3 to 5 distinct, highly-targeted research steps."

                CRITICAL RULES FOR THE STEPS:
                1. Each step MUST be actionable by a search engine. Do not write vague steps
                like 'Understand the topic'. Write exact search directives like 'Search for recent
                statistics on X'.
                2. Each step must be logically sequential. (e.g., Step 1 gathers definitions,
                Step 2 gathers current events, Step 3 gathers expert opinions).
                3. Do NOT include steps about 'writing', 'drafting', or 'formatting'. Your steps
                must strictly dictate data gathering only. The drafting phase happens automatically later.
                4. Keep each step under 15 words."""
        )
    )

    messages = [system_promt] + state["messages"]

    response = await planner_llm.ainvoke(messages)

    return {"plan": response.steps}
