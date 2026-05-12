from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from backend.agent.state import AgentState


class ResearchPlan(BaseModel):
    steps: list[str] = Field(
        description="A list of 3-5 technical steps to research the topic"
    )


async def planner_node(state: AgentState, llm):
    planner_llm = llm.with_structured_output(ResearchPlan)

    system_promt_content = """
    You are a Senior Research Architect. Your job is to break a user's request
    into exactly 3 to 5 distinct, highly-targeted research steps.

    STRATEGIC DIRECTIVES:
    1. FOCUS ON DEPTH: Aim for technical, scholarly, or statistical data.
       Do not just ask for "basics." Ask for "Current peer-reviewed perspectives on X" or
       "Quantifiable economic data regarding Y."
    2. LOGICAL PROGRESSION: Start with foundational definitions, move to
       conflicting viewpoints or technical specifics, and end with future trends or
       expert syntheses.
    3. LOGICAL CONSTRAINTS: Steps must be actionable by a search engine.
       Keep each step under 15 words.
    4. NO FORMATTING STEPS: Do not include steps about writing or drafting.
    """
    system_prompt = SystemMessage(content=system_promt_content.strip())

    messages = [system_prompt] + state["messages"]

    response = await planner_llm.ainvoke(messages)

    return {"plan": response.steps}
