from langchain_core.messages import SystemMessage

from agent.state import AgentState
from agent.tools import ALL_TOOLS


async def worder_node(state: AgentState, llm):
    plan = state["plan"]
    completed = state.get("completed_steps", [])

    current_step_index = len(completed)
    current_task = plan[current_step_index]

    worker_llm = llm.bind_tools(ALL_TOOLS)

    system_prompt = SystemMessage(
        content=(
            f"""You are a specialized Researcher. Your current task is: {current_task}\n
            Use the provided tools to gather information.
            Once you have sufficient information to satisfy this specific research step,
            summarize your findings clearly."""
        )
    )

    messages = [system_prompt] + state["messages"]
    response = await worker_llm.ainvoke(messages)

    return {"messages": [response]}
