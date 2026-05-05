from langchain_core.messages import SystemMessage

from agent.state import AgentState
from agent.tools import ALL_TOOLS


async def worker_node(state: AgentState, llm):
    plan = state["plan"]
    completed = state.get("completed_steps", [])

    if len(completed) >= len(plan):
        system_prompt = SystemMessage(
            content=(
                """You are a Technical Writer. Review the research gathered in the message history
                    and write a final markdown draft. Do not use any tools.
                    STRICT RULES:
                    1. Formatting: Output pure Markdown only. Do not include any conversational filler
                        (e.g., do not say 'Here is the draft'). Start immediately with a # header.
                    2. Citations: You MUST include a 'Sources' section at the bottom, listing the URLs
                        gathered during the research steps.
                    3. Completeness: Ensure all aspects of the original plan are represented in the text."""
            )
        )
        messages = [system_prompt] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"final_draft": response.content}

    current_step_index = len(completed)
    current_task = plan[current_step_index]

    worker_llm = llm.bind_tools(ALL_TOOLS)

    system_prompt = SystemMessage(
        content=(
            f"""You are a specialized Researcher. Your current task is: {current_task}
                STRICT RULES:
                1. Use the provided tools to gather factual data.
                2. When you have found the answer, write a technical summary of the findings.
                3. MANDATORY: You must explicitly list the URL(s) of the sources you used at the
                    bottom of your summary so the Writer node can cite them later.
                4. If you cannot find the information after 2 tool calls, stop searching. Write a
                    summary of what you did find and state that further data is unavailable."""
        )
    )

    messages = [system_prompt] + state["messages"]
    response = await worker_llm.ainvoke(messages)

    if not response.tool_calls:
        return {"messages": [response], "completed_steps": [current_task]}
    else:
        return {"messages": [response]}
