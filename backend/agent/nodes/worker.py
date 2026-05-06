from langchain_core.messages import SystemMessage
from langchain_core.messages.human import HumanMessage
from langgraph.graph.message import RemoveMessage

from backend.agent.state import AgentState
from backend.agent.tools import ALL_TOOLS


async def worker_node(state: AgentState, llm):
    plan = state["plan"]
    completed = state.get("completed_steps", [])
    notes = state.get("research_notes", [])

    if len(completed) >= len(plan):
        system_prompt_content = """
            You are a Technical Writer. Review the research gathered in the message history
            and write a final markdown draft. Do not use any tools.
            STRICT RULES:
            1. Structure: Your response MUST have: 'Introduction', 'Key Findings', and 'Conclusion'.
            2. Formatting: Output pure Markdown only. Do not include any conversational filler
                (e.g., do not say 'Here is the draft'). Start immediately with a # header.
            3. Citations: You MUST include a 'Sources' section at the bottom, listing the URLs
                gathered during the research steps.
            4. Completeness: Ensure all aspects of the original plan are represented in the text.
            """
        system_prompt = SystemMessage(content=system_prompt_content.strip())
        compiled_research = "\n\n".join(notes)
        human_msg = HumanMessage(
            content=f"Here is the gathered research:{compiled_research}"
        )

        response = await llm.ainvoke([system_prompt, human_msg])
        return {"final_draft": response.content}

    current_step_index = len(completed)
    current_task = plan[current_step_index]

    worker_llm = llm.bind_tools(ALL_TOOLS)

    researcher_prompt = f"""
            You are a specialized Researcher. Your current task is: {current_task}

            STRICT RULES:
            1. Detailed Summary: Write a 2-3 paragraph summary of your findings.
            2. Source Tracking: List the URL(s) at the bottom of your summary.
            3. Be concise: Extract only the most important factual data.
            """

    system_prompt = SystemMessage(content=researcher_prompt.strip())
    messages = [system_prompt] + state["messages"]
    print(f"--- WORKER: Asking LLM to research Step {current_step_index + 1} ---")
    response = await worker_llm.ainvoke(messages)

    if not response.tool_calls:
        print("--- WORKER: Step Complete Wiping memory and saving notes. ---")
        summary = response.content

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]

        return {
            "messages": delete_messages,
            "completed_steps": [current_task],
            "research_notes": [summary],
        }
    else:
        print(f"--- WORKER: LLM requested tools: {response.tool_calls} ---")
        return {"messages": [response]}
