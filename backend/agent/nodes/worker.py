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
        You are a Professional Technical Writer. Your goal is to synthesize the
        provided research notes into a cohesive, scholarly report.

        WRITING STANDARDS:
        1. TONE: Maintain a formal, analytical, and objective tone. Avoid hyperbolic
           language. Synthesize the findings into a narrative rather than listing them.
        2. SYNTHESIS: Do not just repeat the research notes. Connect the dots
           between different steps of the research to provide a holistic view.
        3. CITATIONS:
           - Use inline footnotes like [1] when referencing a specific fact.
           - At the end, provide a 'Sources' section with a numbered list.
           - Format: [Source Title](URL).

        STRUCTURE:
        # [Title of Research]
        ## Introduction
        ## Deep Analysis (Group findings into logical themes)
        ## Conclusion
        ## Sources
        """
        system_prompt = SystemMessage(content=system_prompt_content.strip())
        compiled_research = "\n\n".join(notes)
        research_msg = HumanMessage(
            content=f"Here is the gathered research:\n{compiled_research}"
        )

        messages_to_send = [system_prompt, research_msg] + state.get("messages", [])

        print("--- WRITER: Drafting report (Checking Critic feedback if any) ---")
        response = await llm.ainvoke(messages_to_send)

        return {"final_draft": response.content}

    current_step_index = len(completed)
    current_task = plan[current_step_index]

    worker_llm = llm.bind_tools(ALL_TOOLS)

    researcher_prompt = f"""
    You are a specialized Researcher. Your current task is: {current_task}

    WORKFLOW:
    1. SEARCH: Use `search_tool` to find the most relevant URLs.
    2. SMART READ: For the 1-2 most promising URLs, you MUST use `read_url_tool`.
        Pass the URL and the 'current_task' exactly as written above to extract
        the raw factual chunks.
    3. SYNTHESIZE: Write your summary based ONLY on the high-quality extracts
        returned by the tools.

    SOURCE INTEGRITY PROTOCOL:
    1. CREDIBILITY CHECK: Evaluate every source before summarizing. Prioritize
       academic institutions (.edu), government agencies (.gov), and reputable
       news/journalistic outlets.
    2. EXCLUDE RADICAL CONTENT: Do not include manifestos, extremist blogs, or
       unverified opinion pieces unless the task explicitly asks for "controversial viewpoints."
    3. FACT EXTRACTION: Prioritize specific statistics, dates, names of experts,
       and complex causal relationships over general descriptions.

    STRICT OUTPUT RULES:
    - You must provide the summary as PLAIN TEXT.
    - DO NOT attempt to call any tools or use JSON in your summary.
    - DO NOT include any trailing punctuation like closing brackets or JSON syntax at the end of your message.
    - ANALYTICAL SUMMARY: Write 2-3 paragraphs of deep synthesis. Do not use
      bullet points for the main findings; write them as an integrated narrative.
    - CITATION DATA: At the bottom of your summary, list the source as:
        Source: Title of Article - URL
        (Do not use brackets or parentheses for the URL to avoid tool-calling errors).

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
