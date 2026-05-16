from langchain_core.messages import SystemMessage
from langchain_core.messages.human import HumanMessage
from langgraph.graph.message import RemoveMessage

from backend.agent.state import AgentState, SourceVerifiedNote
from backend.agent.tools import ALL_TOOLS
from backend.agent.utils import safe_ainvoke


async def worker_node(state: AgentState, llm, writer_llm=None):
    plan = state["plan"]
    completed = state.get("completed_steps", [])
    notes = state.get("research_notes", [])

    if len(completed) >= len(plan):
        system_prompt_content = """
        You are an Executive Intelligence Analyst. Synthesize the provided research notes into a concise, data-driven Research Brief.

        STRICT CITATION RULES:
        1. Only use numerical citations [1], [2], etc., that correspond to the sources provided.
        2 At the end, you MUST provide a 'Sources' section.
        3. DO NOT invent citation numbers. If you have 4 sources, your highest citation must be [4].
        4. Every factual claim must be followed by a citation

        WRITING STANDARDS:
        1. TONE: Informative, clear, and concise.
        2. LENGTH: The entire brief should be around 200-300 words. Do NOT write a massive scholarly report.
        3. STRUCTURE:
            - # [Title of Summary]
            - 2 to 3 short paragraphs summarizing the key findings. Do not use complex sub-headers.
            - ## Sources
                (Numbered list of ONLY the URLs explicitly provided to you: * [Source Title](URL))
        """

        context_blocks = []
        all_sources = []

        for i, note in enumerate(notes):
            context_blocks.append(f"SECTION {i + 1} DATA:\n{note.summary_narrative}")
            all_sources.extend(note.sources)

        unique_sources = sorted(list(set([s for s in all_sources if s.strip()])))
        source_reference_block = "\n".join(
            [f"[{i + 1}] {url}" for i, url in enumerate(unique_sources)]
        )

        system_prompt = SystemMessage(content=system_prompt_content.strip())
        research_msg = HumanMessage(
            content=f"STRICT RESEARCH DATA TO USE:\n{chr(10).join(context_blocks)}\n\n"
            f"STRICT SOURCE LIST (YOU CAN ONLY USE THESE EXACT LINKS. DO NOT INVENT URLS):\n{source_reference_block}\n\n"
            f"INSTRUCTION:\n"
            f"1. Write the Research Brief.\n"
            f"2. Cite data using the corresponding bracketed number (e.g., [1]).\n"
            f"3. The 'Sources' section MUST exactly match the STRICT SOURCE LIST above."
        )

        messages_to_send = [system_prompt, research_msg] + state.get("messages", [])

        print("--- WRITER: Drafting report (Checking Critic feedback if any) ---")
        if writer_llm is not None:
            response = await safe_ainvoke(writer_llm, messages_to_send)

            return {"final_draft": response.content}

    current_step_index = len(completed)
    current_task = plan[current_step_index]

    worker_llm = llm.bind_tools(ALL_TOOLS)

    researcher_prompt = f"""
    You are a specialized Researcher. Your current task is: {current_task}

    WORKFLOW:
    PHASE 1: GATHERING (Mandatory)
        1. Use `search_tool` to find URLs.
        2. Use `read_url_tool` on the most promising URL to get raw facts.
        - CRITICAL: You are strictly forbidden from writing your final summary until you have successfully used `read_url_tool` at least once.

    PHASE 2: SYNTHESIS (Only after gathering)
        - Write a concise summary (maximum 150 words) based ONLY on the high-quality extracts.
        - Include the 2 or 3 most important statistics, dates, or names. Do not overcomplicate it.
        - Format as PLAIN TEXT. Do not attempt to call tools or use JSON in this phase.

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
    - CITATION DATA: At the bottom of your summary, list the source as:
        Source: Title of Article - URL
        (Do not use brackets or parentheses for the URL to avoid tool-calling errors).

    """

    system_prompt = SystemMessage(content=researcher_prompt.strip())
    task_human_message = HumanMessage(
        content=f"Please research this specific step: {current_task}"
    )

    messages = [system_prompt] + state["messages"] + [task_human_message]

    sanitized_messages = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            sanitized_messages.append(msg)
            continue

        if isinstance(msg.content, str) and not msg.content.strip():
            msg.content = "No text content found or tool returned empty."
        sanitized_messages.append(msg)

    print(f"--- WORKER: Asking LLM to research Step {current_step_index + 1} ---")
    response = await safe_ainvoke(worker_llm, sanitized_messages)

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        print(
            f"TOKENS USED | Input: {usage.get('input_tokens', 0)} | Output: {usage.get('output_tokens', 0)} | Total: {usage.get('total_tokens', 0)}"
        )

    if not response.tool_calls:
        print("--- WORKER: Step Complete Wiping memory and saving notes. ---")
        structured_llm = llm.with_structured_output(SourceVerifiedNote)

        research_context = "\n\n".join(
            [
                m.content
                for m in sanitized_messages
                if isinstance(m, HumanMessage) or hasattr(m, "tool_outputs")
            ]  # type: ignore
        )

        synthesis_prompt = [
            SystemMessage(
                content="Convert the provided research context into the SourceVerifiedNote JSON schema. Do not write any conversational text.\n\n"
                "CRITICAL SCHEMA RULE:\n"
                "The 'sources' array MUST be a flat list of strings. DO NOT wrap the strings in objects.\n"
                'CORRECT: ["https://linuxfoundation.eu/newsroom/ai-act-explainer"]\n'
                'WRONG: [{"type": "string", "value": "https://linuxfoundation.eu/newsroom/ai-act-explainer"}]'
            ),
            HumanMessage(
                content=f"Context to format:\n{research_context}\n\nFinal Output to extract from:\n{response.content}"
            ),
        ]
        structured_response = await safe_ainvoke(structured_llm, synthesis_prompt)
        print(
            f"Step {current_step_index + 1} summarized with {len(structured_response.sources)} sources."
        )

        messages_to_remove = state["messages"][1:]
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]

        return {
            "messages": delete_messages,
            "completed_steps": [current_task],
            "research_notes": [structured_response],
        }
    else:
        print(f"--- WORKER: LLM requested tools: {response.tool_calls} ---")
        return {"messages": [response]}
