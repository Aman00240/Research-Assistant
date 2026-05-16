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
        2. TONE: Maintain a formal, analytical, and objective tone.
        3. STRUCTURAL VOLUME: Each sub-section MUST consist of at least 3-4 dense paragraphs. Avoid brief summaries; aim for exhaustive analysis.
        4. RAW DATA UTILIZATION: Every metric, unit, percentage, date, and proper noun found in the research
            notes MUST be integrated into the narrative. Never use vague adjectives (e.g., "fast")
            when a specific value (e.g., "10ms latency") is available in the data.
        5. COMPARATIVE SYNTHESIS: Identify the primary competing entities, theories, or technologies in
            the notes and dedicate a section to a 'Comparative Delta' analysis. Use data-heavy
            comparisons to highlight differences.
        6. COHESIVE INTEGRATION: Do not treat research steps as isolated entries. Use transitional logical
            flow to connect the findings from Step 1 to the implications found in the subsequent steps.
        7. TECHNICAL DEFINITION: For every major technical concept or acronym introduced,
            provide a brief, high-level context or definition based on the research data to ensure technical depth.

        REPORT STRUCTURE:
        # [Title of Research]

        ## Introduction
        (Briefly state the research goal and current industry context).

        ## Findings & Analysis
        (create 3-4 logical sub-headers based on the research topics, e.g., '### Energy Density Milestones'
        or '### Production Timelines'). Provide depth and specific numbers.

        ## Conclusion
        (Summarize the outlook and key takeaways).

        ## Sources
        (numbered list of all URLs used in the format: * [Source Title](URL))
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
        - CRITICAL: You are strictly forbidden from writing your final summary until you have
            successfully used `read_url_tool` at least once. Search snippets are not enough.
        - If you have not used `read_url_tool` yet, you ARE NOT ALLOWED to synthesize a summary.

    PHASE 2: SYNTHESIS (Only after gathering)
        - Write a deep-dive summary (at least 500 words) based ONLY on the high-quality extracts.
        - Include every statistic, date, name of expert, and technical value found in the extracts.
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
    - ANALYTICAL SUMMARY: Write 2-3 paragraphs of deep synthesis. Do not use
      bullet points for the main findings; write them as an integrated narrative.
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
                content="Convert the provided research context into the SourceVerifiedNote JSON schema. Do not write any conversational text."
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
