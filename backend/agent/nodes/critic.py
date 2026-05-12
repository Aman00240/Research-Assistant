from langchain_core.messages import SystemMessage

from backend.agent.state import AgentState


async def critic_node(state: AgentState, llm):
    plan = state.get("plan", [])
    draft = state.get("final_draft", "")
    notes = state.get("research_notes", [])

    system_prompt_content = f"""
    You are an objective Evaluation System. Your task is to verify if the provided Final Draft
    meets all strict professional standards based on the Research Plan and Raw Research Notes.

    INPUTS:
    Plan: {plan}
    Raw Research Notes: {notes}
    Final Draft: {draft}

    EVALUATION CRITERIA:
    1. ACCURACY & DEPTH: Does the draft reflect the facts in the notes? mark as FAILED if it
        is a shallow summary. It must include specific statistics, technical
        nuances, and an academic, analytical tone.
    2. SOURCE INTEGRITY: Check every URL. Mark as FAILED if it uses manifestos, extremist blogs,
        or unverified opinion pieces. Sources must be neutral and credible (.edu, .gov, or reputable news).
    3. FORMATTING & CITATIONS:
        - Must use inline footnotes like [1] for specific claims.
        - The 'Sources' section must use the format: [Article Title](URL).
        - NO CONVERSATIONAL FILLER. If the draft starts with "Here is your report" or
            includes any introductory chatter, mark as FAILED. It must start with a # header.
    4. COMPLETENESS: Does the draft address every single step in the Plan?

    OUTPUT RULES:
    - If ALL criteria are met perfectly, output exactly: APPROVED
    - If ANY criterion fails, output a precise 1-2 sentence directive explaining exactly
        what the Worker must change (e.g., "Failed. Remove conversational filler and add
        statistics from Step 2.").
    """
    system_prompt = SystemMessage(content=system_prompt_content.strip())

    response = await llm.ainvoke([system_prompt])

    return {"messages": [response]}
