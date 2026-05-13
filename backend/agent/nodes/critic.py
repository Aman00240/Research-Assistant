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
    1. NO CONVERSATIONAL FILLER:
        - The draft MUST start directly with a # Title.
        - Mark as FAILED if there is any "Here is the report" or "Based on your request" chatter.
    2. SOURCE INTEGRITY: Check every URL. Mark as FAILED if it uses manifestos, extremist blogs,
        or unverified opinion pieces. Sources must be neutral and credible (.edu, .gov, or reputable news).
    3. FORMATTING & CITATIONS:
        - Must use inline footnotes like [1] for specific claims.
        - The 'Sources' section must use the format: [Article Title](URL).
        - NO CONVERSATIONAL FILLER. If the draft starts with "Here is your report" or
            includes any introductory chatter, mark as FAILED. It must start with a # header.
    4. COMPLETENESS:
          Does the draft address every single step in the Plan?
    5. QUANTITATIVE DEPTH:
        - Mark as FAILED if any sub-section in 'Findings & Analysis' contains fewer than 3 full paragraphs.
        - Mark as FAILED if the total length of the 'Findings & Analysis' section feels like a summary rather
            than an exhaustive technical analysis.
    6. DATA DENSITY:
        - Check for 'Raw Data Utilization'. Every specific number, percentage, and technical term (e.g., specific chemical compounds or voltage figures) found in the Raw Notes MUST appear in the Draft.
        - Mark as FAILED if the draft uses vague language like "significant improvement" instead of the exact figures found in the notes.

    OUTPUT RULES:
    - If ALL criteria are met perfectly, output exactly: APPROVED
    - If ANY criterion fails, output a precise directive for the Worker.
          Example: "FAILED. Section 2 is too short; expand it into 3 paragraphs using the
          technical data regarding [X] from the notes. Remove the introductory filler."
    """
    system_prompt = SystemMessage(content=system_prompt_content.strip())

    response = await llm.ainvoke([system_prompt])

    return {"messages": [response]}
