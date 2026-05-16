from langchain_core.messages import SystemMessage

from backend.agent.state import AgentState
from backend.agent.utils import safe_ainvoke


async def critic_node(state: AgentState, llm):
    plan = state.get("plan", [])
    draft = state.get("final_draft", "")
    notes = state.get("research_notes", [])

    system_prompt_content = f"""
    You are an objective Evaluation System. Your task is to verify if the provided Final Draft is a clean, accurate, and CONCISE summary based on the Raw Research Notes.

    INPUTS:
    Plan: {plan}
    Raw Research Notes: {notes}
    Final Draft: {draft}

    EVALUATION CRITERIA:
    1. NO CONVERSATIONAL FILLER:
        - Must start directly with a # Title. Mark as FAILED if you see "Here is your summary" or similar chatty intros.
    2. CONCISENESS:
        - The draft must be a short summary (around 200-300 words). Mark as FAILED if it looks like a massive scholarly report or contains unnecessary fluff.
    3. SOURCE INTEGRITY & HALLUCINATIONS:
        - Check the 'Sources' section. It MUST ONLY contain URLs provided in the Raw Research Notes.
        - Mark as FAILED if you see fake/generic links (like nature.com or example.com) that were not explicitly in the notes.
    4. CITATIONS:
        - Must use inline footnotes like [1] that correspond to the sources.

    OUTPUT RULES:
    - If ALL criteria are met perfectly, output exactly: APPROVED
    - If ANY criterion fails, output a precise directive for the Worker.
        Example: "FAILED. The draft is way too long. Condense it to 2 paragraphs and ensure the URLs match the notes exactly."
    """

    system_prompt = SystemMessage(content=system_prompt_content.strip())

    response = await safe_ainvoke(llm, [system_prompt])

    return {"messages": [response]}
