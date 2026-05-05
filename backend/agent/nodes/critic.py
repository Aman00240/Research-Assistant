from langchain_core.messages import SystemMessage

from backend.agent.state import AgentState


async def critic_node(state: AgentState, llm):
    plan = state.get("plan", [])
    draft = state.get("final_draft", "")

    system_prompt = SystemMessage(
        content=(
            f"""You are an objective Evaluation System. Your task is to verify if the provided Final Draft
            meets all the strict requirements based on the Research plan that are denoted by backticks(```).

            INPUTS:
            Plan: ```{plan}```
            Final Draft: ```{draft}```

            EVALUATION CRITERIA:
            1. COMPLETENESS: Does the draft explicitly address every step listed in the Plan?
            2. CITATIONS: Does the draft include inline citations or a source list derived from a web search?
            3. FORMAT: Is the draft formatted in clean Markdown without conversational introductory text?

            OUTPUT RULES:
            - if ALL the criteria are met prefectly, output exactly the word: APPROVED
            - If ANY criterion fails, do NOT output 'APPROVED'. Instead, output a precise, 1-2 sentence directive
                explaining exactly what the Worker node must add or change to pass the evaluation.
                Specify the exact missing step or missing formatting.
            """
        )
    )

    response = await llm.ainvoke([system_prompt])

    return {"messages": [response]}
