from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.pregel.main import partial

from backend.agent.nodes.critic import critic_node
from backend.agent.nodes.planner import planner_node
from backend.agent.nodes.worker import worker_node
from backend.agent.state import AgentState
from backend.agent.tools import ALL_TOOLS
from backend.config import settings

researcher_llm = ChatGroq(
    api_key=settings.groq_key,
    model=settings.llama_scout,
    temperature=0,
    max_tokens=1500,
)

writer_llm = ChatGroq(
    api_key=settings.groq_key,
    model=settings.llama_versatile,
    temperature=0,
    max_tokens=7000,
)


def route_worker(state: AgentState):
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    completed = state.get("completed_steps", [])
    plan = state.get("plan", [])

    if len(completed) >= len(plan) and state.get("final_draft"):
        return "critic"

    return "worker"


def route_critic(state: AgentState):
    messages = state.get("messages", [])
    last_message = messages[-1]

    if "APPROVED" in last_message.content:
        return END

    return "worker"


builder = StateGraph(AgentState)
builder.add_node("planner", partial(planner_node, llm=researcher_llm))
builder.add_node(
    "worker", partial(worker_node, llm=researcher_llm, writer_llm=writer_llm)
)
builder.add_node("critic", partial(critic_node, llm=researcher_llm))
builder.add_node("tools", ToolNode(ALL_TOOLS))

builder.add_edge(START, "planner")
builder.add_edge("planner", "worker")
builder.add_edge("tools", "worker")


builder.add_conditional_edges(
    "worker", route_worker, {"tools": "tools", "critic": "critic", "worker": "worker"}
)

builder.add_conditional_edges("critic", route_critic, {END: END, "worker": "worker"})

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
