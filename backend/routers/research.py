import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from langgraph.pregel.main import RunnableConfig
from pydantic import BaseModel

from backend.agent.graph import graph
from backend.agent.state import AgentState

router = APIRouter(prefix="/research", tags=["Research"])


class ResearchRequest(BaseModel):
    topic: str


@router.post("/")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [("user", f"Research this topic: {request.topic}")],
        "plan": [],
        "completed_steps": [],
        "final_draft": "",
    }
    background_tasks.add_task(graph.ainvoke, initial_state, config, version="v2")

    return {"thread_id": thread_id, "status": "started"}


@router.get("/status/{thread_id}")
async def get_status(thread_id: str):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    if not state.values:
        raise HTTPException(status_code=404, detail="Research job not found.")

    values = state.values
    plan = values.get("plan", [])
    completed = values.get("completed_steps", [])
    final_draft = values.get("final_draft", "")

    status = "completed" if final_draft else ("running" if plan else "planning")

    return {
        "thread_id": thread_id,
        "status": status,
        "plan": plan,
        "completed_steps": completed,
        "progress": f"{len(completed)}/{len(plan)}" if plan else "0/0",
        "final_draft": final_draft,
    }
