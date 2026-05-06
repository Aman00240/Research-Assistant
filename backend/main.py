from fastapi import FastAPI

from backend.routers import research

app = FastAPI(title="Agentic Research API")

app.include_router(research.router)


@app.get("/")
async def root():
    return {"message": "Agentic Research API is online"}
