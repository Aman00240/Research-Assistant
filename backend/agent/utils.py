import asyncio

from langchain_core.runnables import RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential

llm_semaphore = asyncio.Semaphore(2)


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    reraise=True,
)
async def safe_ainvoke(llm_instance, payload, config: RunnableConfig | None = None):
    async with llm_semaphore:
        if config:
            return await llm_instance.ainvoke(payload, config=config)
        return await llm_instance.ainvoke(payload)
