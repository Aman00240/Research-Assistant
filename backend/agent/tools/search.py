from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from backend.config import settings

tavily = AsyncTavilyClient(api_key=settings.tavily_key.get_secret_value())


@tool
async def search_tool(query: str) -> str:
    """
    Search the internet for information on a given topic.
    Use this to find current events, facts, or technical details
    required to complete a research step.
    """

    print(f"Searching the web for {query}...")

    try:
        response = await tavily.search(query=query, search_depth="basic")
        results = response.get("results", [])

        if not results:
            return f"No results found for the exact query: '{query}'. Try a broader or simpler search term."

        content = "\n".join(
            [f"- {res['content']} (Source: {res['url']})" for res in results]
        )

        if not content.strip():
            return "Search completed, but the results contained no readable text."

        return content

    except Exception as e:
        return f"Search engine error: {str(e)}. Try a different query."
