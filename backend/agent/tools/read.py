from langchain_core.tools import tool
from tavily.asynclient import aiohttp


@tool
async def read_url_tool(url: str) -> str:
    """
    Read and extract the text content from a specific URL.
    Use this when you have a specific link from a search result
    and need to read the full webpage detail.
    """

    print(f"Reading Webpage: {url}...")

    jina_url = f"https://r.jina.ai/{url}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(jina_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return content[:8000]

                else:
                    return (
                        f"Error: Could not read webpage. Status code {response.status}"
                    )

    except Exception as e:
        return f"Error reading url: {str(e)}"
