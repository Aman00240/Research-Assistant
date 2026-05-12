from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily.asynclient import aiohttp

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@tool
async def read_url_tool(url: str, current_task: str) -> str:
    """
    Read a URL and extract only the relevant parts for the current task.
    Use this when you have a specific link and need to find specific facts.
    """
    print(f"Reading & Analyzing Webpage: {url}...")

    jina_url = f"https://r.jina.ai/{url}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(jina_url) as response:
                if response.status != 200:
                    return f"Error: Could not read webpage, Status {response.status}"

                full_content = await response.text()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=100
                )
                chunks = text_splitter.split_text(full_content)

                vectorstore = FAISS.from_texts(chunks, embeddings)

                relevant_chunks = vectorstore.similarity_search(current_task, k=5)

                summary_context = "\n\n".join(
                    [chunk.page_content for chunk in relevant_chunks]
                )

                return f"--- RELEVANT DATA FROM {url} ---\n\n{summary_context}"

    except Exception as e:
        return f"Error reading url: {str(e)}"
