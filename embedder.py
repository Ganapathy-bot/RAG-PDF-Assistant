import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-large"

client = OpenAI(
    base_url=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

_EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "RAG HR Assistant",
}


def _embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        extra_headers=_EXTRA_HEADERS,
    )
    return response.data[0].embedding


def embed_chunks(chunks):
    return [_embed_text(chunk) for chunk in chunks]


def embed_User_query(query: str) -> list[float]:
    return _embed_text(query)
