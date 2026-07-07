from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI or OpenRouter client
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if openrouter_key:
    openrouter_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    # Basic validation: detect common wrong keys (e.g., Google API keys starting with 'AIza')
    if openrouter_key.startswith("AIza"):
        raise ValueError(
            "The value in OPENROUTER_API_KEY looks like a Google API key (starts with 'AIza').\n"
            "Please provide your OpenRouter API key or set OPENAI_API_KEY instead."
        )

    client = OpenAI(base_url=openrouter_base, api_key=openrouter_key)
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_llm_with_context(query: str, context: str):
    system_content = """You are a helpful assistant for answering user queries based on provided context. 
    use the context to provide accurate and relevant answers. Do not make assumptions beyond the context provided.
    If the context does not contain enough information to answer the query, 
    let the user know that you cannot provide an answer based on the given context.
    """
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
        ],
        temperature=0.4,
        extra_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG HR Assistant",
        },
    )
    return response.choices[0].message.content
