from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

def verify_key():
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("OPENROUTER_API_KEY is not set. Set it in your .env or environment.")
        return

    # Basic validation of obviously wrong keys
    if openrouter_key.startswith("AIza"):
        print("The key in OPENROUTER_API_KEY looks like a Google API key (starts with 'AIza').")
        return

    openrouter_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    client = OpenAI(base_url=openrouter_base, api_key=openrouter_key)

    try:
        resp = client.embeddings.create(model="text-embedding-3-large", input="test")
        emb = resp.data[0].embedding
        print("OpenRouter key appears to work: received embedding of length", len(emb))
    except Exception as e:
        print("OpenRouter key test failed:", type(e).__name__, str(e))

if __name__ == "__main__":
    verify_key()
