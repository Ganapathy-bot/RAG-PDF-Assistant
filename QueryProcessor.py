from dataprocessor import ensure_indexed
from embedder import embed_User_query
from vectorstore import search_in_pinecone
from llm import query_llm_with_context


def process_user_query(query: str):
    ensure_indexed()

    # Embed the user's query to create a vector representation
    query_vector = embed_User_query(query)
    # Search the vector DB to find top matching chunks related to the usr's question
    matched_chunks = search_in_pinecone(query_vector)

    if not matched_chunks:
        print("No relevant information found for your query.")
        return

    context = "\n\n".join(matched_chunks)
    generated_response = query_llm_with_context(query, context)
    print(generated_response)

if __name__ == "__main__":
    user_query = "What is the work timing policy?"
    process_user_query(user_query)