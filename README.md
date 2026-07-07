# RAG PDF Assistant

A Python RAG (Retrieval-Augmented Generation) application that answers questions from PDF documents. It reads files from `resources/`, builds a local vector index, retrieves relevant passages, and uses an LLM to generate grounded answers.

## Features

- **PDF ingestion** — Extract text from one or more PDFs in `resources/`
- **Local vector storage** — No Pinecone or cloud database required
- **Auto re-indexing** — Rebuilds the index automatically when PDFs are added or changed
- **OpenRouter support** — Uses OpenRouter for embeddings and chat completions
- **OpenAI fallback** — Optional direct OpenAI key for LLM responses

## How It Works

```
PDFs in resources/
        ↓
  dataprocessor.py  →  chunk text  →  embed  →  data/vectors.npz
        ↓
  QueryProcessor.py →  embed query  →  search  →  LLM answer
```

1. PDFs are split into overlapping text chunks
2. Each chunk is embedded with `text-embedding-3-large`
3. Vectors are saved locally to `data/vectors.npz`
4. On query, the most relevant chunks are retrieved and sent to the LLM as context

## Requirements

- Python 3.10+
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

```bash
git clone https://github.com/Ganapathy-bot/RAG-PDF-Assistant.git
cd RAG-PDF-Assistant
pip install -r requirements.txt
```

Create your environment file:

```bash
copy .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
```

Verify the API key:

```bash
python verify_openrouter_key.py
```

## Usage

### Ask a question

```bash
python QueryProcessor.py
```

Or from Python:

```python
from QueryProcessor import process_user_query

process_user_query("What is biostatistics?")
```

`QueryProcessor` checks whether PDFs in `resources/` have changed. If they have, it re-indexes automatically before answering.

### Manually rebuild the index

```bash
python dataprocessor.py
```

### Add new documents

1. Place PDF files in `resources/`
2. Run `python QueryProcessor.py` — indexing happens automatically if files changed

## Project Structure

```
RAG-PDF-Assistant/
├── resources/              # PDF documents to index
├── data/                   # Local vector store (auto-generated, gitignored)
├── dataprocessor.py        # Indexing pipeline
├── QueryProcessor.py       # Query pipeline
├── pdfreader.py            # PDF text extraction
├── chunker.py              # Text chunking
├── embedder.py             # OpenRouter embeddings
├── vectorstore.py          # Local vector search
├── llm.py                  # LLM response generation
├── verify_openrouter_key.py
├── requirements.txt
└── .env.example
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key for embeddings and LLM |
| `OPENROUTER_API_BASE` | No | OpenRouter endpoint (default: `https://openrouter.ai/api/v1`) |
| `OPENAI_API_KEY` | No | Use OpenAI directly for LLM instead of OpenRouter |
| `VECTOR_STORE_PATH` | No | Path to vector store file (default: `./data/vectors.npz`) |

## Notes

- `data/` and `.env` are gitignored — never commit API keys or generated indexes
- Copy `.env.example` to `.env` before running the app
- The default query in `QueryProcessor.py` can be changed in the `__main__` block

## License

This project is provided as-is for educational and research use.