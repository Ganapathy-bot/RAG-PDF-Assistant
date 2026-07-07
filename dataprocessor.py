import json
from pathlib import Path

from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import VECTOR_STORE_PATH, store_in_pinecone

RESOURCES_DIR = Path("./resources")
INDEX_META_PATH = Path("./data/index_meta.json")


def get_pdf_fingerprints() -> dict[str, dict[str, float | int]]:
    fingerprints = {}
    for pdf_path in sorted(RESOURCES_DIR.glob("*.pdf")):
        stat = pdf_path.stat()
        fingerprints[pdf_path.name] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }
    return fingerprints


def load_index_meta() -> dict | None:
    if not INDEX_META_PATH.exists():
        return None
    with INDEX_META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def save_index_meta(fingerprints: dict[str, dict[str, float | int]]) -> None:
    INDEX_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_META_PATH.open("w", encoding="utf-8") as f:
        json.dump({"pdfs": fingerprints}, f, indent=2)


def needs_reindex() -> bool:
    fingerprints = get_pdf_fingerprints()
    if not fingerprints:
        raise FileNotFoundError(f"No PDF files found in {RESOURCES_DIR.resolve()}")

    if not Path(VECTOR_STORE_PATH).exists():
        return True

    meta = load_index_meta()
    if meta is None:
        return True

    return meta.get("pdfs") != fingerprints


def run() -> None:
    pdf_paths = sorted(RESOURCES_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {RESOURCES_DIR.resolve()}")

    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path.name}...")
        pages = read_pdf(str(pdf_path))
        chunks = chunk_pages(pages, chunk_size=900, chunk_overlap=150)
        all_chunks.extend(chunks)
        print(f"  Added {len(chunks)} chunks")

    print(f"Embedding {len(all_chunks)} total chunks...")
    embedded_chunks = embed_chunks(all_chunks)
    store_in_pinecone(all_chunks, embedded_chunks, namespace="")
    save_index_meta(get_pdf_fingerprints())


def ensure_indexed() -> None:
    if needs_reindex():
        print("PDFs changed or index missing. Re-indexing...")
        run()
    else:
        print("Index is up to date.")


if __name__ == "__main__":
    run()