import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .rag_index import load_faq_data


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "faqs.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "faqs_metadata.npy")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.astype("float32")


def main():
    items = load_faq_data(DATA_PATH)
    print(f"Loaded {len(items)} FAQ items from faqs.json")

    if not items:
        raise RuntimeError("No data found to build index (faqs.json is empty).")

    texts = [f"{item['question']}\n{item['answer']}" for item in items]

    print(f"Embedding {len(texts)} items...")
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    meta = np.array(
        [
            {
                "question": item["question"],
                "answer": item["answer"],
            }
            for item in items
        ],
        dtype=object,
    )
    np.save(META_PATH, meta)

    print(f"Index built and saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()