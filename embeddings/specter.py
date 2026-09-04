import numpy as np
import faiss

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.preprocessing.text import build_doc_text_from_beir


def l2_normalize(
    x: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """Apply L2 normalization to embedding vectors."""

    return x / (
        np.linalg.norm(
            x,
            axis=1,
            keepdims=True
        ) + eps
    )


def build_specter_faiss(
    corpus,
    model_name="allenai/specter",
    batch_size=64
):
    """Build a SPECTER FAISS index for the document corpus."""

    model = SentenceTransformer(model_name)

    doc_ids = list(corpus.keys())

    doc_texts = [
        build_doc_text_from_beir(corpus[doc_id])
        for doc_id in doc_ids
    ]

    embeddings = []

    for i in tqdm(
        range(0, len(doc_texts), batch_size),
        desc="SPECTER doc embeddings"
    ):
        batch = doc_texts[i:i + batch_size]

        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings.append(batch_embeddings)

    doc_embeddings = np.vstack(
        embeddings
    ).astype("float32")

    doc_embeddings = l2_normalize(
        doc_embeddings
    )

    dim = doc_embeddings.shape[1]

    # Cosine similarity via inner product
    # after L2 normalization
    index = faiss.IndexFlatIP(dim)

    index.add(doc_embeddings)

    return model, index, doc_ids, doc_texts
