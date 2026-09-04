import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# Dense Retrieval (SciBERT / SPECTER / SciNCL)

def run_dense_model(
    model_name,
    corpus_ids,
    corpus_texts,
    queries,
    top_k=100
):
    print(f"\n=== Running dense model: {model_name} ===")

    model = SentenceTransformer(model_name)

    # 1) Encode the document corpus
    corpus_emb = model.encode(
        corpus_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    ).astype("float32")

    dim = corpus_emb.shape[1]

    # Create a FAISS index.
    # With normalized embeddings, inner product
    # is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_emb)

    # 2) Retrieve the top-k documents for each query
    results = {}

    for qid, query in tqdm(
        queries.items(),
        desc=f"Searching {model_name}"
    ):
        query_emb = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        query_emb = query_emb.reshape(1, -1)

        scores, indices = index.search(
            query_emb,
            top_k
        )

        scores = scores[0]
        indices = indices[0]

        ranked = {
            corpus_ids[i]: float(scores[j])
            for j, i in enumerate(indices)
        }

        results[qid] = ranked

    return results


DENSE_MODELS = {
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "SPECTER": "allenai/specter",
    "SciNCL": "malteos/scincl",
}
