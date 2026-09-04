import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def simple_tokenize(text):
    """Tokenize text using lowercase whitespace splitting."""
    return text.lower().split()


def run_bm25(
    corpus_ids,
    corpus_texts,
    queries,
    top_k=100
):
    """Run BM25 retrieval for all queries."""

    print("\n=== Running BM25 baseline ===")

    # Tokenize the document corpus
    tokenized_corpus = [
        simple_tokenize(text)
        for text in corpus_texts
    ]

    # Build the BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    results = {}

    # Retrieve the top-k documents for each query
    for qid, query in tqdm(
        queries.items(),
        desc="Searching BM25"
    ):
        tokenized_query = simple_tokenize(query)

        scores = bm25.get_scores(tokenized_query)

        indices = np.argsort(scores)[::-1][:top_k]

        ranked = {
            corpus_ids[i]: float(scores[i])
            for i in indices
        }

        results[qid] = ranked

    return results
