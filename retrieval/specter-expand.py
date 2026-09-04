import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from src.embeddings.specter import l2_normalize


def extract_terms_tfidf(
    neighbor_texts,
    top_m=12,
    ngram_range=(1, 2),
    max_df=0.95,
    stop_words="english"
):
    """
    Extract the highest-scoring TF-IDF terms
    from neighboring documents.
    """

    if not neighbor_texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_df=max_df
    )

    matrix = vectorizer.fit_transform(
        neighbor_texts
    )

    scores = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    terms = np.array(
        vectorizer.get_feature_names_out()
    )

    if scores.size == 0:
        return []

    top_indices = np.argsort(
        -scores
    )[:top_m]

    return terms[top_indices].tolist()


def expand_query(
    query_text,
    specter_model,
    faiss_index,
    doc_texts,
    k_neighbors=10,
    top_m_terms=12
):
    """
    Expand a query using SPECTER nearest-neighbor
    documents and TF-IDF terms.
    """

    query_embedding = specter_model.encode(
        [query_text],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    query_embedding = l2_normalize(
        query_embedding
    )

    _, neighbor_indices = faiss_index.search(
        query_embedding,
        k_neighbors
    )

    neighbor_indices = (
        neighbor_indices[0].tolist()
    )

    neighbor_texts = [
        doc_texts[i]
        for i in neighbor_indices
        if i >= 0
    ]

    terms = extract_terms_tfidf(
        neighbor_texts,
        top_m=top_m_terms
    )

    expanded_query = query_text.strip()

    if terms:
        expanded_query += (
            " " + " ".join(terms)
        )

    return (
        expanded_query,
        terms,
        neighbor_indices
    )
