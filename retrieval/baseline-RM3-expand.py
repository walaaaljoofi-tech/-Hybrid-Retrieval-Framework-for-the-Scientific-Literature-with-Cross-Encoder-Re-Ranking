
from collections import Counter


def rm3_expand_query(
    original_query,
    top_docs,
    docid_to_index,
    doc_texts,
    alpha=0.7,
    fb_docs=5,
    fb_terms=20
):
    """
    Expand a query using pseudo-relevance feedback
    from the top-ranked BM25 documents.
    """

    term_counts = Counter()

    # Collect terms from the top feedback documents
    for doc_id, score in top_docs[:fb_docs]:

        # Map document ID to its index in the corpus
        idx = docid_to_index[doc_id]

        # Tokenize the document text
        tokens = doc_texts[idx].lower().split()

        # Count term frequencies
        term_counts.update(tokens)

    # Select the most frequent expansion terms
    expansion_terms = [
        term
        for term, _ in term_counts.most_common(fb_terms)
    ]

    # Append expansion terms to the original query
    expanded_query = (
        original_query
        + " "
        + " ".join(expansion_terms)
    )

    return expanded_query
