def rrf_fuse(results_1, results_2, k=60):
    """
    Fuse two ranked retrieval result sets using
    Reciprocal Rank Fusion (RRF).

    Args:
        results_1: First retrieval results.
        results_2: Second retrieval results.
        k: RRF ranking constant.

    Returns:
        Fused ranked results for each query.
    """

    fused_results = {}

    for query_id in results_1:

        # Collect all documents appearing in either result set
        documents = set(results_1[query_id]) | set(
            results_2.get(query_id, {})
        )

        # Convert rankings to document -> rank dictionaries
        ranks_1 = {
            doc_id: rank
            for rank, doc_id in enumerate(
                results_1[query_id],
                start=1
            )
        }

        ranks_2 = {
            doc_id: rank
            for rank, doc_id in enumerate(
                results_2.get(query_id, {}),
                start=1
            )
        }

        fused_scores = {}

        for doc_id in documents:
            score = 0.0

            if doc_id in ranks_1:
                score += 1.0 / (k + ranks_1[doc_id])

            if doc_id in ranks_2:
                score += 1.0 / (k + ranks_2[doc_id])

            fused_scores[doc_id] = score

        # Sort documents by fused RRF score
        fused_results[query_id] = dict(
            sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

    return fused_results
