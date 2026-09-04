import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Re-rank retrieved documents using a Cross-Encoder model.
    """

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        max_length=512
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=max_length
        )

    def rerank(
        self,
        results,
        queries,
        doc_text_by_id,
        top_k=100
    ):
        """
        Re-rank the top-k retrieved documents for each query.

        Args:
            results:
                Dictionary in the form:
                {qid: {doc_id: score, ...}}

            queries:
                Dictionary:
                {qid: query_text}

            doc_text_by_id:
                Dictionary:
                {doc_id: document_text}

            top_k:
                Number of candidate documents to rerank.

        Returns:
            Re-ranked retrieval results.
        """

        reranked_results = {}

        for qid, doc_scores in tqdm(
            results.items(),
            desc="Cross-Encoder Re-ranking"
        ):
            query = queries[qid]

            candidate_docs = list(
                doc_scores.keys()
            )[:top_k]

            pairs = [
                [query, doc_text_by_id[doc_id]]
                for doc_id in candidate_docs
            ]

            ce_scores = self.model.predict(pairs)

            order = np.argsort(
                ce_scores
            )[::-1]

            reranked = {
                candidate_docs[i]: float(ce_scores[i])
                for i in order
            }

            reranked_results[qid] = reranked

        return reranked_results
