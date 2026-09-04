import pandas as pd

from beir.retrieval.evaluation import EvaluateRetrieval


def evaluate_results(
    results,
    qrels,
    k_values=(1, 3, 5, 10)
):
    """
    Evaluate retrieval results using BEIR metrics.

    Metrics:
    - NDCG
    - MAP
    - MRR
    - Recall
    """

    evaluator = EvaluateRetrieval()

    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels,
        results,
        list(k_values)
    )

    mrr = evaluator.evaluate_custom(
        qrels,
        results,
        list(k_values),
        metric="mrr"
    )

    rows = []

    for k in k_values:
        rows.append({
            "k": k,
            "NDCG@k": round(ndcg[f"NDCG@{k}"], 4),
            "MAP@k": round(_map[f"MAP@{k}"], 4),
            "MRR@k": round(mrr[f"MRR@{k}"], 4),
            "Recall@k": round(recall[f"Recall@{k}"], 4),
        })

    return pd.DataFrame(rows)
