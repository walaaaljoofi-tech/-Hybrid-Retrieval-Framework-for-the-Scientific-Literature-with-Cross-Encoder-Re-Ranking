import pandas as pd

from src.data.loader import load_scifact

from src.preprocessing.text import (
    build_doc_text,
    build_doc_text_from_beir,
)

from src.retrieval.bm25 import (
    run_bm25,
    build_bm25,
)

from src.retrieval.dense import (
    run_dense_model,
    DENSE_MODELS,
)

from src.embeddings.specter import (
    build_specter_faiss,
)

from src.retrieval.query_expansion import (
    expand_query,
)

from src.retrieval.hybrid import (
    rrf_fuse,
)

from src.reranking.cross_encoder import (
    CrossEncoderReranker,
)

from src.evaluation.metrics import (
    evaluate_results,
)


def run_pipeline(
    top_k=100,
    rrf_k=60,
    run_query_expansion=True,
    run_cross_encoder=True
):
    """
    Run the complete Scientific Paper Retrieval pipeline.

    Pipeline:
    1. Load SciFact
    2. Prepare documents
    3. Run BM25
    4. Run dense retrieval models
    5. Run SPECTER query expansion
    6. Run hybrid RRF retrieval
    7. Run Cross-Encoder reranking
    8. Evaluate all models

    Returns:
        all_results:
            Retrieval results for every model.

        evaluation_table:
            Final evaluation metrics.
    """

    print("=" * 70)
    print("Scientific Paper Retrieval Pipeline")
    print("=" * 70)

    # ==========================================================
    # STEP 1: Load SciFact dataset
    # ==========================================================

    print("\n[1/8] Loading SciFact dataset...")

    corpus, queries, qrels = load_scifact()

    print(f"Documents: {len(corpus)}")
    print(f"Queries:   {len(queries)}")
    print(f"Qrels:     {len(qrels)}")


    # ==========================================================
    # STEP 2: Prepare document texts
    # ==========================================================

    print("\n[2/8] Preparing document texts...")

    doc_ids = list(corpus.keys())

    doc_texts = [
        build_doc_text(corpus[doc_id])
        for doc_id in doc_ids
    ]

    doc_text_by_id = {
        doc_id: build_doc_text(corpus[doc_id])
        for doc_id in doc_ids
    }

    print(f"Prepared {len(doc_texts)} documents.")


    # ==========================================================
    # Container for all retrieval results
    # ==========================================================

    all_results = {}


    # ==========================================================
    # STEP 3: BM25 Baseline
    # ==========================================================

    print("\n[3/8] Running BM25 baseline...")

    results_bm25 = run_bm25(
        corpus_ids=doc_ids,
        corpus_texts=doc_texts,
        queries=queries,
        top_k=top_k
    )

    all_results["BM25"] = results_bm25


    # ==========================================================
    # STEP 4: Dense Retrieval
    # SciBERT / SPECTER / SciNCL
    # ==========================================================

    print("\n[4/8] Running dense retrieval models...")

    for pretty_name, model_name in DENSE_MODELS.items():

        results_dense = run_dense_model(
            model_name=model_name,
            corpus_ids=doc_ids,
            corpus_texts=doc_texts,
            queries=queries,
            top_k=top_k
        )

        all_results[pretty_name] = results_dense


    # ==========================================================
    # STEP 5: SPECTER Query Expansion -> BM25
    # ==========================================================

    if run_query_expansion:

        print(
            "\n[5/8] Running SPECTER nearest-neighbor "
            "query expansion..."
        )

        specter_model, faiss_index, specter_doc_ids, specter_doc_texts = (
            build_specter_faiss(corpus)
        )

        bm25_expansion, bm25_doc_ids = build_bm25(corpus)

        results_specter_expansion = {}

        for qid, query in queries.items():

            expanded_query, terms, neighbors = expand_query(
                query,
                specter_model,
                faiss_index,
                specter_doc_texts,
                k_neighbors=10,
                top_m_terms=12
            )

            tokenized_query = expanded_query.lower().split()

            scores = bm25_expansion.get_scores(
                tokenized_query
            )

            import numpy as np

            indices = np.argsort(
                scores
            )[::-1][:top_k]

            ranked = {
                bm25_doc_ids[i]: float(scores[i])
                for i in indices
            }

            results_specter_expansion[qid] = ranked

        all_results[
            "BM25 + SPECTER Expansion"
        ] = results_specter_expansion

    else:
        print("\n[5/8] Query expansion skipped.")


    # ==========================================================
    # STEP 6: Hybrid Retrieval using RRF
    # ==========================================================

    print("\n[6/8] Running Hybrid Retrieval (RRF)...")

    # Use SciNCL + BM25 as the main hybrid configuration
    results_hybrid = rrf_fuse(
        results_bm25,
        all_results["SciNCL"],
        k=rrf_k
    )

    all_results[
        "Hybrid BM25 + SciNCL"
    ] = results_hybrid


    # ==========================================================
    # STEP 7: Cross-Encoder Re-ranking
    # ==========================================================

    if run_cross_encoder:

        print("\n[7/8] Running Cross-Encoder reranking...")

        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            max_length=512
        )

        results_hybrid_ce = reranker.rerank(
            results=results_hybrid,
            queries=queries,
            doc_text_by_id=doc_text_by_id,
            top_k=top_k
        )

        all_results[
            "Hybrid BM25 + SciNCL + CE"
        ] = results_hybrid_ce

    else:
        print("\n[7/8] Cross-Encoder reranking skipped.")


    # ==========================================================
    # STEP 8: Evaluation
    # ==========================================================

    print("\n[8/8] Evaluating all retrieval models...")

    evaluation_tables = []

    for model_name, results in all_results.items():

        print(f"Evaluating: {model_name}")

        df = evaluate_results(
            results,
            qrels,
            k_values=(1, 3, 5, 10)
        )

        df["Model"] = model_name

        evaluation_tables.append(df)

    evaluation_table = pd.concat(
        evaluation_tables,
        ignore_index=True
    )

    # Rearrange columns
    evaluation_table = evaluation_table[
        [
            "Model",
            "k",
            "NDCG@k",
            "MAP@k",
            "MRR@k",
            "Recall@k",
        ]
    ]


    # ==========================================================
    # Final output
    # ==========================================================

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)

    print("\nFinal Evaluation Results:\n")
    print(evaluation_table)

    return all_results, evaluation_table
