"""
Evaluator Module
================
Computes evaluation metrics for the recommendation system.

Primary metric: Mean Recall@K
"""

import pandas as pd
from typing import List, Tuple, Optional

from .models import EvaluationResult
from .recommender import SHLRecommender, get_recommender


# ============================================================================
# METRICS
# ============================================================================

def recall_at_k(predicted: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Compute Recall@K for a single query.
    """
    if not ground_truth:
        return 0.0

    predicted_set = set(url.rstrip('/').lower() for url in predicted[:k])
    ground_truth_set = set(url.rstrip('/').lower() for url in ground_truth)

    return len(predicted_set & ground_truth_set) / len(ground_truth_set)


def mean_recall_at_k(predictions: dict, ground_truths: dict, k: int = 10) -> float:
    """
    Compute Mean Recall@K across all queries.
    """
    recalls = []
    for query in ground_truths:
        pred = predictions.get(query, [])
        gt = ground_truths.get(query, [])
        recalls.append(recall_at_k(pred, gt, k))

    return sum(recalls) / len(recalls) if recalls else 0.0


# ============================================================================
# TRAIN SET EVALUATION
# ============================================================================

def evaluate_on_train_set(
    train_file: str,
    recommender: Optional[SHLRecommender] = None,
    k: int = 10,
    verbose: bool = True
) -> Tuple[float, List[EvaluationResult]]:
    """
    Evaluate recommender on the Train-Set.
    """

    if recommender is None:
        recommender = get_recommender()

    df = pd.read_excel(train_file, sheet_name="Train-Set")

    query_ground_truths = {}
    for query, group in df.groupby("Query", sort=False):
        query_ground_truths[query] = group["Assessment_url"].tolist()

    results = []
    total_recall = 0.0

    for idx, (query, ground_truth) in enumerate(query_ground_truths.items(), 1):
        recommendations = recommender.get_recommendations(query, num_results=k)
        predicted_urls = [r["url"] for r in recommendations]

        recall = recall_at_k(predicted_urls, ground_truth, k)
        total_recall += recall

        results.append(
            EvaluationResult(
                query=query,
                predicted_urls=predicted_urls,
                ground_truth_urls=ground_truth,
                recall_at_k=recall,
                k=k
            )
        )

        if verbose:
            print(f"\nQuery {idx}/{len(query_ground_truths)}")
            print(f"Recall@{k}: {recall:.4f}")

    mean_recall = total_recall / len(query_ground_truths)

    if verbose:
        print("\n" + "=" * 60)
        print(f"FINAL MEAN Recall@{k}: {mean_recall:.4f}")
        print("=" * 60)

    return mean_recall, results


# ============================================================================
# TEST SET PREDICTIONS (CSV GENERATION)
# ============================================================================

def generate_test_predictions(
    excel_path: str,
    output_path: str,
    recommender: Optional[SHLRecommender] = None,
    k: int = 10,
    verbose: bool = False
):
    """
    Generate predictions for Test-Set and save to CSV in SHL-required format.
    """

    if recommender is None:
        recommender = get_recommender()

    df_test = pd.read_excel(excel_path, sheet_name="Test-Set")
    
    # Print columns for debugging
    if verbose:
        print(f"Test-Set columns: {df_test.columns.tolist()}")
    
    # Find the query column (handle different cases)
    query_col = None
    for col in df_test.columns:
        if col.lower() == 'query':
            query_col = col
            break
    
    if query_col is None:
        raise ValueError(f"Could not find 'query' column. Available columns: {df_test.columns.tolist()}")

    rows = []

    for _, row in df_test.iterrows():
        query = str(row[query_col]).strip()
        
        if not query or query == 'nan':
            continue

        recommendations = recommender.get_recommendations(query, num_results=k)

        for rec in recommendations:
            rows.append({
                "query": query,
                "assessment_url": rec["url"]
            })

    df_predictions = pd.DataFrame(rows)

    if df_predictions.empty:
        raise ValueError("No predictions generated. CSV would be empty.")

    df_predictions.to_csv(output_path, index=False)

    if verbose:
        print("\nCSV VALIDATION")
        print("Columns:", list(df_predictions.columns))
        print("Total rows:", len(df_predictions))
        print("Unique queries:", df_predictions['query'].nunique())

    return df_predictions


# ============================================================================
# REPORTING
# ============================================================================

def print_evaluation_report(results: List[EvaluationResult], k: int = 10) -> None:
    """
    Print a detailed evaluation report.
    """

    recalls = [r.recall_at_k for r in results]
    mean_recall = sum(recalls) / len(recalls)

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"Queries evaluated: {len(results)}")
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(f"Min Recall@{k}: {min(recalls):.4f}")
    print(f"Max Recall@{k}: {max(recalls):.4f}")

    print("\nPer-query summary:")
    for idx, r in enumerate(results, 1):
        status = "✓" if r.recall_at_k >= 0.5 else "✗"
        print(f"{status} Query {idx}: Recall@{k} = {r.recall_at_k:.4f}")