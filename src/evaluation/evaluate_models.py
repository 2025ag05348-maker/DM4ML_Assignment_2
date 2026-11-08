"""Evaluate MovieLens recommendation models for DM4ML Assignment 2.

This module compares the popularity baseline and personalized Truncated-SVD
recommender using a chronological holdout set. Ratings >= 4 are treated as
relevant. Precision@5, Recall@5, NDCG@5, catalogue coverage, and genre-based
intra-list diversity are calculated for both models. Ranking metrics are macro-
averaged over users with at least one relevant test interaction.

Evaluation metrics and report artifacts are also logged to the corresponding
MLflow training runs using the project's SQLite tracking database.

Author: Suraj Gate
BITS Student ID: 2025ag05348
Course: Data Management for Machine Learning (DM4ML)
Assignment: 2
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVALUATION_REPORT_DIR = REPORTS_DIR / "evaluation"

TRAINING_SUMMARY_PATH = PROJECT_ROOT / "reports" / "training" / "training_summary.json"
TEST_PATH = PROCESSED_DIR / "test.parquet"
MOVIES_PATH = PROCESSED_DIR / "movies_validated.parquet"

RECOMMENDATION_FILES = {
    "popularity_baseline": FEATURES_DIR / "recommendations_popularity.parquet",
    "collaborative_filtering_svd": FEATURES_DIR / "recommendations_svd.parquet",
}

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "DM4ML_Assignment_2_Recommendation"

K = 5
RELEVANT_RATING_THRESHOLD = 4

EVALUATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.propagate = False

_LOG_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_LOG_FORMAT)
LOGGER.addHandler(_console_handler)

_file_handler = logging.FileHandler(
    EVALUATION_REPORT_DIR / "evaluation.log", encoding="utf-8"
)
_file_handler.setFormatter(_LOG_FORMAT)
LOGGER.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_evaluation_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the chronological test set, movie catalogue, recommendations,
    and training metadata.

    Returns:
        Tuple containing the test interactions, movie catalogue, and training
        summary dictionary.

    Raises:
        FileNotFoundError: If a required evaluation input is missing.
        ValueError: If an input does not contain required columns.
    """
    required_paths = [TEST_PATH, MOVIES_PATH, TRAINING_SUMMARY_PATH]
    required_paths.extend(RECOMMENDATION_FILES.values())
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required evaluation inputs are missing:\n- " + "\n- ".join(missing)
        )

    test = pd.read_parquet(TEST_PATH)
    movies = pd.read_parquet(MOVIES_PATH)
    training_summary = json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))

    if not {"UserID", "MovieID", "Rating"}.issubset(test.columns):
        raise ValueError("test.parquet must contain UserID, MovieID, and Rating")
    if not {"MovieID", "Genres"}.issubset(movies.columns):
        raise ValueError("movies_validated.parquet must contain MovieID and Genres")

    for model_name, path in RECOMMENDATION_FILES.items():
        recs = pd.read_parquet(path)
        if not {"UserID", "Rank", "MovieID", "Score"}.issubset(recs.columns):
            raise ValueError(
                f"{path.name} must contain UserID, Rank, MovieID, and Score"
            )
        LOGGER.info("Loaded %s recommendation rows for %s", f"{len(recs):,}", model_name)

    LOGGER.info("Loaded %s chronological test interactions", f"{len(test):,}")
    LOGGER.info("Loaded %s movies for catalogue and genre evaluation", f"{len(movies):,}")
    return test, movies, training_summary


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _dcg_at_k(relevance: list[int], k: int) -> float:
    """Calculate discounted cumulative gain for binary relevance labels."""
    values = np.asarray(relevance[:k], dtype=np.float64)
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2, dtype=np.float64))
    return float(np.sum(values / discounts))


def _genre_tokens(value: Any) -> set[str]:
    """Convert a pipe-separated MovieLens genre field into a token set."""
    if pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split("|") if token.strip()}


def _intra_list_diversity(movie_ids: list[int], genre_map: dict[int, set[str]]) -> float:
    """Calculate mean pairwise genre Jaccard distance for one recommendation list.

    Jaccard distance is 1 - intersection/union. If both genre sets are empty,
    similarity is defined as zero and distance as one. MovieLens normally
    provides a genre label, including the explicit 'Unknown' label.
    """
    if len(movie_ids) < 2:
        return 0.0

    distances: list[float] = []
    for first_id, second_id in combinations(movie_ids, 2):
        first = genre_map.get(int(first_id), set())
        second = genre_map.get(int(second_id), set())
        union = first | second
        similarity = (len(first & second) / len(union)) if union else 0.0
        distances.append(1.0 - similarity)

    return float(np.mean(distances)) if distances else 0.0


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_recommender(
    model_name: str,
    recommendation_path: Path,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = K,
) -> dict[str, Any]:
    """Evaluate one recommendation output against held-out test interactions.

    Relevance is defined as an observed test rating >= 4. Precision@K,
    Recall@K and NDCG@K are averaged over users who have at least one relevant
    test item. Precision uses K as its denominator. Coverage uses the complete
    validated movie catalogue. Diversity is mean pairwise genre Jaccard
    distance across each user's top-K list, averaged over users with at least
    two recommendations.

    Args:
        model_name: Stable name identifying the recommendation model.
        recommendation_path: Parquet file containing ranked recommendations.
        test: Chronological test interactions.
        movies: Complete validated movie catalogue with genres.
        k: Number of top recommendations to evaluate.

    Returns:
        Dictionary containing model name, metric values, and evaluation counts.
    """
    recommendations = pd.read_parquet(recommendation_path)
    recommendations = recommendations.sort_values(
        ["UserID", "Rank"], ascending=[True, True]
    )
    top_k = recommendations.groupby("UserID", sort=False).head(k).copy()

    relevant_rows = test.loc[test["Rating"] >= RELEVANT_RATING_THRESHOLD, ["UserID", "MovieID"]]
    relevant_by_user = (
        relevant_rows.groupby("UserID")["MovieID"].apply(lambda values: set(map(int, values)))
    ).to_dict()
    relevant_users = sorted(relevant_by_user)

    recommendation_lists = {
        int(user_id): list(map(int, group["MovieID"].tolist()))
        for user_id, group in top_k.groupby("UserID", sort=False)
    }

    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []

    for user_id in relevant_users:
        relevant_items = relevant_by_user[user_id]
        recommended_items = recommendation_lists.get(int(user_id), [])
        relevance = [1 if movie_id in relevant_items else 0 for movie_id in recommended_items]
        hits = int(sum(relevance))

        precision_values.append(hits / k)
        recall_values.append(hits / len(relevant_items))

        dcg = _dcg_at_k(relevance, k)
        ideal_length = min(k, len(relevant_items))
        ideal_dcg = _dcg_at_k([1] * ideal_length, k)
        ndcg_values.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    genre_map = {
        int(row.MovieID): _genre_tokens(row.Genres)
        for row in movies[["MovieID", "Genres"]].itertuples(index=False)
    }

    diversity_values = [
        _intra_list_diversity(movie_ids, genre_map)
        for movie_ids in recommendation_lists.values()
        if len(movie_ids) >= 2
    ]

    recommended_catalog_items = {
        int(movie_id) for movie_id in top_k["MovieID"].unique()
    }
    catalogue_size = int(movies["MovieID"].nunique())
    coverage = (
        len(recommended_catalog_items) / catalogue_size if catalogue_size else 0.0
    )

    result = {
        "model": model_name,
        f"precision_at_{k}": float(np.mean(precision_values)) if precision_values else 0.0,
        f"recall_at_{k}": float(np.mean(recall_values)) if recall_values else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "catalogue_coverage": float(coverage),
        "unique_recommended_items": int(len(recommended_catalog_items)),
        "catalogue_size": catalogue_size,
        "diversity": float(np.mean(diversity_values)) if diversity_values else 0.0,
        "ranking_evaluation_users": int(len(relevant_users)),
        "recommendation_users": int(len(recommendation_lists)),
        "recommendation_rows_at_k": int(len(top_k)),
        "relevant_test_interactions": int(len(relevant_rows)),
        "relevance_rating_threshold": RELEVANT_RATING_THRESHOLD,
        "k": int(k),
    }

    LOGGER.info(
        "%s evaluation: Precision@%s=%.5f Recall@%s=%.5f NDCG@%s=%.5f Coverage=%.5f Diversity=%.5f",
        model_name,
        k,
        result[f"precision_at_{k}"],
        k,
        result[f"recall_at_{k}"],
        k,
        result[f"ndcg_at_{k}"],
        result["catalogue_coverage"],
        result["diversity"],
    )
    return result


# ---------------------------------------------------------------------------
# MLflow evaluation metric logging
# ---------------------------------------------------------------------------

def log_evaluation_to_mlflow(
    training_summary: dict[str, Any],
    model_name: str,
    metrics: dict[str, Any],
    report_paths: list[Path],
) -> None:
    """Attach evaluation metrics and reports to the corresponding MLflow run.

    Args:
        training_summary: Existing training summary containing MLflow run IDs.
        model_name: Key for the model in the training summary.
        metrics: Evaluation metrics to log.
        report_paths: JSON and CSV report files to attach to the run.

    Raises:
        KeyError: If the training summary does not contain the expected run ID.
    """
    run_id = training_summary["models"][model_name]["mlflow_run_id"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key != "k"
        and key != "relevance_rating_threshold"
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {f"evaluation_{key}": value for key, value in numeric_metrics.items()},
            step=1,
        )
        for report_path in report_paths:
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")


# ---------------------------------------------------------------------------
# Evaluation pipeline entry point
# ---------------------------------------------------------------------------

def run_evaluation() -> dict[str, Any]:
    """Evaluate both recommenders, persist comparison reports, and log metrics.

    Returns:
        Dictionary containing evaluation definitions, both model results,
        and output report paths.
    """
    LOGGER.info("Starting DM4ML recommendation model evaluation")
    test, movies, training_summary = load_evaluation_inputs()

    results: list[dict[str, Any]] = []
    for model_name, recommendation_path in RECOMMENDATION_FILES.items():
        results.append(
            evaluate_recommender(
                model_name=model_name,
                recommendation_path=recommendation_path,
                test=test,
                movies=movies,
                k=K,
            )
        )

    comparison = pd.DataFrame(results)
    comparison_path = EVALUATION_REPORT_DIR / "model_evaluation_results.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "project": "DM4ML Assignment 2 - Recommendation System Pipeline",
        "author": "Suraj Gate",
        "student_id": "2025ag05348",
        "dataset": "MovieLens 1M",
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_methodology": {
            "test_partition": "Per-user chronological holdout (latest 20% of each user's interactions)",
            "relevance_definition": f"Test rating >= {RELEVANT_RATING_THRESHOLD}",
            "k": K,
            "precision_recall_ndcg_population": "Users with at least one relevant test interaction; users without relevant test items are excluded from macro-averaging these three metrics.",
            "precision_denominator": K,
            "coverage_denominator": "All unique movies in movies_validated.parquet",
            "diversity_definition": "Mean pairwise Jaccard distance between genre sets within each user's top-K recommendation list, then averaged across users with at least two recommendations.",
            "test_data_used_for_training": False,
        },
        "models": {result["model"]: result for result in results},
        "comparison_csv": str(comparison_path),
    }

    summary_path = EVALUATION_REPORT_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    report_paths = [comparison_path, summary_path]
    for result in results:
        log_evaluation_to_mlflow(
            training_summary=training_summary,
            model_name=result["model"],
            metrics=result,
            report_paths=report_paths,
        )

    LOGGER.info("Evaluation comparison saved to %s", comparison_path)
    LOGGER.info("Evaluation summary saved to %s", summary_path)
    LOGGER.info("Both model evaluations were logged to their MLflow runs")
    return summary


def main() -> None:
    """Execute model evaluation and print the resulting JSON summary."""
    print(json.dumps(run_evaluation(), indent=4))


if __name__ == "__main__":
    main()
