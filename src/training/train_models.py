"""Train the baseline and personalized MovieLens recommendation models.

This module implements two recommendation approaches for DM4ML Assignment 2:

1. Popularity-based recommendation: ranks movies by training interaction
   count, with mean training rating and MovieID used as deterministic tie-breaks.
2. Personalized collaborative filtering: applies truncated SVD to the sparse
   user-item ratings matrix and ranks unseen movies by reconstructed rating.

The module also logs model parameters, training diagnostics, and artifacts to
MLflow, and writes per-user top-N recommendation lists for later evaluation.

Author: Suraj Gate
BITS Student ID: 2025ag05348
Course: Data Management for Machine Learning (DM4ML)
Assignment: 2
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAINING_REPORT_DIR = REPORTS_DIR / "training"

# MLflow SQLite tracking configuration
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

MLFLOW_EXPERIMENT_NAME = (
    "DM4ML_Assignment_2_Recommendation"
)


TOP_N = 10
SVD_COMPONENTS = 32
SVD_ITERATIONS = 7
RANDOM_STATE = 42

for directory in (
    MODELS_DIR,
    TRAINING_REPORT_DIR,
    MLFLOW_ARTIFACT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


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
    TRAINING_REPORT_DIR / "training.log", encoding="utf-8"
)
_file_handler.setFormatter(_LOG_FORMAT)
LOGGER.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def load_training_inputs() -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the sparse matrix, ID mappings, training data, and item features.

    Returns:
        tuple:
            ``(interaction_matrix, user_index, item_index, item_features)``.

    Raises:
        FileNotFoundError: If any expected preparation output is missing.
        ValueError: If matrix dimensions or mappings are inconsistent.
    """
    required_paths = [
        FEATURES_DIR / "user_item_interaction_matrix.npz",
        FEATURES_DIR / "user_index.csv",
        FEATURES_DIR / "item_index.csv",
        PROCESSED_DIR / "train.parquet",
        FEATURES_DIR / "item_features.parquet",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required training inputs are missing:\n- " + "\n- ".join(missing)
        )

    matrix = load_npz(FEATURES_DIR / "user_item_interaction_matrix.npz").tocsr()
    user_index = pd.read_csv(FEATURES_DIR / "user_index.csv")
    item_index = pd.read_csv(FEATURES_DIR / "item_index.csv")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    item_features = pd.read_parquet(FEATURES_DIR / "item_features.parquet")

    if matrix.shape != (len(user_index), len(item_index)):
        raise ValueError(
            "Sparse matrix shape does not match the user/item index files: "
            f"matrix={matrix.shape}, users={len(user_index)}, items={len(item_index)}"
        )

    required_user = {"matrix_index", "UserID"}
    required_item = {"matrix_index", "MovieID"}
    if not required_user.issubset(user_index.columns):
        raise ValueError(f"user_index.csv must contain {sorted(required_user)}")
    if not required_item.issubset(item_index.columns):
        raise ValueError(f"item_index.csv must contain {sorted(required_item)}")

    if not {"UserID", "MovieID", "Rating"}.issubset(train.columns):
        raise ValueError("train.parquet must contain UserID, MovieID, and Rating")
    if not {"MovieID", "item_interaction_count", "item_avg_rating"}.issubset(
        item_features.columns
    ):
        raise ValueError(
            "item_features.parquet is missing popularity or rating columns"
        )

    if matrix.nnz != len(train):
        raise ValueError(
            "Matrix non-zero count does not match training interaction count: "
            f"matrix.nnz={matrix.nnz}, train_rows={len(train)}"
        )

    LOGGER.info(
        "Loaded training matrix %s with %s non-zero interactions",
        matrix.shape,
        f"{matrix.nnz:,}",
    )
    return matrix, user_index, item_index, item_features


def _save_recommendations(
    rows: list[dict[str, Any]], output_name: str
) -> Path:
    """Persist a long-form recommendation table as Parquet.

    Args:
        rows: Recommendation records with UserID, Rank, MovieID, and Score.
        output_name: Output filename under ``data/features``.

    Returns:
        Path to the saved Parquet file.
    """
    output_path = FEATURES_DIR / output_name
    frame = pd.DataFrame(
        rows, columns=["UserID", "Rank", "MovieID", "Score"]
    )
    frame.to_parquet(output_path, engine="pyarrow", index=False)
    LOGGER.info("Saved %s recommendation rows to %s", f"{len(frame):,}", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Model 1: popularity-based baseline
# ---------------------------------------------------------------------------

def train_popularity_model(
    user_index: pd.DataFrame,
    item_index: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_matrix: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build and persist a global popularity recommender.

    Popularity is computed from training interactions only. Movies are sorted
    by interaction count descending, mean rating descending, then MovieID
    ascending. Previously rated movies are excluded from each user's list.
    The complete validated movie catalogue is retained, so items not observed
    in training receive a zero popularity score rather than being discarded.

    Args:
        user_index: Matrix row to UserID mapping.
        item_index: Matrix column to MovieID mapping for training-known movies.
        item_features: Item features for the complete movie catalogue.
        interaction_matrix: CSR training user-item ratings matrix.

    Returns:
        Tuple containing model JSON path, recommendations Parquet path,
        and model metadata.
    """
    popularity = item_features[
        ["MovieID", "item_interaction_count", "item_avg_rating"]
    ].copy()
    popularity["item_interaction_count"] = popularity[
        "item_interaction_count"
    ].fillna(0).astype("int64")
    popularity["item_avg_rating"] = popularity["item_avg_rating"].fillna(0.0)
    popularity = popularity.sort_values(
        ["item_interaction_count", "item_avg_rating", "MovieID"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    popularity_path = MODELS_DIR / "popularity_model.json"
    popularity_payload = {
        "model_name": "popularity_based_recommender",
        "student_id": "2025ag05348",
        "ranking_rule": [
            "training interaction count descending",
            "training average rating descending",
            "MovieID ascending",
        ],
        "training_data_only": True,
        "top_n_default": TOP_N,
        "ranked_items": [
            {
                "MovieID": int(row.MovieID),
                "interaction_count": int(row.item_interaction_count),
                "average_rating": float(row.item_avg_rating),
            }
            for row in popularity.itertuples(index=False)
        ],
    }
    popularity_path.write_text(
        json.dumps(popularity_payload, indent=2), encoding="utf-8"
    )

    ordered_movie_ids = popularity["MovieID"].to_numpy(dtype=np.int64)
    ordered_scores = popularity["item_interaction_count"].to_numpy(dtype=np.float64)
    matrix_movie_ids = item_index.sort_values("matrix_index")["MovieID"].to_numpy(
        dtype=np.int64
    )

    recommendation_rows: list[dict[str, Any]] = []
    for user_row in user_index.sort_values("matrix_index").itertuples(index=False):
        matrix_row = int(user_row.matrix_index)
        user_id = int(user_row.UserID)
        seen_positions = interaction_matrix.getrow(matrix_row).indices
        seen_movie_ids = set(matrix_movie_ids[seen_positions].tolist())

        rank = 0
        for movie_id, score in zip(ordered_movie_ids, ordered_scores):
            movie_id_int = int(movie_id)
            if movie_id_int in seen_movie_ids:
                continue
            rank += 1
            recommendation_rows.append(
                {
                    "UserID": user_id,
                    "Rank": rank,
                    "MovieID": movie_id_int,
                    "Score": float(score),
                }
            )
            if rank == TOP_N:
                break

    recommendation_path = _save_recommendations(
        recommendation_rows, "recommendations_popularity.parquet"
    )

    metadata = {
        "algorithm": "popularity_based",
        "training_interactions": int(interaction_matrix.nnz),
        "users": int(len(user_index)),
        "catalog_items": int(len(item_features)),
        "top_n": TOP_N,
        "recommendation_rows": int(len(recommendation_rows)),
    }
    return popularity_path, recommendation_path, metadata


# ---------------------------------------------------------------------------
# Model 2: collaborative filtering using truncated SVD
# ---------------------------------------------------------------------------

def train_svd_model(
    interaction_matrix: Any,
    user_index: pd.DataFrame,
    item_index: pd.DataFrame,
) -> tuple[Path, Path, dict[str, Any]]:
    """Fit a latent-factor collaborative-filtering model using Truncated SVD.

    The input is the sparse ratings matrix constructed from the training
    partition only. The resulting user factors and movie factors are used to
    estimate preference scores. Movies already present in a user's training
    history are excluded before producing recommendations.

    Args:
        interaction_matrix: CSR matrix of explicit training ratings.
        user_index: Matrix row to UserID mapping.
        item_index: Matrix column to MovieID mapping.

    Returns:
        Tuple containing serialized model path, recommendations Parquet path,
        and model metadata.

    Raises:
        ValueError: If the matrix is too small for the requested components.
    """
    max_components = min(interaction_matrix.shape) - 1
    n_components = min(SVD_COMPONENTS, max_components)
    if n_components < 2:
        raise ValueError(
            f"Interaction matrix {interaction_matrix.shape} is too small for SVD"
        )

    LOGGER.info(
        "Training TruncatedSVD with n_components=%s, n_iter=%s, random_state=%s",
        n_components,
        SVD_ITERATIONS,
        RANDOM_STATE,
    )

    model = TruncatedSVD(
        n_components=n_components,
        n_iter=SVD_ITERATIONS,
        random_state=RANDOM_STATE,
        algorithm="randomized",
    )
    user_factors = model.fit_transform(interaction_matrix)
    item_factors = model.components_

    svd_model_path = MODELS_DIR / "collaborative_filtering_svd.joblib"
    dump(
        {
            "model": model,
            "user_factors": user_factors,
            "user_index": user_index,
            "item_index": item_index,
            "top_n": TOP_N,
        },
        svd_model_path,
        compress=3,
    )

    matrix_movie_ids = item_index.sort_values("matrix_index")["MovieID"].to_numpy(
        dtype=np.int64
    )
    ordered_user_index = user_index.sort_values("matrix_index")

    recommendation_rows: list[dict[str, Any]] = []
    for user_row in ordered_user_index.itertuples(index=False):
        matrix_row = int(user_row.matrix_index)
        user_id = int(user_row.UserID)
        scores = np.asarray(user_factors[matrix_row] @ item_factors).ravel()
        seen_positions = interaction_matrix.getrow(matrix_row).indices
        scores[seen_positions] = -np.inf

        available_count = int(np.isfinite(scores).sum())
        if available_count == 0:
            continue
        candidate_count = min(TOP_N, available_count)
        candidate_positions = np.argpartition(
            scores, -candidate_count
        )[-candidate_count:]
        candidate_positions = candidate_positions[
            np.argsort(scores[candidate_positions])[::-1]
        ]

        for rank, item_position in enumerate(candidate_positions, start=1):
            recommendation_rows.append(
                {
                    "UserID": user_id,
                    "Rank": rank,
                    "MovieID": int(matrix_movie_ids[item_position]),
                    "Score": float(scores[item_position]),
                }
            )

    recommendation_path = _save_recommendations(
        recommendation_rows, "recommendations_svd.parquet"
    )
    total_possible = int(interaction_matrix.shape[0] * interaction_matrix.shape[1])
    metadata = {
        "algorithm": "collaborative_filtering_truncated_svd",
        "n_components": int(n_components),
        "n_iter": SVD_ITERATIONS,
        "random_state": RANDOM_STATE,
        "training_interactions": int(interaction_matrix.nnz),
        "users": int(interaction_matrix.shape[0]),
        "training_known_items": int(interaction_matrix.shape[1]),
        "total_possible_pairs": total_possible,
        "matrix_density": float(interaction_matrix.nnz / total_possible),
        "explained_variance_ratio_sum": float(
            np.sum(model.explained_variance_ratio_)
        ),
        "top_n": TOP_N,
        "recommendation_rows": int(len(recommendation_rows)),
    }
    return svd_model_path, recommendation_path, metadata


# ---------------------------------------------------------------------------
# MLflow experiment tracking
# ---------------------------------------------------------------------------


def log_model_run(
    run_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
) -> str:
    """
    Log model parameters, metrics and artifacts to MLflow.

    Uses a local SQLite database for experiment tracking
    and a local directory for model artifacts.

    Args:
        run_name: Human-readable model run name.
        parameters: Model configuration and training parameters.
        metrics: Numeric training diagnostics.
        artifact_paths: Model files to attach to the run.

    Returns:
        str: MLflow run ID.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )

    if experiment is None:
        mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri(),
        )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as active_run:

        mlflow.set_tags({
            "student_id": "2025ag05348",
            "author": "Suraj Gate",
            "dataset": "MovieLens 1M",
            "assignment": "DM4ML Assignment 2",
        })

        mlflow.log_params({
            key: str(value)
            for key, value in parameters.items()
        })

        mlflow.log_metrics(metrics)

        for artifact_path in artifact_paths:
            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="model_artifacts",
            )

        return active_run.info.run_id



# ---------------------------------------------------------------------------
# End-to-end training entry point
# ---------------------------------------------------------------------------

def run_training() -> dict[str, Any]:
    """Train both recommenders, log experiments, and write a summary report.

    Returns:
        Dictionary containing both model configurations, training metrics,
        MLflow run IDs, and generated artifact paths.
    """
    LOGGER.info("Starting DM4ML recommendation model training")
    matrix, user_index, item_index, item_features = load_training_inputs()

    popularity_path, popularity_recs, popularity_info = train_popularity_model(
        user_index, item_index, item_features, matrix
    )
    popularity_run_id = log_model_run(
        run_name="Popularity_Baseline",
        parameters={
            "algorithm": "popularity_based",
            "ranking_metric": "training_interaction_count",
            "tie_break_1": "training_average_rating_desc",
            "tie_break_2": "MovieID_asc",
            "top_n": TOP_N,
            "training_interactions": popularity_info["training_interactions"],
            "users": popularity_info["users"],
            "catalog_items": popularity_info["catalog_items"],
        },
        metrics={
            "training_interactions": float(popularity_info["training_interactions"]),
            "users": float(popularity_info["users"]),
            "catalog_items": float(popularity_info["catalog_items"]),
            "recommendation_rows": float(popularity_info["recommendation_rows"]),
        },
        artifact_paths=[popularity_path, popularity_recs],
    )

    svd_path, svd_recs, svd_info = train_svd_model(
        matrix, user_index, item_index
    )
    svd_run_id = log_model_run(
        run_name="Collaborative_Filtering_Truncated_SVD",
        parameters={
            "algorithm": svd_info["algorithm"],
            "n_components": svd_info["n_components"],
            "n_iter": svd_info["n_iter"],
            "random_state": svd_info["random_state"],
            "top_n": svd_info["top_n"],
            "training_interactions": svd_info["training_interactions"],
        },
        metrics={
            "training_interactions": float(svd_info["training_interactions"]),
            "users": float(svd_info["users"]),
            "training_known_items": float(svd_info["training_known_items"]),
            "matrix_density": svd_info["matrix_density"],
            "explained_variance_ratio_sum": svd_info[
                "explained_variance_ratio_sum"
            ],
            "recommendation_rows": float(svd_info["recommendation_rows"]),
        },
        artifact_paths=[svd_path, svd_recs],
    )

    summary = {
        "project": "DM4ML Assignment 2 - Recommendation System Pipeline",
        "author": "Suraj Gate",
        "student_id": "2025ag05348",
        "dataset": "MovieLens 1M",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_data_only": True,
        "top_n_recommendations": TOP_N,
        "models": {
            "popularity_baseline": {
                **popularity_info,
                "mlflow_run_id": popularity_run_id,
                "model_artifact": str(popularity_path),
                "recommendations_artifact": str(popularity_recs),
            },
            "collaborative_filtering_svd": {
                **svd_info,
                "mlflow_run_id": svd_run_id,
                "model_artifact": str(svd_path),
                "recommendations_artifact": str(svd_recs),
            },
        },
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
    }

    summary_path = TRAINING_REPORT_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    LOGGER.info("Training summary saved to %s", summary_path)
    LOGGER.info("Both recommendation models trained and tracked successfully")
    return summary


def main() -> None:
    """Run model training and print the resulting summary as JSON."""
    print(json.dumps(run_training(), indent=4))


if __name__ == "__main__":
    main()
