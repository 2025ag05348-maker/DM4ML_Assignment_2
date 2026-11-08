"""Prepare MovieLens interactions for recommendation model development.

This module implements the data preparation and transformation stage of
Suraj Gate's DM4ML Assignment 2 recommendation-system pipeline.

The module reads the validated MovieLens 1M Parquet datasets, converts
Unix timestamps to UTC datetimes, creates a per-user chronological
train/test split, derives user-activity and item-popularity summaries,
and builds a sparse user-item rating matrix from the training partition.

Input files:
    data/processed/users_validated.parquet
    data/processed/movies_validated.parquet
    data/processed/ratings_validated.parquet

Output files:
    data/processed/interactions_prepared.parquet
    data/processed/train.parquet
    data/processed/test.parquet
    data/processed/user_activity_summary.parquet
    data/processed/item_popularity_summary.parquet
    data/features/user_item_interaction_matrix.npz
    data/features/user_index.csv
    data/features/item_index.csv
    data/features/interaction_matrix_metadata.json
    reports/preparation/preparation_summary.json
    reports/preparation/preparation.log

The train/test split is chronological within each user: the earliest 80%
of each user's interactions are assigned to training and the latest 20%
are assigned to testing. This prevents a user's later interactions from
being used to predict that same user's earlier interactions.

Author: Suraj Gate
BITS Student ID: 2025ag05348
Course: Data Management for Machine Learning (DM4ML)
Assignment: 2
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


# ---------------------------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREPARATION_REPORT_DIR = PROJECT_ROOT / "reports" / "preparation"

TRAIN_FRACTION = 0.80
TEST_FRACTION = 1.0 - TRAIN_FRACTION
RANDOM_STATE = 42  # Recorded for reproducibility; no random split is used.

for directory in (PROCESSED_DIR, FEATURES_DIR, PREPARATION_REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("dm4ml.preparation")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.propagate = False

_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_LOG_FORMATTER)
LOGGER.addHandler(_console_handler)

_file_handler = logging.FileHandler(
    PREPARATION_REPORT_DIR / "preparation.log",
    mode="w",
    encoding="utf-8",
)
_file_handler.setFormatter(_LOG_FORMATTER)
LOGGER.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# 3. INPUT LOADING AND SCHEMA CHECKS
# ---------------------------------------------------------------------------

def load_validated_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the validated users, movies, and ratings Parquet datasets.

    Returns:
        A tuple containing ``users``, ``movies``, and ``ratings``
        DataFrames, in that order.

    Raises:
        FileNotFoundError: If any validated Parquet file is missing.
        ValueError: If a loaded dataset is empty or lacks required columns.
    """
    input_files = {
        "users": PROCESSED_DIR / "users_validated.parquet",
        "movies": PROCESSED_DIR / "movies_validated.parquet",
        "ratings": PROCESSED_DIR / "ratings_validated.parquet",
    }

    required_columns = {
        "users": {"UserID"},
        "movies": {"MovieID", "Title", "Genres"},
        "ratings": {"UserID", "MovieID", "Rating", "Timestamp"},
    }

    loaded: dict[str, pd.DataFrame] = {}
    for dataset_name, file_path in input_files.items():
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Validated {dataset_name} dataset not found: {file_path}"
            )

        LOGGER.info("Loading validated %s dataset from %s", dataset_name, file_path)
        frame = pd.read_parquet(file_path)

        if frame.empty:
            raise ValueError(f"Validated {dataset_name} dataset is empty.")

        missing = required_columns[dataset_name] - set(frame.columns)
        if missing:
            raise ValueError(
                f"Validated {dataset_name} dataset is missing required columns: "
                f"{sorted(missing)}"
            )

        loaded[dataset_name] = frame
        LOGGER.info("Loaded %s records from %s", f"{len(frame):,}", dataset_name)

    return loaded["users"], loaded["movies"], loaded["ratings"]


# ---------------------------------------------------------------------------
# 4. INTERACTION PREPARATION AND TIMESTAMP TRANSFORMATION
# ---------------------------------------------------------------------------

def prepare_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps and add calendar attributes to interactions.

    MovieLens stores interaction timestamps as Unix seconds. This function
    converts them to timezone-aware UTC datetimes and adds year, month,
    day, and day-of-week columns for downstream analysis.

    Args:
        ratings: Validated ratings DataFrame.

    Returns:
        A copy of the ratings DataFrame with ``EventTimestamp``,
        ``EventDate``, ``Year``, ``Month``, ``Day``, and ``DayOfWeek``.

    Raises:
        ValueError: If any timestamp cannot be converted or is missing.
    """
    interactions = ratings.copy()

    event_timestamp = pd.to_datetime(
        interactions["Timestamp"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    invalid_timestamp_count = int(event_timestamp.isna().sum())
    if invalid_timestamp_count:
        raise ValueError(
            f"Found {invalid_timestamp_count} timestamps that could not be converted."
        )

    interactions["EventTimestamp"] = event_timestamp
    interactions["EventDate"] = event_timestamp.dt.strftime("%Y-%m-%d")
    interactions["Year"] = event_timestamp.dt.year.astype("int16")
    interactions["Month"] = event_timestamp.dt.month.astype("int8")
    interactions["Day"] = event_timestamp.dt.day.astype("int8")
    interactions["DayOfWeek"] = event_timestamp.dt.dayofweek.astype("int8")

    interactions = interactions.sort_values(
        ["UserID", "EventTimestamp", "MovieID"],
        kind="mergesort",
    ).reset_index(drop=True)

    LOGGER.info(
        "Prepared %s interactions; timestamp range: %s to %s",
        f"{len(interactions):,}",
        interactions["EventTimestamp"].min().isoformat(),
        interactions["EventTimestamp"].max().isoformat(),
    )
    return interactions


# ---------------------------------------------------------------------------
# 5. PER-USER CHRONOLOGICAL TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------

def chronological_user_split(
    interactions: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions chronologically into train and test.

    For every user, interactions are sorted by event timestamp and then
    MovieID. The earliest ``train_fraction`` portion is assigned to the
    training dataset; the most recent remaining interactions are assigned
    to the test dataset. Each user must have at least two interactions so
    that both partitions can contain records.

    Args:
        interactions: Prepared interaction DataFrame.
        train_fraction: Fraction of each user's interactions assigned to
            training. Must be greater than 0 and less than 1.

    Returns:
        A tuple ``(train_df, test_df)``.

    Raises:
        ValueError: If the fraction is invalid, a user has fewer than two
            interactions, or either output partition is empty.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be greater than 0 and less than 1.")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for user_id, user_events in interactions.groupby("UserID", sort=True):
        ordered = user_events.sort_values(
            ["EventTimestamp", "MovieID"], kind="mergesort"
        )
        interaction_count = len(ordered)

        if interaction_count < 2:
            raise ValueError(
                f"UserID {user_id} has fewer than two interactions; "
                "cannot place records in both train and test."
            )

        test_count = max(1, math.ceil(interaction_count * (1.0 - train_fraction)))
        train_count = interaction_count - test_count

        if train_count < 1:
            raise ValueError(
                f"UserID {user_id} does not have enough interactions for the split."
            )

        train_parts.append(ordered.iloc[:train_count])
        test_parts.append(ordered.iloc[train_count:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty partition.")

    # Confirm chronological ordering separately for every user.
    train_last = train_df.groupby("UserID")["EventTimestamp"].max()
    test_first = test_df.groupby("UserID")["EventTimestamp"].min()
    common_users = train_last.index.intersection(test_first.index)
    violations = train_last.loc[common_users] > test_first.loc[common_users]

    if bool(violations.any()):
        bad_users = violations[violations].index.tolist()[:10]
        raise ValueError(
            "Chronological split verification failed for users: "
            f"{bad_users}"
        )

    LOGGER.info(
        "Chronological split completed: %s train records, %s test records",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# 6. DERIVED DATASETS: USER ACTIVITY AND ITEM POPULARITY
# ---------------------------------------------------------------------------

def create_user_activity_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    """Create user activity and rating summaries from training data only.

    Only training interactions are used to calculate these values so that
    test-period information does not leak into training features.

    Args:
        train_df: Chronological training interactions.

    Returns:
        One row per user with activity count, average/minimum/maximum
        rating, first and last interaction timestamps, and active days.
    """
    summary = (
        train_df.groupby("UserID", as_index=False)
        .agg(
            user_activity_count=("MovieID", "size"),
            user_avg_rating=("Rating", "mean"),
            user_min_rating=("Rating", "min"),
            user_max_rating=("Rating", "max"),
            user_first_interaction=("EventTimestamp", "min"),
            user_last_interaction=("EventTimestamp", "max"),
            user_active_days=("EventDate", "nunique"),
        )
    )

    summary["user_avg_rating"] = summary["user_avg_rating"].round(4)
    summary["user_activity_count"] = summary["user_activity_count"].astype("int32")
    summary["user_active_days"] = summary["user_active_days"].astype("int32")

    LOGGER.info("Created user activity summary for %s users", f"{len(summary):,}")
    return summary


def create_item_popularity_summary(
    train_df: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Create item popularity and rating summaries from training data.

    Popularity is measured by the number of training interactions. The
    number of distinct users and average rating are also calculated.
    Movie titles and genres are joined from the validated movie catalogue.

    Args:
        train_df: Chronological training interactions.
        movies: Validated movie metadata.

    Returns:
        One row per movie appearing in training interactions, including
        title, genres, interaction count, unique-user count, average rating,
        and popularity rank.
    """
    summary = (
        train_df.groupby("MovieID", as_index=False)
        .agg(
            item_interaction_count=("UserID", "size"),
            item_unique_user_count=("UserID", "nunique"),
            item_avg_rating=("Rating", "mean"),
        )
    )

    summary["item_avg_rating"] = summary["item_avg_rating"].round(4)
    summary = summary.merge(
        movies[["MovieID", "Title", "Genres"]],
        on="MovieID",
        how="left",
        validate="one_to_one",
    )

    if summary[["Title", "Genres"]].isna().any().any():
        raise ValueError("Movie metadata join produced missing titles or genres.")

    summary = summary.sort_values(
        ["item_interaction_count", "item_avg_rating", "MovieID"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["popularity_rank"] = np.arange(1, len(summary) + 1, dtype=np.int32)
    summary["item_interaction_count"] = summary["item_interaction_count"].astype("int32")
    summary["item_unique_user_count"] = summary["item_unique_user_count"].astype("int32")

    LOGGER.info("Created popularity summary for %s movies", f"{len(summary):,}")
    return summary


# ---------------------------------------------------------------------------
# 7. SPARSE USER-ITEM INTERACTION MATRIX
# ---------------------------------------------------------------------------

def create_sparse_interaction_matrix(
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Create and save a CSR sparse user-item rating matrix.

    Rows represent users, columns represent movies, and non-zero values
    represent ratings from the training dataset. The associated index
    files preserve the mapping between matrix positions and original IDs.

    Args:
        train_df: Chronological training interactions.

    Returns:
        Dictionary with matrix shape, non-zero count, density, and output
        paths for the matrix and index files.

    Notes:
        A sparse matrix is used because most user-movie combinations have
        no rating. This avoids allocating a large mostly-empty dense array.
    """
    user_ids = np.sort(train_df["UserID"].unique())
    movie_ids = np.sort(train_df["MovieID"].unique())

    user_index = pd.DataFrame(
        {"matrix_index": np.arange(len(user_ids), dtype=np.int32), "UserID": user_ids}
    )
    item_index = pd.DataFrame(
        {"matrix_index": np.arange(len(movie_ids), dtype=np.int32), "MovieID": movie_ids}
    )

    row_positions = user_index.set_index("UserID").loc[train_df["UserID"], "matrix_index"].to_numpy()
    column_positions = item_index.set_index("MovieID").loc[train_df["MovieID"], "matrix_index"].to_numpy()
    rating_values = train_df["Rating"].to_numpy(dtype=np.float32)

    matrix = csr_matrix(
        (rating_values, (row_positions, column_positions)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )

    matrix_path = FEATURES_DIR / "user_item_interaction_matrix.npz"
    user_index_path = FEATURES_DIR / "user_index.csv"
    item_index_path = FEATURES_DIR / "item_index.csv"
    metadata_path = FEATURES_DIR / "interaction_matrix_metadata.json"

    save_npz(matrix_path, matrix, compressed=True)
    user_index.to_csv(user_index_path, index=False)
    item_index.to_csv(item_index_path, index=False)

    total_cells = int(matrix.shape[0] * matrix.shape[1])
    density = float(matrix.nnz / total_cells) if total_cells else 0.0

    metadata = {
        "matrix_type": "scipy.sparse.csr_matrix",
        "matrix_semantics": "training user-item ratings",
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "non_zero_interactions": int(matrix.nnz),
        "total_possible_user_item_pairs": total_cells,
        "density": density,
        "value_type": "rating (float32)",
        "matrix_file": str(matrix_path),
        "user_index_file": str(user_index_path),
        "item_index_file": str(item_index_path),
    }

    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    LOGGER.info(
        "Saved sparse user-item matrix with shape %s and %s non-zero entries "
        "(density %.6f)",
        matrix.shape,
        f"{matrix.nnz:,}",
        density,
    )

    return metadata


# ---------------------------------------------------------------------------
# 8. SUMMARY AND OUTPUT PERSISTENCE
# ---------------------------------------------------------------------------

def save_prepared_outputs(
    interactions: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_summary: pd.DataFrame,
    item_summary: pd.DataFrame,
) -> dict[str, str]:
    """Persist prepared interactions, train/test partitions and summaries.

    Args:
        interactions: All prepared interactions with timestamp features.
        train_df: Earlier interactions assigned to training.
        test_df: Most recent interactions assigned to testing.
        user_summary: Training-only user activity summary.
        item_summary: Training-only item popularity summary.

    Returns:
        Mapping of logical output names to saved Parquet file paths.
    """
    datasets = {
        "interactions_prepared": interactions,
        "train": train_df,
        "test": test_df,
        "user_activity_summary": user_summary,
        "item_popularity_summary": item_summary,
    }

    outputs: dict[str, str] = {}
    for name, frame in datasets.items():
        output_path = PROCESSED_DIR / f"{name}.parquet"
        frame.to_parquet(output_path, engine="pyarrow", index=False)
        outputs[name] = str(output_path)
        LOGGER.info("Saved %s records to %s", f"{len(frame):,}", output_path.name)

    return outputs


def create_preparation_summary(
    interactions: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outputs: dict[str, str],
    matrix_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create a JSON report describing preparation and transformation.

    Args:
        interactions: All prepared interactions.
        train_df: Training partition.
        test_df: Testing partition.
        outputs: Paths of generated Parquet files.
        matrix_metadata: Metadata for the saved sparse matrix.

    Returns:
        A JSON-serialisable dictionary containing split details,
        record counts, date ranges, and generated output paths.
    """
    user_train_last = train_df.groupby("UserID")["EventTimestamp"].max()
    user_test_first = test_df.groupby("UserID")["EventTimestamp"].min()
    shared_users = user_train_last.index.intersection(user_test_first.index)
    chronological_users = int(
        (user_train_last.loc[shared_users] <= user_test_first.loc[shared_users]).sum()
    )

    summary = {
        "project": "DM4ML Assignment 2 - Recommendation System Pipeline",
        "author": "Suraj Gate",
        "student_id": "2025ag05348",
        "dataset": "MovieLens 1M",
        "preparation_timestamp": datetime.now(timezone.utc).isoformat(),
        "split_method": "Per-user chronological holdout",
        "train_fraction_target": TRAIN_FRACTION,
        "test_fraction_target": TEST_FRACTION,
        "split_description": (
            "For each user, interactions were ordered by timestamp and MovieID. "
            "The earliest 80% were assigned to training and the latest 20% "
            "were assigned to testing; at least one event is held out per user."
        ),
        "record_counts": {
            "prepared_interactions": int(len(interactions)),
            "training_interactions": int(len(train_df)),
            "testing_interactions": int(len(test_df)),
            "users_in_training": int(train_df["UserID"].nunique()),
            "users_in_testing": int(test_df["UserID"].nunique()),
            "movies_in_training": int(train_df["MovieID"].nunique()),
            "movies_in_testing": int(test_df["MovieID"].nunique()),
        },
        "timestamp_range_utc": {
            "first_interaction": interactions["EventTimestamp"].min().isoformat(),
            "last_interaction": interactions["EventTimestamp"].max().isoformat(),
        },
        "chronological_split_check": {
            "users_checked": int(len(shared_users)),
            "users_with_train_before_test": chronological_users,
            "passed": bool(chronological_users == len(shared_users)),
        },
        "derived_datasets": {
            "user_activity_summary_records": int(train_df["UserID"].nunique()),
            "item_popularity_summary_records": int(train_df["MovieID"].nunique()),
        },
        "sparse_interaction_matrix": matrix_metadata,
        "outputs": outputs,
    }

    report_path = PREPARATION_REPORT_DIR / "preparation_summary.json"
    report_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    LOGGER.info("Preparation summary saved to %s", report_path)
    return summary


# ---------------------------------------------------------------------------
# 9. MAIN PREPARATION PIPELINE
# ---------------------------------------------------------------------------

def run_preparation() -> dict[str, Any]:
    """Run the complete data preparation and transformation workflow.

    The workflow loads validated source datasets, prepares interaction
    timestamps, performs a per-user chronological split, calculates
    training-only user and item summaries, saves Parquet outputs, creates
    a sparse user-item matrix, and writes a JSON preparation report.

    Returns:
        Preparation summary dictionary.

    Raises:
        FileNotFoundError: If validated input files are missing.
        ValueError: If schema, timestamp, split, or join checks fail.
        Exception: If a storage or processing operation fails.
    """
    LOGGER.info("=" * 72)
    LOGGER.info("STARTING DATA PREPARATION AND TRANSFORMATION")
    LOGGER.info("=" * 72)

    users, movies, ratings = load_validated_datasets()
    interactions = prepare_interactions(ratings)
    train_df, test_df = chronological_user_split(interactions)

    user_summary = create_user_activity_summary(train_df)
    item_summary = create_item_popularity_summary(train_df, movies)

    outputs = save_prepared_outputs(
        interactions=interactions,
        train_df=train_df,
        test_df=test_df,
        user_summary=user_summary,
        item_summary=item_summary,
    )

    matrix_metadata = create_sparse_interaction_matrix(train_df)
    summary = create_preparation_summary(
        interactions=interactions,
        train_df=train_df,
        test_df=test_df,
        outputs=outputs,
        matrix_metadata=matrix_metadata,
    )

    LOGGER.info("=" * 72)
    LOGGER.info("DATA PREPARATION AND TRANSFORMATION COMPLETED")
    LOGGER.info(
        "Prepared: %s | Train: %s | Test: %s",
        f"{len(interactions):,}",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
    )
    LOGGER.info("=" * 72)

    return summary


if __name__ == "__main__":
    run_preparation()
