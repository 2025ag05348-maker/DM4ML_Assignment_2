"""Build reusable user, item, and interaction features for MovieLens 1M.

This module implements the Feature Management stage of the DM4ML Assignment 2
recommendation pipeline. It reads only validated data and training-derived
aggregates when constructing historical features. Test interactions are used
only as evaluation examples/labels; their ratings are not used to calculate
historical user or item aggregates.

Inputs (created by earlier pipeline stages):
    data/processed/users_validated.parquet
    data/processed/movies_validated.parquet
    data/processed/train.parquet
    data/processed/test.parquet
    data/processed/user_activity_summary.parquet
    data/processed/item_popularity_summary.parquet

Outputs:
    data/features/user_features.parquet
    data/features/item_features.parquet
    data/features/interaction_features_train.parquet
    data/features/interaction_features_test.parquet
    data/features/feature_catalog.csv
    reports/features/feature_summary.json
    reports/features/feature_build.log

Feature design notes:
    * User and item aggregates are computed from training data only.
    * Training interaction history features use only earlier training events.
    * Test interaction history features use the training snapshot only.
    * Rating is retained as the target/label, not used as a predictor for the
      same interaction's historical features.

Author: Suraj Gate
BITS Student ID: 2025ag05348
Course: Data Management for Machine Learning (DM4ML)
Assignment: 2
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. PROJECT PATHS AND LOGGING
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports" / "features"
LOG_PATH = REPORTS_DIR / "feature_build.log"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2. INPUT / OUTPUT HELPERS
# ---------------------------------------------------------------------------

def read_parquet_input(file_name: str, required_columns: set[str]) -> pd.DataFrame:
    """Read one required Parquet input and validate its schema.

    Args:
        file_name: File name relative to ``data/processed``.
        required_columns: Column names that must exist in the loaded table.

    Returns:
        A DataFrame containing the requested Parquet data.

    Raises:
        FileNotFoundError: If the requested input does not exist.
        ValueError: If one or more required columns are missing.
    """
    path = PROCESSED_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")

    frame = pd.read_parquet(path)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Input {file_name} is missing required columns: {missing}. "
            f"Available columns: {frame.columns.tolist()}"
        )

    LOGGER.info("Loaded %s (%s rows)", file_name, f"{len(frame):,}")
    return frame


def save_parquet(frame: pd.DataFrame, file_name: str) -> Path:
    """Save a feature DataFrame as Parquet and return its path.

    Args:
        frame: DataFrame to persist.
        file_name: Output file name within ``data/features``.

    Returns:
        The absolute path of the written Parquet file.
    """
    output_path = FEATURES_DIR / file_name
    frame.to_parquet(output_path, engine="pyarrow", index=False)
    LOGGER.info("Saved %s (%s rows)", output_path.name, f"{len(frame):,}")
    return output_path


# ---------------------------------------------------------------------------
# 3. USER FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def derive_preferred_genre(
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Find each user's most frequently highly-rated genre from training data.

    Ratings of 4 or 5 are treated as positive preference signals. Movie genre
    strings can contain multiple pipe-separated genres, so interactions are
    expanded to one row per genre before counting. If a user has no positive
    training ratings, all of that user's training interactions are used as a
    fallback. Ties are resolved alphabetically for reproducibility.

    Args:
        train: Chronological training interactions.
        movies: Validated movie metadata with MovieID and Genres.

    Returns:
        One row per user with preferred_genre and user_genre_diversity.

    Raises:
        ValueError: If a training MovieID cannot be joined to movie metadata.
    """
    genre_lookup = movies[["MovieID", "Genres"]].copy()
    genre_lookup["Genres"] = genre_lookup["Genres"].fillna("")

    selected = train[["UserID", "MovieID", "Rating"]].merge(
        genre_lookup, on="MovieID", how="left", validate="many_to_one"
    )
    if selected["Genres"].isna().any():
        raise ValueError("Training interactions contain MovieIDs without genre metadata.")

    selected["Genre"] = selected["Genres"].str.split("|")
    expanded = selected.explode("Genre")
    expanded["Genre"] = expanded["Genre"].astype("string").str.strip()
    expanded = expanded[expanded["Genre"].ne("")]

    # Diversity is calculated over all training interactions, not only high ratings.
    diversity = (
        expanded.groupby("UserID", as_index=False)
        .agg(user_genre_diversity=("Genre", "nunique"))
    )

    positive = expanded[expanded["Rating"] >= 4]
    counts = (
        positive.groupby(["UserID", "Genre"], as_index=False)
        .size()
        .rename(columns={"size": "genre_interaction_count"})
    )

    # For users with no 4/5-star training rating, use all of their interactions.
    users_with_positive = set(counts["UserID"].unique())
    fallback = expanded[~expanded["UserID"].isin(users_with_positive)]
    if not fallback.empty:
        fallback_counts = (
            fallback.groupby(["UserID", "Genre"], as_index=False)
            .size()
            .rename(columns={"size": "genre_interaction_count"})
        )
        counts = pd.concat([counts, fallback_counts], ignore_index=True)

    counts = counts.sort_values(
        ["UserID", "genre_interaction_count", "Genre"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    preferred = counts.drop_duplicates("UserID", keep="first")[["UserID", "Genre"]]
    preferred = preferred.rename(columns={"Genre": "preferred_genre"})

    result = diversity.merge(preferred, on="UserID", how="left", validate="one_to_one")
    result["user_genre_diversity"] = result["user_genre_diversity"].astype("int16")
    return result


def build_user_features(
    users: pd.DataFrame,
    user_activity: pd.DataFrame,
    train: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Build one reusable feature row per user using training history.

    The feature table combines validated demographic attributes with activity
    aggregates and a preferred genre derived exclusively from training ratings.
    These features can be joined to user IDs during model training or scoring.

    Args:
        users: Validated user profile table.
        user_activity: Training-derived user activity summary.
        train: Training interactions only.
        movies: Validated movie metadata.

    Returns:
        User feature table with one row per UserID.

    Raises:
        ValueError: If duplicate user identifiers exist after feature joins.
    """
    preferred_genres = derive_preferred_genre(train, movies)
    user_rating_stats = (
        train.groupby("UserID", as_index=False)
        .agg(
            user_rating_std=("Rating", "std"),
            user_positive_rating_rate=("Rating", lambda values: float((values >= 4).mean())),
            user_unique_movie_count=("MovieID", "nunique"),
        )
    )

    features = users.merge(
        user_activity, on="UserID", how="left", validate="one_to_one"
    )
    features = features.merge(
        user_rating_stats, on="UserID", how="left", validate="one_to_one"
    )
    features = features.merge(
        preferred_genres, on="UserID", how="left", validate="one_to_one"
    )

    features["user_rating_std"] = features["user_rating_std"].fillna(0.0).round(4)
    features["user_positive_rating_rate"] = (
        features["user_positive_rating_rate"].fillna(0.0).round(4)
    )
    features["user_unique_movie_count"] = (
        features["user_unique_movie_count"].fillna(0).astype("int32")
    )
    features["user_genre_diversity"] = (
        features["user_genre_diversity"].fillna(0).astype("int16")
    )
    features["preferred_genre"] = features["preferred_genre"].fillna("Unknown")

    if features["UserID"].duplicated().any():
        raise ValueError("User feature table contains duplicate UserID values.")
    if len(features) != users["UserID"].nunique():
        raise ValueError("User feature row count does not match validated user count.")

    LOGGER.info("Built user features for %s users", f"{len(features):,}")
    return features


# ---------------------------------------------------------------------------
# 4. ITEM FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_item_features(
    movies: pd.DataFrame,
    item_popularity: pd.DataFrame,
) -> pd.DataFrame:
    """Build one reusable feature row per movie.

    The output combines validated movie metadata with popularity, distinct
    user count and average-rating features calculated from training data only.
    It also derives release year and genre count from the movie metadata.

    Args:
        movies: Validated movie catalogue.
        item_popularity: Training-derived item popularity summary.

    Returns:
        Item feature table with one row per MovieID.

    Raises:
        ValueError: If the metadata join creates duplicate IDs or missing titles.
    """
    features = movies.merge(
        item_popularity[
            [
                "MovieID",
                "item_interaction_count",
                "item_unique_user_count",
                "item_avg_rating",
                "popularity_rank",
            ]
        ],
        on="MovieID",
        how="left",
        validate="one_to_one",
    )

    features["item_interaction_count"] = (
        features["item_interaction_count"].fillna(0).astype("int32")
    )
    features["item_unique_user_count"] = (
        features["item_unique_user_count"].fillna(0).astype("int32")
    )
    features["item_avg_rating"] = features["item_avg_rating"].fillna(0.0).round(4)
    features["popularity_rank"] = features["popularity_rank"].fillna(0).astype("int32")

    # MovieLens titles commonly end in a four-digit year in parentheses.
    extracted_year = features["Title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    features["movie_year"] = pd.to_numeric(extracted_year, errors="coerce").astype("Int16")
    features["genre_count"] = (
        features["Genres"].fillna("").apply(
            lambda value: 0 if not value else len(value.split("|"))
        ).astype("int8")
    )
    features["is_in_training_history"] = features["item_interaction_count"].gt(0)

    if features["MovieID"].duplicated().any():
        raise ValueError("Item feature table contains duplicate MovieID values.")
    if features["Title"].isna().any():
        raise ValueError("Item feature table contains missing movie titles.")

    LOGGER.info("Built item features for %s movies", f"{len(features):,}")
    return features


# ---------------------------------------------------------------------------
# 5. INTERACTION-LEVEL HISTORICAL FEATURES
# ---------------------------------------------------------------------------

def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features from the already prepared event timestamp.

    Args:
        frame: DataFrame containing EventTimestamp.

    Returns:
        A copy with event_year, event_month, event_day and event_day_of_week.
    """
    result = frame.copy()
    event_time = pd.to_datetime(result["EventTimestamp"], utc=True, errors="coerce")
    if event_time.isna().any():
        raise ValueError("Interaction feature input has invalid EventTimestamp values.")
    result["event_year"] = event_time.dt.year.astype("int16")
    result["event_month"] = event_time.dt.month.astype("int8")
    result["event_day"] = event_time.dt.day.astype("int8")
    result["event_day_of_week"] = event_time.dt.dayofweek.astype("int8")
    return result


def build_training_interaction_features(train: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-aware historical features for training interactions.

    For each training event, user and item history counts/average ratings are
    calculated from events that occurred strictly earlier in the training order.
    The current row's rating is retained as ``target_rating`` and is excluded
    from its own historical aggregates.

    Args:
        train: Prepared training interactions with UserID, MovieID, Rating,
            and EventTimestamp.

    Returns:
        Training interaction features, including historical counts, averages,
        time since prior user event, calendar fields and target_rating.
    """
    required = {"UserID", "MovieID", "Rating", "EventTimestamp"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Training interactions missing columns: {missing}")

    frame = train.copy().sort_values(
        ["UserID", "EventTimestamp", "MovieID"], kind="mergesort"
    ).reset_index(drop=True)
    frame["target_rating"] = frame["Rating"].astype("float32")

    # Previous user history: cumulative aggregates exclude the current row.
    frame["user_history_count_before_event"] = (
        frame.groupby("UserID", sort=False).cumcount().astype("int32")
    )
    user_cumulative_sum = frame.groupby("UserID", sort=False)["Rating"].cumsum()
    prior_user_sum = user_cumulative_sum - frame["Rating"]
    frame["user_history_avg_rating_before_event"] = np.where(
        frame["user_history_count_before_event"] > 0,
        prior_user_sum / frame["user_history_count_before_event"].replace(0, np.nan),
        np.nan,
    )
    frame["days_since_previous_user_interaction"] = (
        frame.groupby("UserID", sort=False)["EventTimestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
    )

    # Previous item history, independently ordered by movie and event time.
    item_order = frame.sort_values(
        ["MovieID", "EventTimestamp", "UserID"], kind="mergesort"
    ).copy()
    item_order["item_history_count_before_event"] = (
        item_order.groupby("MovieID", sort=False).cumcount().astype("int32")
    )
    item_cumulative_sum = item_order.groupby("MovieID", sort=False)["Rating"].cumsum()
    prior_item_sum = item_cumulative_sum - item_order["Rating"]
    item_order["item_history_avg_rating_before_event"] = np.where(
        item_order["item_history_count_before_event"] > 0,
        prior_item_sum / item_order["item_history_count_before_event"].replace(0, np.nan),
        np.nan,
    )

    item_history = item_order[
        [
            "UserID",
            "MovieID",
            "EventTimestamp",
            "item_history_count_before_event",
            "item_history_avg_rating_before_event",
        ]
    ]
    frame = frame.merge(
        item_history,
        on=["UserID", "MovieID", "EventTimestamp"],
        how="left",
        validate="one_to_one",
    )
    frame = _add_calendar_features(frame)

    # Retain the label separately and remove raw Rating to discourage its use
    # as an input feature by downstream training code.
    frame = frame.drop(columns=["Rating"])
    frame["item_history_avg_rating_before_event"] = (
        frame["item_history_avg_rating_before_event"].astype("float32")
    )
    frame["user_history_avg_rating_before_event"] = (
        frame["user_history_avg_rating_before_event"].astype("float32")
    )

    LOGGER.info("Built historical features for %s training interactions", f"{len(frame):,}")
    return frame


def build_test_interaction_features(
    test: pd.DataFrame,
    train: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build test-time interaction features from the training snapshot only.

    Each test interaction is represented as a future recommendation request.
    Its features include the user's last training activity, training history
    count/average rating, and the item's training popularity/average rating.
    No test rating or other test-period interaction is used in these features.

    Args:
        test: Chronological test interactions; Rating is the evaluation target.
        train: Training interactions only.
        user_features: Training-derived user feature table.
        item_features: Training-derived item feature table.

    Returns:
        Test interactions with training-snapshot features and target_rating.
    """
    result = test.copy()
    result["target_rating"] = result["Rating"].astype("float32")

    user_snapshot = train.groupby("UserID", as_index=False).agg(
        user_history_count_before_prediction=("MovieID", "size"),
        user_history_avg_rating_before_prediction=("Rating", "mean"),
        user_last_training_interaction=("EventTimestamp", "max"),
    )
    item_snapshot = train.groupby("MovieID", as_index=False).agg(
        item_history_count_before_prediction=("UserID", "size"),
        item_history_avg_rating_before_prediction=("Rating", "mean"),
    )

    result = result.merge(
        user_snapshot, on="UserID", how="left", validate="many_to_one"
    )
    result = result.merge(
        item_snapshot, on="MovieID", how="left", validate="many_to_one"
    )
    result = result.merge(
        user_features[["UserID", "preferred_genre", "user_genre_diversity"]],
        on="UserID", how="left", validate="many_to_one"
    )
    result = result.merge(
        item_features[["MovieID", "Title", "Genres", "movie_year", "genre_count"]],
        on="MovieID", how="left", validate="many_to_one"
    )

    result["days_since_last_training_interaction"] = (
        pd.to_datetime(result["EventTimestamp"], utc=True)
        - pd.to_datetime(result["user_last_training_interaction"], utc=True)
    ).dt.total_seconds().div(86400.0)
    result["user_history_avg_rating_before_prediction"] = (
        result["user_history_avg_rating_before_prediction"].astype("float32")
    )
    result["item_history_avg_rating_before_prediction"] = (
        result["item_history_avg_rating_before_prediction"].astype("float32")
    )
    result = _add_calendar_features(result)
    result = result.drop(columns=["Rating"])

    # Movies first seen only in the test partition are retained as cold-start
    # evaluation cases; their training popularity is represented as zero.
    result["item_history_count_before_prediction"] = (
        result["item_history_count_before_prediction"].fillna(0).astype("int32")
    )
    result["item_history_avg_rating_before_prediction"] = (
        result["item_history_avg_rating_before_prediction"].fillna(0.0).astype("float32")
    )
    result["preferred_genre"] = result["preferred_genre"].fillna("Unknown")
    result["user_genre_diversity"] = result["user_genre_diversity"].fillna(0).astype("int16")

    LOGGER.info("Built historical features for %s test interactions", f"{len(result):,}")
    return result


# ---------------------------------------------------------------------------
# 6. FEATURE CATALOG
# ---------------------------------------------------------------------------

def build_feature_catalog() -> pd.DataFrame:
    """Create a documented catalog of reusable pipeline features.

    Returns:
        A table describing feature name, entity, type, definition, source,
        and whether the feature is based on training data only.
    """
    entries = [
        ("Age", "user", "int", "MovieLens age-group code for the user.", "users_validated", "profile"),
        ("Gender", "user", "string", "MovieLens gender value for the user.", "users_validated", "profile"),
        ("Occupation", "user", "int", "MovieLens occupation code for the user.", "users_validated", "profile"),
        ("user_activity_count", "user", "int", "Number of training interactions for the user.", "train", "training_only"),
        ("user_avg_rating", "user", "float", "Mean rating provided by the user in training data.", "train", "training_only"),
        ("user_positive_rating_rate", "user", "float", "Share of training ratings greater than or equal to 4.", "train", "training_only"),
        ("preferred_genre", "user", "string", "Most frequent genre among positive training ratings; fallback to all training interactions.", "train + movies_validated", "training_only"),
        ("user_genre_diversity", "user", "int", "Number of distinct genres in the user's training history.", "train + movies_validated", "training_only"),
        ("Title", "item", "string", "Movie title from the validated movie catalogue.", "movies_validated", "metadata"),
        ("Genres", "item", "string", "Pipe-separated movie genres.", "movies_validated", "metadata"),
        ("movie_year", "item", "nullable int", "Four-digit release year parsed from the movie title when available.", "movies_validated", "metadata"),
        ("item_interaction_count", "item", "int", "Number of training interactions for the movie.", "train", "training_only"),
        ("item_unique_user_count", "item", "int", "Number of distinct training users who rated the movie.", "train", "training_only"),
        ("item_avg_rating", "item", "float", "Mean training rating received by the movie.", "train", "training_only"),
        ("popularity_rank", "item", "int", "Rank by training interaction count, with deterministic tie-breaking.", "train", "training_only"),
        ("user_history_count_before_event", "interaction", "int", "Number of earlier training interactions by the user.", "train history", "past_only"),
        ("user_history_avg_rating_before_event", "interaction", "float", "Mean rating of the user's earlier training interactions.", "train history", "past_only"),
        ("days_since_previous_user_interaction", "interaction", "float", "Days between this event and the user's preceding training event.", "train history", "past_only"),
        ("item_history_count_before_event", "interaction", "int", "Number of earlier training interactions for the movie.", "train history", "past_only"),
        ("item_history_avg_rating_before_event", "interaction", "float", "Mean rating from earlier training interactions for the movie.", "train history", "past_only"),
        ("days_since_last_training_interaction", "interaction", "float", "Days from the user's last training event to the test event timestamp.", "train snapshot + test event time", "training_snapshot_only"),
        ("event_month", "interaction", "int", "Calendar month extracted from the event timestamp in UTC.", "EventTimestamp", "event_time"),
        ("event_day_of_week", "interaction", "int", "Day of week extracted from the event timestamp in UTC; Monday is 0.", "EventTimestamp", "event_time"),
        ("target_rating", "label", "float", "Observed rating retained as the supervised/evaluation target, not as a predictor.", "ratings", "label_only"),
    ]
    columns = [
        "feature_name",
        "entity",
        "data_type",
        "description",
        "source",
        "availability",
    ]
    return pd.DataFrame(entries, columns=columns)


# ---------------------------------------------------------------------------
# 7. PIPELINE SUMMARY AND EXECUTION
# ---------------------------------------------------------------------------

def run_feature_management() -> dict[str, Any]:
    """Build user, item, and interaction features and persist all artifacts.

    Returns:
        A summary dictionary containing entity row counts, interaction
        feature row counts, output paths, and feature catalog information.

    Raises:
        FileNotFoundError: If any required upstream Parquet file is absent.
        ValueError: If required columns or unique-key expectations fail.
    """
    LOGGER.info("Starting DM4ML feature management stage")

    users = read_parquet_input("users_validated.parquet", {"UserID", "Gender", "Age", "Occupation"})
    movies = read_parquet_input("movies_validated.parquet", {"MovieID", "Title", "Genres"})
    train = read_parquet_input("train.parquet", {"UserID", "MovieID", "Rating", "EventTimestamp"})
    test = read_parquet_input("test.parquet", {"UserID", "MovieID", "Rating", "EventTimestamp"})
    user_activity = read_parquet_input("user_activity_summary.parquet", {"UserID", "user_activity_count", "user_avg_rating"})
    item_popularity = read_parquet_input("item_popularity_summary.parquet", {"MovieID", "item_interaction_count", "item_avg_rating", "popularity_rank"})

    user_features = build_user_features(users, user_activity, train, movies)
    item_features = build_item_features(movies, item_popularity)
    train_interaction_features = build_training_interaction_features(train)
    test_interaction_features = build_test_interaction_features(
        test, train, user_features, item_features
    )

    catalog = build_feature_catalog()

    output_paths = {
        "user_features": save_parquet(user_features, "user_features.parquet"),
        "item_features": save_parquet(item_features, "item_features.parquet"),
        "interaction_features_train": save_parquet(
            train_interaction_features, "interaction_features_train.parquet"
        ),
        "interaction_features_test": save_parquet(
            test_interaction_features, "interaction_features_test.parquet"
        ),
    }

    catalog_path = FEATURES_DIR / "feature_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    LOGGER.info("Saved feature catalog: %s", catalog_path.name)

    summary: dict[str, Any] = {
        "project": "DM4ML Assignment 2 - Recommendation System Pipeline",
        "author": "Suraj Gate",
        "student_id": "2025ag05348",
        "dataset": "MovieLens 1M",
        "feature_build_timestamp": datetime.now(timezone.utc).isoformat(),
        "leakage_controls": {
            "user_and_item_aggregates_use_training_data_only": True,
            "training_interaction_history_excludes_current_event": True,
            "test_interaction_features_use_training_snapshot_only": True,
            "observed_rating_is_stored_as_target_not_predictor": True,
        },
        "record_counts": {
            "user_features": int(len(user_features)),
            "item_features": int(len(item_features)),
            "training_interaction_features": int(len(train_interaction_features)),
            "test_interaction_features": int(len(test_interaction_features)),
            "feature_catalog_entries": int(len(catalog)),
        },
        "cold_start": {
            "test_interactions_for_items_absent_from_training": int(
                test_interaction_features["item_history_count_before_prediction"].eq(0).sum()
            ),
            "note": "Test-only items are retained; their training popularity and history rating are represented as zero.",
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "feature_catalog": str(catalog_path),
    }

    summary_path = REPORTS_DIR / "feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    LOGGER.info("Feature summary saved to %s", summary_path)
    LOGGER.info("Feature management completed successfully")
    return summary


def main() -> None:
    """Run feature management and print a concise execution summary."""
    summary = run_feature_management()
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
