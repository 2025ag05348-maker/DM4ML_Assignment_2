"""Automated data-quality validation for the MovieLens 1M pipeline.

This module validates the ingested users, movies, and ratings datasets.
It checks required values, data types, duplicate business keys, domain
constraints, timestamps, and interaction referential integrity. Invalid
rows are written to data/errors, while valid rows are written to
 data/processed as Parquet files. A machine-readable validation report
is generated under reports/validation.

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

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype


# ---------------------------------------------------------------------------
# Project paths and logging
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ERROR_DIR = PROJECT_ROOT / "data" / "errors"
VALIDATION_REPORT_DIR = PROJECT_ROOT / "reports" / "validation"

ERROR_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = VALIDATION_REPORT_DIR / "validation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic validation helpers
# ---------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """Ensure that a DataFrame contains every required column.

    Args:
        df: Dataset being inspected.
        required_columns: Column names required by the pipeline.
        dataset_name: Human-readable dataset name for error messages.

    Raises:
        ValueError: If one or more required columns are absent.
    """
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def missing_required_values_mask(
    df: pd.DataFrame,
    required_columns: list[str],
    string_columns: list[str],
) -> pd.Series:
    """Return a row-level mask for null or blank required values."""
    mask = df[required_columns].isna().any(axis=1)
    for column in string_columns:
        blank_mask = df[column].astype("string").str.strip().eq("").fillna(False)
        mask = mask | blank_mask
    return mask.fillna(True)


def invalid_integer_mask(series: pd.Series) -> pd.Series:
    """Identify values that cannot be represented as whole numbers."""
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric.isna() | numeric.mod(1).ne(0)).fillna(True)


def invalid_string_type_mask(series: pd.Series) -> pd.Series:
    """Identify non-string values in columns expected to contain text."""
    return (~series.map(lambda value: isinstance(value, str))).fillna(True)


def schema_dtype_report(
    df: pd.DataFrame,
    expected_types: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Report actual pandas dtypes against the expected logical types.

    The row-level validation rules separately identify values that cannot
    be converted to the required types. This report makes the DataFrame
    schema check explicit for the assignment evidence.
    """
    report: dict[str, dict[str, Any]] = {}
    for column, expected_type in expected_types.items():
        dtype = df[column].dtype
        if expected_type == "integer":
            passed = bool(is_integer_dtype(dtype))
        elif expected_type == "string":
            passed = bool(is_string_dtype(dtype))
        else:
            raise ValueError(f"Unsupported expected type: {expected_type}")
        report[column] = {
            "expected_type": expected_type,
            "actual_dtype": str(dtype),
            "passed": passed,
        }
    return report


def split_valid_and_invalid_rows(
    df: pd.DataFrame,
    checks: list[tuple[str, pd.Series]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split a dataset into valid and invalid rows and attach error reasons.

    Args:
        df: Original DataFrame to validate.
        checks: Pairs of a human-readable rule name and a Boolean row mask.

    Returns:
        A tuple containing valid rows, invalid rows, and per-rule violation
        counts. Violation counts may overlap when a row fails multiple rules;
        invalid row count is the number of unique rejected rows.
    """
    reasons = pd.Series("", index=df.index, dtype="string")
    violation_counts: dict[str, int] = {}

    for rule_name, raw_mask in checks:
        mask = pd.Series(raw_mask, index=df.index).reindex(df.index)
        mask = mask.fillna(True).astype(bool)
        violation_counts[rule_name] = int(mask.sum())

        if mask.any():
            current = reasons.loc[mask].fillna("")
            separator = current.ne("").map({True: "; ", False: ""})
            reasons.loc[mask] = current + separator + rule_name

    invalid_mask = reasons.ne("")
    valid_df = df.loc[~invalid_mask].copy()
    invalid_df = df.loc[invalid_mask].copy()
    invalid_df["error_reasons"] = reasons.loc[invalid_mask].to_numpy()

    return valid_df, invalid_df, violation_counts


def validate_integer_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> list[tuple[str, pd.Series]]:
    """Build row-level checks for integer-valued columns."""
    return [
        (f"invalid_integer:{column}", invalid_integer_mask(df[column]))
        for column in columns
    ]


def validate_string_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> list[tuple[str, pd.Series]]:
    """Build row-level checks for columns that must contain text."""
    return [
        (f"invalid_string_type:{column}", invalid_string_type_mask(df[column]))
        for column in columns
    ]


# ---------------------------------------------------------------------------
# Dataset-specific validation
# ---------------------------------------------------------------------------

def validate_users(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate the MovieLens user dimension.

    Checks required fields, logical types, unique user identifiers, known
    MovieLens age-group codes, valid gender values, and occupation codes.
    """
    required = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    require_columns(df, required, "users")

    checks: list[tuple[str, pd.Series]] = [
        (
            "missing_required_values",
            missing_required_values_mask(df, required, ["Gender", "Zip-code"]),
        ),
    ]
    checks.extend(validate_integer_columns(df, ["UserID", "Age", "Occupation"]))
    checks.extend(validate_string_columns(df, ["Gender", "Zip-code"]))
    checks.extend([
        ("duplicate_UserID", df["UserID"].duplicated(keep=False)),
        ("invalid_gender", ~df["Gender"].isin(["F", "M"])),
        ("invalid_age_group", ~pd.to_numeric(df["Age"], errors="coerce").isin([1, 18, 25, 35, 45, 50, 56])),
        (
            "invalid_occupation_code",
            ~pd.to_numeric(df["Occupation"], errors="coerce").between(0, 20),
        ),
    ])

    valid, invalid, counts = split_valid_and_invalid_rows(df, checks)
    schema = schema_dtype_report(
        df,
        {
            "UserID": "integer",
            "Gender": "string",
            "Age": "integer",
            "Occupation": "integer",
            "Zip-code": "string",
        },
    )
    return valid, invalid, {"rule_violation_counts": counts, "schema_checks": schema}


def validate_movies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate the MovieLens movie/item dimension."""
    required = ["MovieID", "Title", "Genres"]
    require_columns(df, required, "movies")

    checks: list[tuple[str, pd.Series]] = [
        (
            "missing_required_values",
            missing_required_values_mask(df, required, ["Title", "Genres"]),
        ),
    ]
    checks.extend(validate_integer_columns(df, ["MovieID"]))
    checks.extend(validate_string_columns(df, ["Title", "Genres"]))
    checks.append(("duplicate_MovieID", df["MovieID"].duplicated(keep=False)))

    valid, invalid, counts = split_valid_and_invalid_rows(df, checks)
    schema = schema_dtype_report(
        df,
        {"MovieID": "integer", "Title": "string", "Genres": "string"},
    )
    return valid, invalid, {"rule_violation_counts": counts, "schema_checks": schema}


def validate_ratings(
    df: pd.DataFrame,
    valid_user_ids: set[int],
    valid_movie_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate ratings and their references to valid users and movies.

    Ratings must be integer values from 1 through 5. Timestamps must be
    positive Unix epoch seconds that pandas can convert to UTC datetimes
    and must not be in the future. Each user/movie pair must be unique.
    """
    required = ["UserID", "MovieID", "Rating", "Timestamp"]
    require_columns(df, required, "ratings")

    numeric_timestamp = pd.to_numeric(df["Timestamp"], errors="coerce")
    timestamp_datetime = pd.to_datetime(
        numeric_timestamp,
        unit="s",
        errors="coerce",
        utc=True,
    )
    now_utc = pd.Timestamp.now(tz="UTC")

    numeric_rating = pd.to_numeric(df["Rating"], errors="coerce")
    numeric_user_id = pd.to_numeric(df["UserID"], errors="coerce")
    numeric_movie_id = pd.to_numeric(df["MovieID"], errors="coerce")

    checks: list[tuple[str, pd.Series]] = [
        ("missing_required_values", df[required].isna().any(axis=1)),
    ]
    checks.extend(validate_integer_columns(df, required))
    checks.extend([
        ("invalid_rating_range", ~numeric_rating.between(1, 5)),
        (
            "invalid_timestamp",
            numeric_timestamp.isna()
            | numeric_timestamp.le(0)
            | timestamp_datetime.isna()
            | timestamp_datetime.gt(now_utc),
        ),
        (
            "duplicate_user_movie_interaction",
            df.duplicated(subset=["UserID", "MovieID"], keep=False),
        ),
        (
            "unknown_UserID_referential_integrity",
            ~numeric_user_id.isin(valid_user_ids),
        ),
        (
            "unknown_MovieID_referential_integrity",
            ~numeric_movie_id.isin(valid_movie_ids),
        ),
    ])

    valid, invalid, counts = split_valid_and_invalid_rows(df, checks)
    schema = schema_dtype_report(
        df,
        {
            "UserID": "integer",
            "MovieID": "integer",
            "Rating": "integer",
            "Timestamp": "integer",
        },
    )
    return valid, invalid, {"rule_violation_counts": counts, "schema_checks": schema}


# ---------------------------------------------------------------------------
# Persistence and reporting
# ---------------------------------------------------------------------------

def save_validation_outputs(
    dataset_name: str,
    valid_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
) -> dict[str, str]:
    """Persist validated Parquet data and rejected rows as CSV.

    Error CSV files are created even when no invalid rows exist. This makes
    the pipeline output predictable and shows that the error-handling stage
    was executed.
    """
    valid_path = PROCESSED_DIR / f"{dataset_name}_validated.parquet"
    error_path = ERROR_DIR / f"{dataset_name}_validation_errors.csv"

    valid_df.to_parquet(valid_path, index=False, engine="pyarrow")
    invalid_df.to_csv(error_path, index=False, encoding="utf-8")

    logger.info(
        "%s validation output: %s valid rows, %s invalid rows",
        dataset_name,
        f"{len(valid_df):,}",
        f"{len(invalid_df):,}",
    )

    return {"validated_file": str(valid_path), "error_file": str(error_path)}


def build_dataset_report(
    source_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
    validation_details: dict[str, Any],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """Build a JSON-serialisable summary for one validated dataset."""
    return {
        "source_records": int(len(source_df)),
        "valid_records": int(len(valid_df)),
        "invalid_records": int(len(invalid_df)),
        "rule_violation_counts": validation_details["rule_violation_counts"],
        "schema_checks": validation_details["schema_checks"],
        "all_schema_checks_passed": all(
            check["passed"]
            for check in validation_details["schema_checks"].values()
        ),
        "outputs": output_paths,
    }


def run_validation() -> dict[str, Any]:
    """Run all MovieLens data-quality checks and write their outputs.

    Processing order:
        1. Read ingested users and movies.
        2. Validate both dimensions.
        3. Build reference sets from valid users and movies.
        4. Validate ratings, including referential integrity.
        5. Save valid datasets, error datasets, and a JSON report.

    Returns:
        A summary dictionary containing record counts, validation rules,
        schema checks, and output file locations.
    """
    logger.info("Starting MovieLens data validation")

    input_paths = {
        "users": PROCESSED_DIR / "users_ingested.parquet",
        "movies": PROCESSED_DIR / "movies_ingested.parquet",
        "ratings": PROCESSED_DIR / "ratings_ingested.parquet",
    }
    for dataset_name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing ingested {dataset_name} dataset: {path}. "
                "Run the ingestion module first."
            )

    users_source = pd.read_parquet(input_paths["users"])
    movies_source = pd.read_parquet(input_paths["movies"])
    ratings_source = pd.read_parquet(input_paths["ratings"])

    users_valid, users_invalid, users_details = validate_users(users_source)
    movies_valid, movies_invalid, movies_details = validate_movies(movies_source)

    valid_user_ids = set(pd.to_numeric(users_valid["UserID"], errors="coerce").dropna().astype(int))
    valid_movie_ids = set(pd.to_numeric(movies_valid["MovieID"], errors="coerce").dropna().astype(int))

    ratings_valid, ratings_invalid, ratings_details = validate_ratings(
        ratings_source,
        valid_user_ids=valid_user_ids,
        valid_movie_ids=valid_movie_ids,
    )

    users_outputs = save_validation_outputs("users", users_valid, users_invalid)
    movies_outputs = save_validation_outputs("movies", movies_valid, movies_invalid)
    ratings_outputs = save_validation_outputs("ratings", ratings_valid, ratings_invalid)

    report = {
        "project": "DM4ML Assignment 2 - Recommendation System Pipeline",
        "author": "Suraj Gate",
        "student_id": "2025ag05348",
        "dataset": "MovieLens 1M",
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Violation counts can overlap because one row may fail multiple checks.",
            "invalid_records counts unique rows rejected by at least one rule.",
            "Referential integrity is checked against users and movies that pass validation.",
        ],
        "datasets": {
            "users": build_dataset_report(
                users_source, users_valid, users_invalid, users_details, users_outputs
            ),
            "movies": build_dataset_report(
                movies_source, movies_valid, movies_invalid, movies_details, movies_outputs
            ),
            "ratings": build_dataset_report(
                ratings_source, ratings_valid, ratings_invalid, ratings_details, ratings_outputs
            ),
        },
    }

    report_path = VALIDATION_REPORT_DIR / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    logger.info("Validation report saved to %s", report_path)
    logger.info("Data validation completed successfully")

    for name, details in report["datasets"].items():
        logger.info(
            "%s: %s valid / %s invalid / %s source records",
            name.upper(),
            f"{details['valid_records']:,}",
            f"{details['invalid_records']:,}",
            f"{details['source_records']:,}",
        )

    return report


if __name__ == "__main__":
    run_validation()
