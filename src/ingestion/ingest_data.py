
"""
Module: ingest_data.py

Purpose:
    Ingest the MovieLens 1M raw dataset into the recommendation
    pipeline and store the loaded datasets in Parquet format.

Dataset:
    MovieLens 1M

Input files:
    1. users.dat
    2. movies.dat
    3. ratings.dat

Output files:
    1. users_ingested.parquet
    2. movies_ingested.parquet
    3. ratings_ingested.parquet
    4. ingestion_summary.json

Author: Suraj Gate
BITS Student ID: 2025ag05348
Course: Data Management for Machine Learning (DM4ML)
Assignment: 2
"""

import json
import logging

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-1m"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

LOG_DIR = PROJECT_ROOT / "reports"
LOG_FILE = LOG_DIR / "ingestion.log"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOGGING CONFIGURATION
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 3. DATASET SCHEMAS
# ---------------------------------------------------------

USERS_COLUMNS = [
    "UserID",
    "Gender",
    "Age",
    "Occupation",
    "Zip-code",
]

MOVIES_COLUMNS = [
    "MovieID",
    "Title",
    "Genres",
]

RATINGS_COLUMNS = [
    "UserID",
    "MovieID",
    "Rating",
    "Timestamp",
]


# ---------------------------------------------------------
# 4. COMMON DATA LOADING FUNCTION
# ---------------------------------------------------------

def read_movielens_file(
    file_path: Path,
    columns: list[str],
    data_types: dict[str, Any],
) -> pd.DataFrame:
    """
    Read a MovieLens .dat file into a pandas DataFrame.

    MovieLens 1M uses double-colon (::) as its delimiter.
    The original files do not contain a header row.

    Args:
        file_path:
            Absolute or relative path to the source .dat file.

        columns:
            Column names to assign to the loaded DataFrame.

        data_types:
            Mapping of column names to their intended pandas
            data types.

    Returns:
        pd.DataFrame:
            Loaded dataset with the specified columns and types.

    Raises:
        FileNotFoundError:
            If the specified source file does not exist.

        ValueError:
            If the file is empty or does not match the expected
            number of columns.

        Exception:
            If pandas encounters an unexpected loading error.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset file not found: {file_path}"
        )

    logger.info("Reading source file: %s", file_path.name)

    try:
        df = pd.read_csv(
            file_path,
            sep="::",
            engine="python",
            header=None,
            names=columns,
            dtype=data_types,
            encoding="latin-1",
        )

        if df.empty:
            raise ValueError(
                f"The source file contains no records: {file_path}"
            )

        if df.shape[1] != len(columns):
            raise ValueError(
                f"Unexpected column count in {file_path.name}. "
                f"Expected {len(columns)}, received {df.shape[1]}."
            )

        logger.info(
            "Successfully loaded %s records from %s",
            f"{len(df):,}",
            file_path.name,
        )

        return df

    except Exception:
        logger.exception(
            "Failed to load source file: %s",
            file_path.name,
        )
        raise


# ---------------------------------------------------------
# 5. INGEST USERS DATA
# ---------------------------------------------------------

def ingest_users() -> pd.DataFrame:
    """
    Load the MovieLens users dataset.

    The users dataset contains demographic information about
    users registered in the MovieLens system.

    Returns:
        pd.DataFrame:
            User information including ID, gender, age group,
            occupation and ZIP code.
    """

    file_path = RAW_DATA_DIR / "users.dat"

    data_types = {
        "UserID": "int32",
        "Gender": "string",
        "Age": "int8",
        "Occupation": "int8",
        "Zip-code": "string",
    }

    return read_movielens_file(
        file_path,
        USERS_COLUMNS,
        data_types,
    )


# ---------------------------------------------------------
# 6. INGEST MOVIES DATA
# ---------------------------------------------------------

def ingest_movies() -> pd.DataFrame:
    """
    Load the MovieLens movie catalogue.

    The movies dataset contains movie identifiers, titles
    and genre information.

    Returns:
        pd.DataFrame:
            Movie information including MovieID, Title and Genres.
    """

    file_path = RAW_DATA_DIR / "movies.dat"

    data_types = {
        "MovieID": "int32",
        "Title": "string",
        "Genres": "string",
    }

    return read_movielens_file(
        file_path,
        MOVIES_COLUMNS,
        data_types,
    )


# ---------------------------------------------------------
# 7. INGEST RATINGS DATA
# ---------------------------------------------------------

def ingest_ratings() -> pd.DataFrame:
    """
    Load the MovieLens user-movie interactions dataset.

    Each record represents a rating given by a user to a movie.

    The original timestamp is preserved as a Unix timestamp.
    Its conversion into a datetime field will be performed
    during the preparation and transformation stages.

    Returns:
        pd.DataFrame:
            User ID, Movie ID, Rating and Unix Timestamp.
    """

    file_path = RAW_DATA_DIR / "ratings.dat"

    data_types = {
        "UserID": "int32",
        "MovieID": "int32",
        "Rating": "int8",
        "Timestamp": "int64",
    }

    return read_movielens_file(
        file_path,
        RATINGS_COLUMNS,
        data_types,
    )


# ---------------------------------------------------------
# 8. SAVE INGESTED DATASETS
# ---------------------------------------------------------

def save_ingested_data(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, str]:
    """
    Save all ingested datasets in Parquet format.

    Parquet provides efficient storage and preserves the
    structured data types of the DataFrames.

    Args:
        datasets:
            Dictionary containing the loaded users, movies
            and ratings DataFrames.

    Returns:
        dict[str, str]:
            Mapping of dataset names to their generated
            Parquet file paths.
    """

    output_files = {}

    for dataset_name, df in datasets.items():

        output_path = (
            PROCESSED_DATA_DIR
            / f"{dataset_name}_ingested.parquet"
        )

        df.to_parquet(
            output_path,
            engine="pyarrow",
            index=False,
        )

        output_files[dataset_name] = str(output_path)

        logger.info(
            "Saved %s records to %s",
            f"{len(df):,}",
            output_path.name,
        )

    return output_files


# ---------------------------------------------------------
# 9. GENERATE INGESTION SUMMARY
# ---------------------------------------------------------

def generate_ingestion_summary(
    datasets: dict[str, pd.DataFrame],
    output_files: dict[str, str],
) -> dict[str, Any]:
    """
    Generate a metadata summary of the ingestion process.

    The summary records source information, record counts,
    dataset schemas, generated output paths and ingestion time.

    Args:
        datasets:
            Dictionary containing all ingested DataFrames.

        output_files:
            Mapping of dataset names to their output paths.

    Returns:
        dict:
            Complete ingestion metadata.
    """

    summary = {
        "project": "DM4ML Assignment 2",
        "author": "Suraj Gate",
        "student_id": "2025ag05348",
        "dataset": "MovieLens 1M",
        "ingestion_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "datasets": {},
    }

    for dataset_name, df in datasets.items():

        summary["datasets"][dataset_name] = {
            "record_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "output_file": output_files[dataset_name],
        }

    summary_path = (
        PROCESSED_DATA_DIR / "ingestion_summary.json"
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    logger.info(
        "Ingestion summary saved to %s",
        summary_path,
    )

    return summary


# ---------------------------------------------------------
# 10. MAIN INGESTION PIPELINE
# ---------------------------------------------------------

def run_ingestion() -> dict[str, Any]:
    """
    Execute the complete data ingestion process.

    Workflow:
        1. Load user information.
        2. Load movie information.
        3. Load user-movie ratings.
        4. Save datasets as Parquet files.
        5. Generate an ingestion summary.

    Returns:
        dict:
            Ingestion summary containing dataset metadata
            and processing information.

    Raises:
        FileNotFoundError:
            If any required input dataset is missing.

        Exception:
            If any ingestion or storage operation fails.
    """

    logger.info("=" * 60)
    logger.info("STARTING MOVIELENS DATA INGESTION")
    logger.info("=" * 60)

    try:

        datasets = {
            "users": ingest_users(),
            "movies": ingest_movies(),
            "ratings": ingest_ratings(),
        }

        output_files = save_ingested_data(datasets)

        summary = generate_ingestion_summary(
            datasets,
            output_files,
        )

        logger.info("=" * 60)
        logger.info("DATA INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        for name, information in summary["datasets"].items():

            logger.info(
                "%s: %s records",
                name.upper(),
                f"{information['record_count']:,}",
            )

        return summary

    except Exception:
        logger.exception("DATA INGESTION FAILED")
        raise


# ---------------------------------------------------------
# 11. SCRIPT EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()
