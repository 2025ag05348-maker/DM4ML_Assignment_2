
"""
Module: pipeline.py

Purpose:
    Orchestrate the complete DM4ML Assignment 2 recommendation
    system pipeline using Prefect.

Workflow:
    1. Data ingestion
    2. Data validation
    3. Data preparation and transformation
    4. Feature engineering
    5. Model training
    6. Model evaluation
    7. Final artifact verification

Each stage executes as a separate Prefect task.
The workflow stops when a task fails.

Author: Suraj Gate
BITS Student ID: 2025ag05348
Course: Data Management for Machine Learning (DM4ML)
Assignment: 2
"""

import logging
import subprocess
import sys

from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.logging import get_run_logger


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# COMMON MODULE EXECUTION
# ---------------------------------------------------------

def execute_module(module_name: str) -> dict[str, Any]:
    """
    Execute an existing pipeline module as a Python subprocess.

    Args:
        module_name:
            Fully qualified Python module name.

    Returns:
        dict:
            Module name, return code and execution status.

    Raises:
        subprocess.CalledProcessError:
            If the executed module returns a non-zero exit code.
    """

    logger = get_run_logger()

    logger.info("Starting module: %s", module_name)

    command = [
        sys.executable,
        "-m",
        module_name,
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error(
            "Module failed: %s (exit code: %s)",
            module_name,
            result.returncode,
        )

        raise subprocess.CalledProcessError(
            result.returncode,
            command,
        )

    logger.info("Completed module: %s", module_name)

    return {
        "module": module_name,
        "return_code": result.returncode,
        "status": "SUCCESS",
    }


# ---------------------------------------------------------
# TASK 1: DATA INGESTION
# ---------------------------------------------------------

@task(name="01 - Data Ingestion")
def ingestion_task() -> dict[str, Any]:
    """Execute MovieLens raw-data ingestion."""
    return execute_module("src.ingestion.ingest_data")


# ---------------------------------------------------------
# TASK 2: DATA VALIDATION
# ---------------------------------------------------------

@task(name="02 - Data Validation")
def validation_task() -> dict[str, Any]:
    """Validate users, movies and interaction records."""
    return execute_module("src.validation.validate_data")


# ---------------------------------------------------------
# TASK 3: DATA PREPARATION
# ---------------------------------------------------------

@task(name="03 - Data Preparation")
def preparation_task() -> dict[str, Any]:
    """Clean interactions and create chronological train/test data."""
    return execute_module("src.preparation.prepare_data")


# ---------------------------------------------------------
# TASK 4: FEATURE ENGINEERING
# ---------------------------------------------------------

@task(name="04 - Feature Engineering")
def feature_task() -> dict[str, Any]:
    """Generate reusable user, item and interaction features."""
    return execute_module("src.features.build_features")


# ---------------------------------------------------------
# TASK 5: MODEL TRAINING
# ---------------------------------------------------------

@task(name="05 - Model Training")
def training_task() -> dict[str, Any]:
    """Train popularity and collaborative filtering models."""
    return execute_module("src.training.train_models")


# ---------------------------------------------------------
# TASK 6: MODEL EVALUATION
# ---------------------------------------------------------

@task(name="06 - Model Evaluation")
def evaluation_task() -> dict[str, Any]:
    """Evaluate both recommenders and log evaluation metrics."""
    return execute_module("src.evaluation.evaluate_models")


# ---------------------------------------------------------
# TASK 7: OUTPUT VERIFICATION
# ---------------------------------------------------------

@task(name="07 - Verify Pipeline Outputs")
def verify_outputs_task() -> dict[str, Any]:
    """
    Verify that all important pipeline artifacts exist.

    Returns:
        dict:
            Verification status and checked file count.

    Raises:
        FileNotFoundError:
            When any expected pipeline artifact is missing.
    """

    logger = get_run_logger()

    required_files = [
        "data/processed/users_validated.parquet",
        "data/processed/movies_validated.parquet",
        "data/processed/ratings_validated.parquet",
        "data/processed/train.parquet",
        "data/processed/test.parquet",
        "data/features/user_features.parquet",
        "data/features/item_features.parquet",
        "data/features/feature_catalog.csv",
        "models/popularity_model.json",
        "models/collaborative_filtering_svd.joblib",
        "data/features/recommendations_popularity.parquet",
        "data/features/recommendations_svd.parquet",
        "reports/validation/validation_report.json",
        "reports/preparation/preparation_summary.json",
        "reports/features/feature_summary.json",
        "reports/training/training_summary.json",
        "reports/evaluation/evaluation_summary.json",
        "reports/evaluation/model_evaluation_results.csv",
        "mlflow.db",
    ]

    missing_files = [
        file_name
        for file_name in required_files
        if not (PROJECT_ROOT / file_name).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Pipeline artifact verification failed:\n"
            + "\n".join(missing_files)
        )

    logger.info(
        "All %s required pipeline artifacts verified.",
        len(required_files),
    )

    return {
        "status": "SUCCESS",
        "verified_files": len(required_files),
    }


# ---------------------------------------------------------
# COMPLETE PREFECT WORKFLOW
# ---------------------------------------------------------

@flow(
    name="DM4ML Assignment 2 Recommendation Pipeline",
    description=(
        "End-to-end MovieLens recommendation pipeline "
        "covering ingestion, validation, preparation, "
        "feature engineering, training and evaluation."
    ),
    log_prints=True,
)
def recommendation_pipeline() -> dict[str, Any]:
    """
    Execute the complete recommendation system pipeline.

    Tasks are executed sequentially to maintain correct
    data dependencies.

    Returns:
        dict:
            Execution status and all task results.
    """

    logger = get_run_logger()

    logger.info("=" * 65)
    logger.info("STARTING DM4ML RECOMMENDATION PIPELINE")
    logger.info("=" * 65)

    # Stage 1
    ingestion_result = ingestion_task()

    # Stage 2
    validation_result = validation_task()

    # Stage 3
    preparation_result = preparation_task()

    # Stage 4
    feature_result = feature_task()

    # Stage 5
    training_result = training_task()

    # Stage 6
    evaluation_result = evaluation_task()

    # Final verification
    verification_result = verify_outputs_task()

    summary = {
        "project": "DM4ML Assignment 2",
        "author": "Suraj Gate",
        "student_id": "2025ag05348",
        "pipeline": "Recommendation System",
        "status": "SUCCESS",
        "tasks": [
            ingestion_result,
            validation_result,
            preparation_result,
            feature_result,
            training_result,
            evaluation_result,
            verification_result,
        ],
    }

    logger.info("=" * 65)
    logger.info("RECOMMENDATION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)

    return summary


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    result = recommendation_pipeline()

    print("\nFINAL PIPELINE SUMMARY")
    print(result)
