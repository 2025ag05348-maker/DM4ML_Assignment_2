# End-to-End Data Management Pipeline for a Recommendation System

**Course:** Data Management for Machine Learning (DM4ML)  
**Assignment:** Assignment 2  
**Student:** Suraj Gate  
**BITS Student ID:** `2025ag05348`  
**Dataset:** MovieLens 1M  
**Primary language:** Python 3.11

## 1. Project overview

This project implements a reproducible, end-to-end recommendation-system data pipeline using the MovieLens 1M dataset. It covers raw-data ingestion, automated data-quality validation, chronological train/test preparation, reusable feature generation, recommendation-model training, model evaluation, experiment tracking, and workflow orchestration.

The pipeline trains and compares two recommenders:

1. **Popularity-based baseline** — ranks movies using training interaction count, with training average rating and MovieID as deterministic tie-breakers.
2. **Personalized collaborative filtering** — applies Truncated SVD to the sparse user–movie ratings matrix and ranks unseen movies using reconstructed preference scores.

The complete pipeline is orchestrated with Prefect. MLflow records model parameters, diagnostic metrics, run metadata, and model/recommendation artifacts using a local SQLite tracking database.

## 2. Dataset

The project uses **MovieLens 1M**, published by GroupLens Research. The archive contains three source files:

| Source file | Description | Original records |
|---|---|---:|
| `users.dat` | User demographics: ID, gender, age group, occupation, ZIP code | 6,040 |
| `movies.dat` | Movie catalogue: ID, title, genres | 3,883 |
| `ratings.dat` | User–movie ratings and Unix timestamps | 1,000,209 |

The `.dat` files use `::` as the field separator and do not have a header row. The ingestion module assigns the column names and data types.

### Download the dataset

The dataset is downloaded from the official GroupLens location. It is not committed to this repository; the download steps below reproduce the local raw-data directory.

```bash
mkdir -p data/raw
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip \
  -O data/raw/ml-1m.zip
unzip data/raw/ml-1m.zip -d data/raw/
```

After extraction, the expected input files are:

```text
data/raw/ml-1m/users.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/ratings.dat
```

Refer to the `README` distributed inside the MovieLens archive and the GroupLens dataset page for dataset details and usage terms: <https://grouplens.org/datasets/movielens/1m/>.

## 3. Technology stack

- **Python 3.11** — implementation language.
- **Pandas and NumPy** — tabular ingestion, validation, preparation, and feature calculations.
- **PyArrow / Parquet** — compact, typed intermediate and output datasets.
- **SciPy sparse matrices** — memory-efficient user–movie interaction matrix.
- **scikit-learn TruncatedSVD** — personalized collaborative-filtering model.
- **Joblib** — serialized SVD model artifact.
- **MLflow with SQLite** — experiment, parameter, metric, and artifact tracking.
- **Prefect** — sequential workflow orchestration and execution monitoring.
- **Git and GitHub** — source control and repository submission.

The exact installed Python package versions are pinned in [`requirements.txt`](requirements.txt).

## 4. Pipeline architecture

The pipeline follows these stages:

1. **Ingestion:** read the three MovieLens `.dat` files, apply explicit column names and data types, and write Parquet datasets plus an ingestion summary.
2. **Validation:** check required values, data types, duplicate identifiers/interactions, rating range, timestamps, and user/movie referential integrity. Invalid records are written to `data/errors/`; valid records are written as validated Parquet files.
3. **Preparation and transformation:** convert Unix timestamps to UTC datetimes, create a per-user chronological 80:20 train/test holdout, compute user activity and item popularity summaries from training data, and create a CSR sparse ratings matrix.
4. **Feature management:** generate user, item, and interaction features and document them in a feature catalog. Historical aggregates are calculated from training data to avoid test-data leakage; observed ratings are retained as labels, not predictors.
5. **Model training:** train the popularity baseline and the Truncated SVD collaborative-filtering model. Generate up to 10 unseen-movie recommendations per user and log both model runs to MLflow.
6. **Evaluation:** evaluate the top 5 recommendations with Precision@5, Recall@5, NDCG@5, catalogue coverage, and genre-based diversity. Ratings of 4 or 5 in the test partition are treated as relevant.
7. **Orchestration:** Prefect executes the six processing modules in dependency order and then verifies the expected output artifacts.

See [`architecture/pipeline_architecture.png`](architecture/pipeline_architecture.png) or [`architecture/pipeline_architecture.jpg`](architecture/pipeline_architecture.jpg) for the rendered diagram. The SVG is available as a scalable vector, and [`architecture/pipeline_architecture.mmd`](architecture/pipeline_architecture.mmd) is the editable Mermaid source.

## 5. Repository structure

```text
2025ag05348_DM4ML_Assignment_2/
├── architecture/
│   ├── pipeline_architecture.png
│   ├── pipeline_architecture.jpg
│   ├── pipeline_architecture.svg
│   └── pipeline_architecture.mmd
├── configs/
├── data/
│   ├── raw/                  # Downloaded locally; not committed
│   ├── processed/            # Parquet datasets and summaries
│   ├── features/             # Feature tables, matrix and recommendations
│   └── errors/               # Rejected records (empty when no errors)
├── dags/                     # Reserved for workflow definitions if required
├── models/                   # Trained model artifacts (generated locally)
├── reports/
│   ├── validation/
│   ├── preparation/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   ├── screenshots/
│   └── 2025ag05348_DM4ML_Assignment_2_Report.pdf
├── src/
│   ├── ingestion/ingest_data.py
│   ├── validation/validate_data.py
│   ├── preparation/prepare_data.py
│   ├── features/build_features.py
│   ├── training/train_models.py
│   ├── evaluation/evaluate_models.py
│   └── orchestration/pipeline.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated data, model files, reports, and local service databases are created by executing the modules. The source repository should contain the code and documentation required to reproduce them, plus only the selected lightweight evidence files.

## 6. Environment setup (Ubuntu/Linux)

### Prerequisites

- Ubuntu/Linux
- Python 3.11
- `python3-venv`, `pip`, `git`, and `unzip`
- Internet access for downloading MovieLens and Python packages

Install the operating-system tools if they are not already installed:

```bash
sudo apt update
sudo apt install -y python3.11-venv python3-pip git unzip
```

Create and activate a virtual environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download and extract MovieLens using the commands in [Section 2](#2-dataset).

## 7. Run the pipeline

Run commands from the repository root with `.venv` activated.

### Run each stage independently

```bash
python -m src.ingestion.ingest_data
python -m src.validation.validate_data
python -m src.preparation.prepare_data
python -m src.features.build_features
python -m src.training.train_models
python -m src.evaluation.evaluate_models
```

### Run the complete workflow with Prefect

**Terminal 1 — start the local Prefect server:**

```bash
source .venv/bin/activate
prefect server start --host 127.0.0.1
```

**Terminal 2 — execute the flow:**

```bash
cd /path/to/2025ag05348_DM4ML_Assignment_2
source .venv/bin/activate
export PREFECT_API_URL=http://127.0.0.1:4200/api
python -m src.orchestration.pipeline
```

Open the Prefect UI at <http://127.0.0.1:4200>. The flow is expected to run these seven Prefect tasks in order: ingestion, validation, preparation, feature engineering, model training, model evaluation, and output verification.

### View MLflow experiments (optional)

The training module stores tracking metadata in the local `mlflow.db` SQLite database and artifacts in `mlartifacts/`. From the repository root, run:

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Then open <http://127.0.0.1:5000>.

## 8. Data validation and preparation results

The executed validation report recorded the following results:

| Dataset | Source records | Valid records | Invalid records |
|---|---:|---:|---:|
| Users | 6,040 | 6,040 | 0 |
| Movies | 3,883 | 3,883 | 0 |
| Ratings | 1,000,209 | 1,000,209 | 0 |
| **Total** | **1,010,132** | **1,010,132** | **0** |

All implemented schema checks passed. No records were rejected, so the error CSV files contain headers and no data rows.

The preparation stage converted timestamps to UTC and performed an **80:20 per-user chronological holdout**:

| Output | Records |
|---|---:|
| Prepared interactions | 1,000,209 |
| Training interactions | 797,758 |
| Testing interactions | 202,451 |
| Users represented in training and test | 6,040 / 6,040 |

The training user–movie matrix has **6,040 rows × 3,667 columns**, with **797,758 non-zero interactions** and approximately **3.60% density**. The test partition contains 54 interactions for movies absent from training; these are retained and documented as cold-start cases.

## 9. Feature management

The feature engineering stage generated:

| Feature output | Records / entries |
|---|---:|
| User feature table | 6,040 |
| Item feature table | 3,883 |
| Training interaction features | 797,758 |
| Test interaction features | 202,451 |
| Feature catalog | 24 entries |

The catalog documents feature name, entity, data type, description, source, and availability. Training-derived user/item aggregates and historical interaction features use training history; test ratings are stored as labels and are not used as predictors.

Primary outputs:

- `data/features/user_features.parquet`
- `data/features/item_features.parquet`
- `data/features/interaction_features_train.parquet`
- `data/features/interaction_features_test.parquet`
- `data/features/feature_catalog.csv`

## 10. Model training

### Popularity-based baseline

Movies are ranked by training interaction count (descending), training average rating (descending), and MovieID (ascending). Movies already rated by a user are excluded from that user's recommendation list.

### Collaborative filtering with Truncated SVD

The personalized model uses the sparse training ratings matrix and scikit-learn `TruncatedSVD` with the following parameters:

| Parameter | Value |
|---|---:|
| Components | 32 |
| Iterations | 7 |
| Random state | 42 |
| Recommendations per user | 10 |

Both models generated **60,400 recommendation rows** (up to 10 for each of 6,040 users). Model files and recommendation tables are saved locally; MLflow logs a separate run for each model.

## 11. Evaluation methodology and results

Evaluation uses the latest 20% chronological interactions per user as the test partition. A test rating of **4 or 5** is considered relevant. Precision, Recall, and NDCG are macro-averaged over the 5,976 users with at least one relevant test item. Coverage uses the complete 3,883-movie catalogue. Diversity is the mean pairwise Jaccard distance between movie genre sets within each user's top-5 list, averaged over users with at least two recommendations.

| Metric | Popularity baseline | Collaborative filtering (SVD) |
|---|---:|---:|
| Precision@5 | 0.08865 | **0.10977** |
| Recall@5 | 0.02635 | **0.04757** |
| NDCG@5 | 0.09375 | **0.11795** |
| Catalogue coverage | 0.02189 (2.19%) | **0.16302 (16.30%)** |
| Diversity | 0.67655 | **0.71913** |

On this evaluation setup, the SVD collaborative-filtering model outperformed the popularity baseline on all five reported metrics. These values are the observed results for this chronological split and relevance definition; they should not be interpreted as universal performance guarantees.

The comparison is available in `reports/evaluation/model_evaluation_results.csv`, and the complete methodology and metric values are in `reports/evaluation/evaluation_summary.json`.

## 12. Orchestration result

The successful Prefect execution on **8 November 2026** completed all seven tasks:

1. Data Ingestion
2. Data Validation
3. Data Preparation
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Verify Pipeline Outputs

The final verification task confirmed that **19 required pipeline artifacts** existed. Execution evidence is stored in `reports/screenshots/` when the screenshots are captured and copied into the repository.

## 13. Limitations

- MovieLens 1M is a historical ratings benchmark; its user and item attributes are limited compared with a live commercial recommendation platform.
- The collaborative-filtering model is trained on movies observed in the training interaction matrix. Test-only movies and new users represent cold-start cases and cannot receive personalized SVD scores from learned factors.
- Ratings of 4 or 5 are treated as relevant for ranking evaluation. Different relevance thresholds or temporal splits may produce different metrics.
- The current workflow is a local reproducible batch pipeline, not a production online serving system. It does not include live event ingestion, an online feature store, or an API for real-time recommendations.
- The pipeline runs sequentially in one local environment; scaling, distributed training, and production scheduling are outside this assignment implementation.

## 14. Reproducibility notes

- Use the Python versions and package pins in `requirements.txt`.
- Download the MovieLens archive using the commands in Section 2.
- Run the pipeline from the repository root.
- The pipeline uses fixed SVD parameters and random state for repeatable model configuration.
- MLflow and Prefect local state are generated on the machine where the pipeline is executed.

## 15. Student information

**Suraj Gate**  
**BITS Student ID:** `2025ag05348`  
**Course:** Data Management for Machine Learning (DM4ML)  
**Assignment 2:** End-to-End Data Management Pipeline for a Recommendation System

## 16. Final report

The concise 3-page assignment report is available at [`reports/2025ag05348_DM4ML_Assignment_2_Report.pdf`](reports/2025ag05348_DM4ML_Assignment_2_Report.pdf). It summarizes the architecture, tools, validation and preparation results, feature engineering, model comparison, orchestration evidence, design decisions and limitations. This README remains the detailed setup and reproduction guide.
