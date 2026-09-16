# PMLDL Assignment 1 — Bank Customer Churn MLOps Pipeline

An automated MLOps pipeline that processes data, trains a churn prediction model, and deploys it through a REST API and a web application, all running in separate Docker containers.

## Dataset

[Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) — predicts whether a bank customer will leave the bank (`Exited`), based on features such as credit score, age, balance, and account activity.

## Project Structure

├── code
│ ├── datasets/prepare_data.py # Stage 1: data loading, cleaning, splitting
│ ├── models/train_model.py # Stage 2: feature engineering, training, MLflow logging
│ ├── pipeline_runner.py # Orchestrates Stage 1 + Stage 2, repeats every 5 minutes
│ └── deployment
│ ├── api/ # FastAPI service (Dockerfile + app.py)
│ ├── app/ # Streamlit app (Dockerfile + streamlit_app.py)
│ ├── pipeline/ # Automation container (Dockerfile)
│ └── docker-compose.yml
├── data
│ ├── raw/churn.csv # Raw dataset (DVC-tracked)
│ └── processed/ # train.csv / test.csv (generated)
├── models/ # Trained model.pkl and scaler.pkl (generated)
├── requirements.txt
└── README.md


## Pipeline Stages

1. **Data Engineering** — loads `churn.csv`, drops non-predictive columns (`RowNumber`, `CustomerId`, `Surname`), splits into train/test (80/20), tracked with **DVC**.
2. **Model Engineering** — one-hot encodes `Geography`/`Gender`, scales features, trains a Logistic Regression model, logs parameters/metrics/model to **MLflow** (Accuracy: 0.8110, ROC-AUC: 0.7789).
3. **Deployment** — a **FastAPI** service serves predictions at `/predict`; a **Streamlit** app provides a form-based UI that calls the API.

## Running the Pipeline

### 1. Start MLflow locally (required to view experiment tracking)

The automation container connects to the MLflow server running on your machine via `host.docker.internal`. **Start this before running Docker Compose**, or the pipeline container's MLflow logging will fail (the rest of the pipeline still works):

```bash
pip install mlflow
mlflow ui --host 0.0.0.0 --allowed-hosts "localhost:5000,127.0.0.1:5000,host.docker.internal:5000"
```

View experiments at `http://127.0.0.1:5000`.

### 2. Build and run all services

```bash
cd code/deployment
docker compose up --build
```

This starts three containers:
- **`churn-pipeline`** — runs Stage 1 + Stage 2, then repeats every 5 minutes
- **`churn-api`** — FastAPI service at `http://localhost:8000` (docs at `/docs`)
- **`churn-app`** — Streamlit app at `http://localhost:8501`

### 3. Use the app

Open `http://localhost:8501`, fill in customer details, and click **Predict** to get a churn prediction.

## Manual / local execution (without Docker)

```bash
pip install -r requirements.txt

# Stage 1
python code/datasets/prepare_data.py

# Stage 2
python code/models/train_model.py

# Stage 3 (in separate terminals)
uvicorn code.deployment.api.app:app --reload --port 8000
streamlit run code/deployment/app/streamlit_app.py
```

Note: when running the API locally (outside Docker), `code/deployment/api/app.py` expects to be run from the repository root (it loads `models/model.pkl` relative to the current directory).

## Data Versioning

Raw data is tracked with DVC, with a local remote (`dvcstore/`):

```bash
dvc pull
```