FROM python:3.11-slim

WORKDIR /app

RUN pip install pandas scikit-learn mlflow joblib

COPY code/datasets/prepare_data.py code/datasets/prepare_data.py
COPY code/models/train_model.py code/models/train_model.py
COPY code/pipeline_runner.py code/pipeline_runner.py
COPY data/raw/churn.csv data/raw/churn.csv

RUN mkdir -p data/processed models

CMD ["python", "code/pipeline_runner.py"]