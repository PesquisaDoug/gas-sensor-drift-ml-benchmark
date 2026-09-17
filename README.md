# Classification, Regression and Clustering under Sensor Drift

A reproducible multi-task machine-learning benchmark for the UCI Gas Sensor Array Drift at Different Concentrations dataset.

Dataset: UCI Machine Learning Repository dataset 270  
DOI: 10.24432/C5MK6M  
Source archive: https://archive.ics.uci.edu/static/public/270/gas%2Bsensor%2Barray%2Bdrift%2Bdataset%2Bat%2Bdifferent%2Bconcentrations.zip

## Research Questions

- RQ1, Classification: how do classical supervised models compare when identifying gas type?
- RQ2, Regression: how accurately can gas concentration be estimated?
- RQ3, Sensor drift: how much does performance degrade on later batches?
- RQ4, Clustering: do clusters align more strongly with gas identity or acquisition batch?
- RQ5, Efficiency: what trade-offs exist between performance, training time, inference time and model size?

## Method

The dataset contains 13,910 measurements, 128 engineered sensor features, six gases and ten acquisition batches. The parser creates `gas_class_original`, zero-based `gas_class`, textual `gas`, `concentration_ppmv`, `batch`, and `feature_001` through `feature_128`.

No target or metadata column is allowed into the feature matrix. The feature set is exactly the 128 engineered feature columns.

Two protocols are preserved from the executed notebook:

- Random hold-out: `train_test_split(test_size=0.20, stratify=gas_class, random_state=42)`.
- Later-batch temporal split: train on batches 1-5 and test on batches 6-10.

The random split and temporal split answer different questions. Near-perfect random results can occur because training and testing share acquisition regimes and batches. The later-batch protocol is deliberately harder and evaluates distribution shift due to sensor drift.

## Models

Classification: Dummy, RandomForest, XGBoost, SVM and MLP.  
Regression: Dummy, RandomForest, XGBoost, SVR and MLP.  
SVM, SVR and MLP use `StandardScaler` inside sklearn pipelines to avoid global scaling before CV.

## Metrics and Outputs

Classification metrics include accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1 and MCC. Regression metrics include MAE, RMSE, median absolute error and R2. Clustering reports internal metrics plus ARI/NMI against gas and batch as post-hoc diagnostics.

Main outputs are written to `results/`, figures to `figures/`, and serialized models to `artifacts/models/`.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip check
```

## QUICK Run

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m scripts.run_classification --config configs/experiment.yaml --profile quick
.\.venv\Scripts\python -m scripts.run_regression --config configs/experiment.yaml --profile quick
.\.venv\Scripts\python -m scripts.run_clustering --config configs/experiment.yaml --profile quick
.\.venv\Scripts\python -m scripts.generate_results --config configs/experiment.yaml --profile quick
```

## FULL Run

Use the same commands with `--profile full`. This increases tuning iterations, bootstrap iterations, inference repeats, clustering sample size and permutation importance repeats.

## Streamlit

```powershell
.\.venv\Scripts\streamlit run app/streamlit_app.py
```

The app works with the 128 engineered UCI features. It does not collect live sensor data and does not process raw time series.

## Limitations

This is a public laboratory dataset. The features are engineered sensor responses, not raw sensor time series. Ten batches represent different acquisition periods. Random hold-out is not temporal deployment. The temporal experiment is a simplified sensor-drift/distribution-shift evaluation; it does not implement drift compensation, domain adaptation or continual learning.

Concentration ranges vary between gases, which affects interpretation of global regression metrics. Hyperparameter search is intentionally bounded. Clustering is exploratory, ARI/NMI are post-hoc, PCA/clustering use bounded samples, and permutation importance is predictive rather than causal. Results should not be generalized automatically to production industrial sensing systems.

## Versioning

The repository is published at https://github.com/PesquisaDoug/gas-sensor-drift-ml-benchmark. No CI/CD, GitHub Actions, or cloud deployment are configured by this project.
