# 🧠 AutoML Studio

> Plateforme **no-code / low-code** qui automatise tout le cycle de vie d'un projet de Machine Learning — de l'import CSV jusqu'au déploiement en API.

![Architecture](docs/architecture.svg)

## ✨ Fonctionnalités

| Étape | Agent | Détails |
|-------|-------|---------|
| Import | `io.py` | CSV, Excel, Parquet, JSON, SQL, API JSON |
| EDA | `EDAAgent` | Stats descriptives, valeurs manquantes, doublons, corrélations, distributions, outliers, déséquilibre, anomalies, dates |
| Prétraitement | `PreprocessingAgent` | Imputation, encodage, scaling, drop colonnes, ratios, dates, PCA |
| Détection du problème | `ProblemDetectionAgent` | Classification / Régression / Clustering / Anomalies / Time Series |
| Feature engineering | `FeatureEngineeringAgent` | Interactions, polynomial features |
| Sélection des modèles | `ModelSelectionAgent` | Registry par type de problème + espace de recherche Optuna |
| Entraînement | `TrainingAgent` | Validation croisée + Optuna + leaderboard |
| Explicabilité | `ExplainabilityAgent` | SHAP + permutation importance |
| Q&A | `QAAgent` | Langage naturel → réponse basée sur EDA + leaderboard |
| Reporting | `ReportingAgent` | HTML, PDF, Markdown exécutif, Notebook reproductible |
| Déploiement | `DeploymentAgent` | Pickle, ONNX (best-effort), FastAPI stub, Dockerfile |

## 🚀 Démarrage rapide

### Pré-requis
- Python 3.10+
- (optionnel) Docker & docker-compose
- 4 Go de RAM minimum

### Installation

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Lancer l'API
```bash
uvicorn automl.api.main:app --reload
```
Swagger : <http://localhost:8000/docs>

### Lancer l'UI Streamlit
```bash
streamlit run frontend/app.py
```
UI : <http://localhost:8501>

### Docker (API + Frontend)
```bash
docker compose -f deploy/docker-compose.yml up --build
```

## 📦 Exemple d'utilisation

```python
from automl.pipeline import AutoMLPipeline

pipeline = AutoMLPipeline(
    source="data/samples/titanic.csv",
    target="Survived",
    optuna_trials=10,
    cv_folds=5,
)
result = pipeline.run()

print("Best model:", result.best_model.name)
print("Metrics:", result.best_model.metrics)
print("Leaderboard:", [(m.name, m.metrics) for m in result.leaderboard])
```

## 🌐 API REST

| Endpoint | Description |
|----------|-------------|
| `POST /upload` | Upload d'un CSV/Excel/Parquet |
| `GET /eda/{dataset_id}` | EDA complet (stats, figures, insights) |
| `POST /run` | Lance le pipeline complet |
| `POST /qa` | Pose une question sur le dataset |
| `POST /predict/{dataset_id}` | Inférence sur un nouveau record |
| `GET /report/{dataset_id}/{kind}` | Télécharge le rapport (html/pdf/executive/notebook) |
| `GET /artifact?path=...` | Sert un artefact (figure, pipeline) |

Exemple `curl` :
```bash
curl -F "file=@data/samples/titanic.csv" -F "target=Survived" http://localhost:8000/upload
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"abc123","target":"Survived","optuna_trials":10,"cv_folds":5}'
```

## 📁 Structure

```
automl_studio/
├── backend/
│   ├── automl/
│   │   ├── agents/          # 10 agents spécialisés
│   │   ├── api/             # FastAPI surface
│   │   ├── core/            # config, io, schema
│   │   └── pipeline.py      # orchestrateur
│   ├── tests/               # tests pytest
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   └── app.py               # UI Streamlit
├── data/
│   ├── samples/             # Titanic + Housing
│   └── generate_samples.py
├── notebooks/
│   └── demo.ipynb
├── deploy/
│   └── docker-compose.yml
├── docs/
│   └── architecture.md
└── README.md
```

## 🧪 Tests

```bash
cd backend
pytest tests/ -v
```

## 🛣️ Roadmap

- [ ] Module séries temporelles (Prophet, auto-ARIMA, LSTM)
- [ ] Module vision par ordinateur (CNNs, transfer learning)
- [ ] Module NLP + RAG
- [ ] Q&A agent basé sur un vrai LLM (function-calling)
- [ ] Backend asynchrone (Celery + Redis)
- [ ] Stockage MinIO pour datasets > 100 MB
- [ ] Auth multi-utilisateurs + partage MLflow
- [ ] Frontend React/Next.js complet (le Streamlit actuel est une démo)

## 🎯 Pourquoi ce projet impressionne un recruteur

- Pipeline **end-to-end** : ingestion, EDA, preprocessing, training, XAI, déploiement.
- **Multi-agents** : EDA, Cleaning, FE, Model Selection, Training, Explainability, Reporting, Q&A, Deployment.
- Code **production-ready** : logging, dataclasses, pydantic, tests pytest, docker.
- Explicabilité intégrée (SHAP).
- Déploiement API + ONNX.
- Couvre un spectre complet : **Data Science + ML Engineering + MLOps**.
