# Architecture AutoML Studio

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Streamlit / React)                 │
│   Upload · EDA · Pretraitement · Leaderboard · XAI · Q&A · Export   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ REST / WebSocket
┌────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI (backend)                              │
│   /upload  /eda  /run  /qa  /predict  /report  /artifact             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────┐   ┌───────────▼────────┐   ┌──────────▼────────┐
│  EDA Agent     │   │  Preprocessing     │   │  Training Agent   │
│  stats, NA,    │   │  Agent             │   │  model registry + │
│  outliers,     │   │  impute/encode/    │   │  Optuna + CV      │
│  correlations  │   │  scale / PCA       │   │                   │
└────────────────┘   └────────────────────┘   └───────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                ┌────────────────▼────────────────┐
                │  Pipeline Orchestrator          │
                │  EDA → Cleaning → FE → Training │
                │  → XAI → Reporting → Deployment │
                └────────────────┬────────────────┘
                                 │
   ┌──────────────┬──────────────┼──────────────┬──────────────┐
   │              │              │              │              │
┌──▼──┐    ┌─────▼─────┐   ┌─────▼─────┐   ┌────▼────┐   ┌─────▼─────┐
│SHAP │    │Reports    │   │Deploy     │   │MLflow   │   │Q&A Agent  │
│LIME │    │HTML/PDF/  │   │ONNX/      │   │Tracking │   │NL→rules   │
│PDP  │    │Notebook   │   │FastAPI    │   │         │   │           │
└─────┘    └───────────┘   └───────────┘   └─────────┘   └───────────┘
```

## Agents spécialisés

| Agent | Rôle | Implémentation |
|-------|------|----------------|
| EDA Agent | Statistiques, outliers, anomalies, distributions | `agents/eda_agent.py` |
| Cleaning Agent | Imputation, encodage, scaling, drop columns | `agents/preprocessing_agent.py` |
| Feature Engineering Agent | Interactions, polynomial features | `agents/feature_engineering_agent.py` |
| Problem Detection Agent | Classification / Régression / Clustering / Anomalie | `agents/problem_detection_agent.py` |
| Model Selection Agent | Registry par type de problème + espace Optuna | `agents/model_selection_agent.py` |
| Training Agent | CV + Optuna + leaderboard | `agents/training_agent.py` |
| Explainability Agent | SHAP, permutation importance | `agents/explainability_agent.py` |
| Reporting Agent | HTML / PDF / MD / Notebook | `agents/reporting_agent.py` |
| Q&A Agent | Langage naturel → réponse basée sur EDA + leaderboard | `agents/qa_agent.py` |
| Deployment Agent | Pickle, ONNX, FastAPI stub, Dockerfile | `agents/deployment_agent.py` |

## Stockage

- `artifacts/uploads/<dataset_id>/` : datasets uploadés
- `artifacts/figures/<dataset_id>/` : graphiques (matplotlib)
- `artifacts/models/<dataset_id>/` : pipelines + leaderboard JSON
- `artifacts/reports/<dataset_id>/` : rapports HTML / PDF / MD
- `artifacts/notebooks/<dataset_id>/` : notebooks reproductibles
- `artifacts/deploy/<dataset_id>/` : bundle de déploiement

## Évolutions prévues

1. Remplacement du Q&A rule-based par un agent LLM (function-calling sur les outils `eda`, `preprocess`, `train`).
2. Module séries temporelles (Prophet, LSTM, auto-ARIMA).
3. Module vision (CNNs, transfer learning).
4. Module NLP (HuggingFace, embeddings, RAG).
5. Backend asynchrone avec Celery + Redis pour les entraînements longs.
6. Stockage MinIO pour les datasets > 100 MB.
7. Authentification multi-utilisateurs + partage de runs via MLflow.
