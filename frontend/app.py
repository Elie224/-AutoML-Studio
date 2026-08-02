"""Streamlit demo UI for AutoML Studio.

Run with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from automl.agents.qa_agent import QAAgent  # noqa: E402
from automl.core.config import get_settings  # noqa: E402
from automl.core.io import dataframe_summary, store_metadata  # noqa: E402
from automl.pipeline import AutoMLPipeline  # noqa: E402


st.set_page_config(page_title="AutoML Studio", page_icon="🧠", layout="wide")
settings = get_settings()
ARTIFACTS_ROOT = settings.artifacts_root

st.title("🧠 AutoML Studio")
st.caption("Plateforme AutoML de bout en bout — ingestion, EDA, preprocessing, entraînement, explicabilité, déploiement.")

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    optuna_trials = st.slider("Essais Optuna", min_value=0, max_value=50, value=10, step=5)
    cv_folds = st.slider("Folds CV", min_value=2, max_value=10, value=5)
    run_shap = st.checkbox("Calculer SHAP", value=True)
    st.markdown("---")
    st.markdown("**Mode démo** : charger le dataset Titanic ou téléversez le vôtre.")

# ----------------------------------------------------------------------
# Dataset selection
# ----------------------------------------------------------------------
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "titanic.csv"
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1️⃣ Importer un dataset")
    uploaded = st.file_uploader("CSV / Excel / Parquet", type=["csv", "tsv", "xlsx", "parquet"])
    use_sample = st.button("Utiliser le dataset Titanic (démo)")

with col2:
    st.subheader("Aperçu")
    preview = None
    if uploaded is not None:
        try:
            preview = pd.read_csv(uploaded)
        except Exception:
            try:
                preview = pd.read_excel(uploaded)
            except Exception as exc:
                st.error(f"Lecture impossible: {exc}")
    elif use_sample and SAMPLE.exists():
        preview = pd.read_csv(SAMPLE)
    if preview is not None:
        st.dataframe(preview.head(10), width="stretch")
        target = st.selectbox("Colonne cible", options=["(aucune)"] + list(preview.columns))
        target = None if target == "(aucune)" else target
        run_button = st.button("🚀 Lancer le pipeline AutoML", type="primary")
    else:
        target = None
        run_button = False

# ----------------------------------------------------------------------
# Pipeline run
# ----------------------------------------------------------------------
if run_button and preview is not None:
    if uploaded is not None:
        save_path = ARTIFACTS_ROOT / "uploads" / "_streamlit" / (uploaded.name or "upload.csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(uploaded.getvalue())
        source = save_path
        name = Path(uploaded.name).stem
    else:
        source = SAMPLE
        name = SAMPLE.stem

    df = pd.read_csv(source) if str(source).endswith(".csv") else pd.read_excel(source)
    summary = dataframe_summary(df, "streamlit", name, target=target)
    store_metadata(summary)

    with st.spinner("Pipeline AutoML en cours..."):
        pipeline = AutoMLPipeline(
            source=source,
            target=target,
            name=name,
            optuna_trials=optuna_trials,
            cv_folds=cv_folds,
            run_explainability=run_shap,
        )
        result = pipeline.run()
        st.session_state["last_result"] = result
        st.session_state["last_df"] = df
        st.success("Pipeline terminé ✅")

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
result = st.session_state.get("last_result")
df = st.session_state.get("last_df")
if result is not None and df is not None:
    st.markdown("---")
    st.header("2️⃣ Résultats")

    tabs = st.tabs(["📊 Vue d'ensemble", "🔬 EDA", "🧹 Prétraitement", "🏆 Leaderboard", "🧠 Explicabilité", "💬 Q&A", "📥 Exports"])

    with tabs[0]:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Lignes", f"{result.dataset.rows}")
        col_b.metric("Colonnes", f"{result.dataset.columns}")
        col_c.metric("Type de problème", result.problem_type.value)
        col_d.metric("Meilleur modèle", result.best_model.name)
        st.subheader("Métriques du meilleur modèle")
        st.json(result.best_model.metrics)

    with tabs[1]:
        if result.eda:
            st.write("Insights")
            for insight in result.eda.insights:
                st.markdown(f"- {insight}")
            st.write("Figures")
            figure_cols = st.columns(2)
            for idx, fig in enumerate(result.eda.figures):
                with figure_cols[idx % 2]:
                    full_path = ARTIFACTS_ROOT / fig
                    if full_path.exists():
                        st.image(str(full_path), caption=Path(fig).name)
        else:
            st.info("Lance d'abord le pipeline.")

    with tabs[2]:
        st.json(result.preprocessing.to_dict() if hasattr(result.preprocessing, "to_dict") else result.preprocessing)

    with tabs[3]:
        rows = []
        for m in result.leaderboard:
            rows.append({"Model": m.name, "Primary metric": m.metrics, "Time (s)": m.training_time_s})
        st.dataframe(pd.DataFrame(rows), width="stretch")

    with tabs[4]:
        if result.explainability:
            st.write("Méthode:", result.explainability.method)
            items = list(result.explainability.feature_importance.items())[:15]
            imp_df = pd.DataFrame(items, columns=["feature", "importance"])
            st.bar_chart(imp_df.set_index("feature"))
        else:
            st.info("SHAP non disponible.")

    with tabs[5]:
        st.subheader("Posez une question en langage naturel")
        question = st.text_input("Ex: Pourquoi la cible est-elle déséquilibrée ?")
        if question:
            qa = QAAgent(df, target=target, eda=result.eda, leaderboard=result.leaderboard)
            response = qa.ask(question)
            st.info(response["answer"])

    with tabs[6]:
        st.write("Artefacts générés")
        for name, path in result.artifacts.items():
            st.write(f"- **{name}**: `{path}`")
        st.write("Rapports")
        for name, path in result.report_paths.items():
            if path:
                st.write(f"- **{name}**: `{path}`")
        if result.report_paths.get("html"):
            with open(ARTIFACTS_ROOT / result.report_paths["html"], "rb") as f:
                st.download_button("Télécharger le rapport HTML", f, file_name="report.html")
        if result.report_paths.get("pdf"):
            with open(ARTIFACTS_ROOT / result.report_paths["pdf"], "rb") as f:
                st.download_button("Télécharger le rapport PDF", f, file_name="report.pdf")
        if result.artifacts.get("pipeline"):
            with open(ARTIFACTS_ROOT / result.artifacts["pipeline"], "rb") as f:
                st.download_button("Télécharger le pipeline (joblib)", f, file_name="pipeline.joblib")
