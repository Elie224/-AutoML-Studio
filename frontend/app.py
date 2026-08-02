"""Streamlit demo UI for AutoML Studio.

Run with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from automl.agents.qa_agent import QAAgent  # noqa: E402
from automl.core.config import get_settings  # noqa: E402
from automl.core.io import SUPPORTED_EXTENSIONS, find_dataset_file  # noqa: E402
from automl.pipeline import AutoMLPipeline  # noqa: E402


st.set_page_config(page_title="AutoML Studio", page_icon="🧠", layout="wide")
settings = get_settings()
ARTIFACTS_ROOT = settings.artifacts_root.resolve()


def resolve_artifact(relative_path: Optional[str]) -> Optional[Path]:
    """Resolve an artifact path defensively, ensuring it stays under ARTIFACTS_ROOT."""
    if not relative_path:
        return None
    try:
        candidate = (ARTIFACTS_ROOT / Path(relative_path)).resolve()
        candidate.relative_to(ARTIFACTS_ROOT)
    except (ValueError, OSError):
        return None
    return candidate


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
    uploaded = st.file_uploader(
        "CSV / TSV / Excel / Parquet / JSON",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    )
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
    dataset_id = "streamlit"
    dataset_dir = settings.upload_root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if uploaded is not None:
        save_path = dataset_dir / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        source = save_path
        name = Path(uploaded.name).stem
    else:
        source = SAMPLE
        name = SAMPLE.stem

    df = pd.read_csv(source) if str(source).endswith(".csv") else pd.read_excel(source)
    summary = {"dataset_id": dataset_id, "name": name, "target": target}
    st.session_state["dataset_meta"] = summary

    with st.spinner("Pipeline AutoML en cours..."):
        pipeline = AutoMLPipeline(
            source=source,
            target=target,
            name=name,
            dataset_id=dataset_id,
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
                    full_path = resolve_artifact(fig)
                    if full_path and full_path.is_file():
                        st.image(str(full_path), caption=Path(fig).name)
                    else:
                        st.warning(f"Figure introuvable : {fig}")
        else:
            st.info("Lance d'abord le pipeline.")

    with tabs[2]:
        preprocessing = result.preprocessing
        if hasattr(preprocessing, "to_dict"):
            st.json(preprocessing.to_dict())
        else:
            st.json(preprocessing)

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
            qa = QAAgent(df, target=st.session_state.get("dataset_meta", {}).get("target"), eda=result.eda, leaderboard=result.leaderboard)
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

        html_path = resolve_artifact(result.report_paths.get("html"))
        if html_path and html_path.is_file():
            st.download_button(
                "Télécharger le rapport HTML",
                data=html_path.read_bytes(),
                file_name="report.html",
                mime="text/html",
            )
        else:
            st.warning(f"Rapport HTML introuvable ({html_path}).")

        pdf_path = resolve_artifact(result.report_paths.get("pdf"))
        if pdf_path and pdf_path.is_file():
            st.download_button(
                "Télécharger le rapport PDF",
                data=pdf_path.read_bytes(),
                file_name="report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning(f"Rapport PDF introuvable ({pdf_path}).")

        pipeline_path = resolve_artifact(result.artifacts.get("pipeline"))
        if pipeline_path and pipeline_path.is_file():
            st.download_button(
                "Télécharger le pipeline",
                data=pipeline_path.read_bytes(),
                file_name="pipeline.joblib",
                mime="application/octet-stream",
            )
        else:
            st.warning(f"Pipeline introuvable ({pipeline_path}).")
