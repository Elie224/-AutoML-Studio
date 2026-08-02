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
from automl.agents.eda_agent import EDAAgent  # noqa: E402
from automl.agents.id_detector import detect_id_columns  # noqa: E402
from automl.agents.preprocessing_agent import PreprocessingAgent  # noqa: E402
from automl.agents.problem_detection_agent import ProblemDetectionAgent  # noqa: E402
from automl.agents.target_recommender import recommend_target  # noqa: E402
from automl.core.io import SUPPORTED_EXTENSIONS, find_dataset_file  # noqa: E402
from automl.pipeline import AutoMLPipeline  # noqa: E402

st.set_page_config(page_title='AutoML Studio', page_icon='🧠', layout='wide')
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

        with st.expander("🔍 Diagnostic intelligent (avant lancement)", expanded=True):
            suggested_target = None
            try:
                preview_meta = preview.copy()
                id_cols = detect_id_columns(preview_meta)
                id_names = [c.name for c in id_cols]
                analysis = preview_meta.drop(columns=id_names, errors="ignore")
                eda_preview = EDAAgent("preview").run(analysis, target=target)
                suggestions = recommend_target(preview_meta, target)
                q = eda_preview.quality_score or {}
                st.metric("Score de qualité", f"{q.get('score', 0)}/100", q.get("grade", "?"))
                if id_names:
                    st.markdown(f"**🆔 Identifiants détectés** : {', '.join(id_names)}")
                if suggestions:
                    st.markdown("**🎯 Cibles suggérées :**")
                    for s in suggestions[:3]:
                        reasons = "; ".join(s.reasons)
                        st.markdown(f"- `{s.column}` → {s.problem_type.value} (score {s.score}) — {reasons}")
                    if target is None:
                        choice = st.radio("Appliquer la cible suggérée", options=[s.column for s in suggestions[:1]], index=0, key="apply_suggestion")
                        suggested_target = choice
                if q.get("issues"):
                    st.markdown("**⚠️ Problèmes identifiés :**")
                    for issue in q["issues"]:
                        st.markdown(f"- {issue}")
            except Exception as exc:
                st.warning(f"Diagnostic indisponible : {exc}")
            if suggested_target and target is None:
                target = suggested_target
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
            persist_source=False,
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
        if not result.eda:
            st.info("Lance d'abord le pipeline.")
        else:
            eda = result.eda
            quality = eda.quality_score or {}
            score = quality.get("score", 0)
            grade = quality.get("grade", "?")
            quality_col, badge_col = st.columns([3, 1])
            with quality_col:
                st.subheader(f"Qualité des données : {score}/100")
            with badge_col:
                grade_color = {
                    "excellent": "#16a34a",
                    "bon": "#22c55e",
                    "moyen": "#f59e0b",
                    "faible": "#ef4444",
                }.get(str(grade).lower(), "#64748b")
                st.markdown(f"<span style=\"background:{grade_color};color:white;padding:6px 12px;border-radius:999px;font-weight:600\">{grade.upper()}</span>", unsafe_allow_html=True)
            with st.expander("⚠️ Problèmes identifiés", expanded=True):
                issues = quality.get("issues", [])
                if issues:
                    for issue in issues:
                        st.markdown(f"- {issue}")
                else:
                    st.markdown("_Aucun problème détecté._")
            if eda.id_columns:
                with st.expander("🆔 Identifiants détectés", expanded=True):
                    names = ", ".join(c.get("name", "?") for c in eda.id_columns)
                    st.markdown(f"**{names}** — exclus des analyses et du pipeline d'entraînement.")
            with st.expander("💡 Insights", expanded=True):
                if eda.insights:
                    for insight in eda.insights:
                        st.markdown(f"- {insight}")
                else:
                    st.markdown("_Le dataset semble propre._")
            with st.expander("🎯 Cible suggérée", expanded=True):
                if result.target_suggestions:
                    for s in result.target_suggestions[:3]:
                        col_label = s.get("column")
                        col_type = s.get("problem_type")
                        col_score = s.get("score")
                        st.markdown(f"- `{col_label}` → **{col_type}** (score {col_score})")
                else:
                    st.markdown("_Aucune suggestion disponible._")
            st.subheader("Figures")
            if eda.figures:
                figure_cols = st.columns(2)
                for idx, fig in enumerate(eda.figures):
                    with figure_cols[idx % 2]:
                        full_path = resolve_artifact(fig)
                        if full_path and full_path.is_file():
                            st.image(str(full_path), caption=Path(fig).name)
                        else:
                            st.warning(f"Figure introuvable : {fig}")
            else:
                st.info("Aucune figure générée.")
        plan = result.preprocessing
        drop = list(plan.drop_columns or [])
        impute = plan.imputation or {}
        encoders = plan.encoders or {}
        st.subheader("Colonnes supprimées")
        if drop:
            for col in drop: st.markdown(f"- 🗑️ **{col}**")
        else:
            st.markdown("_Aucune colonne supprimée._")
        st.subheader("Imputation des valeurs manquantes")
        if impute:
            for col, strategy in impute.items():
                st.markdown(f"- `{col}` → {strategy}")
        else:
            st.markdown("_Aucune imputation nécessaire._")
        st.subheader("Encodage")
        if encoders:
            for col, strategy in encoders.items():
                st.markdown(f"- `{col}` → {strategy}")
        else:
            st.markdown("_Aucun encodage défini._")
        st.subheader("Normalisation")
        st.markdown(f"- Mise à l'échelle : **{plan.scaling}**")
        if plan.feature_engineering:
            st.subheader("Feature engineering")
            for rule in plan.feature_engineering:
                st.markdown(f"- {rule}")
        if plan.rationale:
            st.subheader("Justification")
            for note in plan.rationale:
                st.markdown(f"- {note}")
        with st.expander("Configuration technique (JSON)"):
            st.json(plan.to_dict() if hasattr(plan, "to_dict") else plan)

    with tabs[3]:
        rows = []
        for rank, m in enumerate(result.leaderboard, start=1):
            row = {"#": rank, "Modèle": m.name, "Temps (s)": round(m.training_time_s, 2)}
            for metric_name, metric_value in (m.metrics or {}).items():
                row[metric_name] = metric_value
            rows.append(row)
        df_lb = pd.DataFrame(rows)
        st.dataframe(df_lb, width="stretch")
        if not df_lb.empty:
            metric_cols = [c for c in df_lb.columns if c not in {"#", "Modèle", "Temps (s)"}]
            if metric_cols:
                primary = metric_cols[0]
                chart_df = df_lb[["Modèle", primary]].set_index("Modèle")
                st.bar_chart(chart_df)

    with tabs[4]:
        if not result.explainability:
            st.info("Explicabilité non disponible." )
        else:
            st.write("Méthode:", result.explainability.method)
            items = list(result.explainability.feature_importance.items())[:15]
            if items:
                imp_df = pd.DataFrame(items, columns=["feature", "importance"])
                st.bar_chart(imp_df.set_index("feature"))
            if result.problem_type.value in {"clustering", "anomaly_detection"}:
                st.subheader("Profils de clusters")
                for note in result.explainability.notes:
                    st.markdown(f"- {note}")
            for fig in result.explainability.figures:
                full_path = resolve_artifact(fig)
                if full_path and full_path.is_file():
                    st.image(str(full_path), caption=Path(fig).name)

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
            st.download_button("Télécharger le rapport HTML", data=html_path.read_bytes(), file_name="report.html", mime="text/html")
        else:
            st.warning(f"Rapport HTML introuvable ({html_path}).")

        pdf_path = resolve_artifact(result.report_paths.get("pdf"))
        if pdf_path and pdf_path.is_file():
            st.download_button("Télécharger le rapport PDF", data=pdf_path.read_bytes(), file_name="report.pdf", mime="application/pdf")
        else:
            st.warning(f"Rapport PDF introuvable ({pdf_path}).")

        pipeline_path = resolve_artifact(result.artifacts.get("pipeline"))
        if pipeline_path and pipeline_path.is_file():
            st.download_button("Télécharger le pipeline", data=pipeline_path.read_bytes(), file_name="pipeline.joblib", mime="application/octet-stream")
        else:
            st.warning(f"Pipeline introuvable ({pipeline_path}).")





