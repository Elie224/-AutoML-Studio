
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
# ----------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------
THEME_CSS = '''
<style>
  :root {
    --bg: #0b1020;
    --panel: #111a33;
    --panel-2: #162241;
    --ink: #e6edff;
    --muted: #93a4c4;
    --accent: #6366f1;
    --accent-2: #22d3ee;
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #ef4444;
  }
  .stApp { background: radial-gradient(1200px 600px at 80% -10%, #1d2a5a 0%, var(--bg) 60%) fixed; color: var(--ink); }
  header[data-testid='stHeader'] { background: transparent; }
  section.main > div { padding-top: 0; }
  .ams-hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 45%, #06b6d4 100%);
    border-radius: 18px; padding: 36px 44px; color: white;
    box-shadow: 0 10px 30px rgba(2,6,23,.45); margin-bottom: 22px;
  }
  .ams-hero h1 { margin: 0 0 8px 0; font-size: 32px; font-weight: 800; letter-spacing: -0.01em; }
  .ams-hero p { margin: 0; opacity: .92; font-size: 15px; }
  .ams-hero .ams-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
  .ams-hero .ams-badge {
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
    color: white; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  }
  .ams-card {
    background: var(--panel); border: 1px solid #1f2a4a; border-radius: 14px;
    padding: 18px 20px; box-shadow: 0 1px 0 rgba(255,255,255,.03);
  }
  .ams-stepper { display: flex; gap: 8px; margin: 18px 0 24px 0; flex-wrap: wrap; }
  .ams-step {
    flex: 1 1 120px; background: var(--panel); border: 1px solid #1f2a4a;
    border-radius: 12px; padding: 12px 14px; color: var(--muted);
  }
  .ams-step .ams-step-num { font-size: 11px; opacity: .7; }
  .ams-step .ams-step-title { font-size: 14px; color: var(--ink); font-weight: 600; margin-top: 2px; }
  .ams-step.ams-step-active { border-color: var(--accent); background: linear-gradient(180deg, #1a2550, var(--panel)); }
  .ams-step.ams-step-done { border-color: var(--good); }
  .ams-grade {
    display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 12px;
    color: white; letter-spacing: .04em;
  }
  .ams-grade-excellent { background: var(--good); }
  .ams-grade-bon { background: #4ade80; }
  .ams-grade-moyen { background: var(--warn); }
  .ams-grade-faible { background: var(--bad); }
  .ams-section-title {
    font-size: 18px; font-weight: 700; margin: 18px 0 10px 0; color: var(--ink);
    display: flex; align-items: center; gap: 8px;
  }
  .ams-section-title .ams-section-icon {
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: inline-flex; align-items: center; justify-content: center; font-size: 16px;
  }
  div[data-testid='stFileUploader'] section { background: var(--panel-2); border: 1px dashed #2a3565; border-radius: 12px; }
  div[data-testid='stMetricValue'] { color: var(--ink); }
  .stButton>button { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: white; border: 0; font-weight: 600; }
  .stButton>button:hover { filter: brightness(1.08); }
  .stTabs [data-baseweb='tab-list'] { background: var(--panel); border-radius: 12px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb='tab'] { color: var(--muted); border-radius: 8px; padding: 8px 14px; }
  .stTabs [aria-selected='true'] { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: white; }
  .ams-footer { text-align: center; color: var(--muted); padding: 32px 0 12px 0; font-size: 12px; }
</style>
'''
st.markdown(THEME_CSS, unsafe_allow_html=True)






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
    st.markdown("#### ⚙️ Paramètres")
    optuna_trials = st.slider("Essais Optuna", min_value=0, max_value=50, value=10, step=5, help="Plus d'essais = meilleure optimisation, mais plus lent.")
    cv_folds = st.slider("Folds CV", min_value=2, max_value=10, value=5)
    run_shap = st.checkbox("Calculer SHAP", value=True)
    st.markdown("---")
    st.caption("Mode démo — charge le dataset Titanic ou téléverse le tien (CSV, Excel, Parquet, JSON, TSV).")

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown('''
<div class="ams-hero">
  <h1>🧠 AutoML Studio</h1>
  <p>Importe un dataset, lance le pipeline et obtiens un modèle entraîné, explicable et déployable — sans une ligne de code.</p>
  <div class="ams-badges">
    <span class="ams-badge">10 agents spécialisés</span>
    <span class="ams-badge">EDA · Prétraitement · Modélisation</span>
    <span class="ams-badge">SHAP · LIME · PDP</span>
    <span class="ams-badge">Exports HTML / PDF / Notebook</span>
  </div>
</div>
''', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Stepper
# ----------------------------------------------------------------------
st.markdown('''
<div class="ams-stepper">
  <div class="ams-step ams-step-active"><div class="ams-step-num">ÉTAPE 1</div><div class="ams-step-title">📂 Importer</div></div>
  <div class="ams-step"><div class="ams-step-num">ÉTAPE 2</div><div class="ams-step-title">🔍 Comprendre</div></div>
  <div class="ams-step"><div class="ams-step-num">ÉTAPE 3</div><div class="ams-step-title">🎯 Cibler</div></div>
  <div class="ams-step"><div class="ams-step-num">ÉTAPE 4</div><div class="ams-step-title">🚀 Entraîner</div></div>
  <div class="ams-step"><div class="ams-step-num">ÉTAPE 5</div><div class="ams-step-title">🧠 Expliquer</div></div>
  <div class="ams-step"><div class="ams-step-num">ÉTAPE 6</div><div class="ams-step-title">📥 Exporter</div></div>
</div>
''', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Dataset selection
# ----------------------------------------------------------------------
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "titanic.csv"
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="ams-section-title"><span class="ams-section-icon">1</span>Importer un dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Glisse ton fichier ici (CSV, TSV, Excel, Parquet, JSON)",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        help="Formats supportés : CSV, TSV, Excel, Parquet, JSON, JSONL.",
        label_visibility="collapsed",
    )
    use_sample = st.button("🎲 Charger le dataset Titanic (démo)", use_container_width=True)

with col2:
    st.markdown('<div class="ams-section-title"><span class="ams-section-icon">👁</span>Aperçu</div>', unsafe_allow_html=True)
    preview = None
    run_button = False
    if uploaded is not None:
        try:
            preview = pd.read_csv(uploaded)
        except Exception:
            try:
                preview = pd.read_excel(uploaded)
            except Exception as exc:
                st.error(f"Lecture impossible : {exc}")
    elif use_sample and SAMPLE.exists():
        preview = pd.read_csv(SAMPLE)
    if preview is not None:
        st.dataframe(preview.head(8), use_container_width=True, height=240)
        target = st.selectbox("Colonne cible (laisser « aucune » pour le clustering)", options=["(aucune)"] + list(preview.columns), index=0)
        target = None if target == "(aucune)" else target

        with st.expander("🔍 Diagnostic intelligent", expanded=True):
            suggested_target = None
            try:
                preview_meta = preview.copy()
                id_cols = detect_id_columns(preview_meta)
                id_names = [c.name for c in id_cols]
                analysis = preview_meta.drop(columns=id_names, errors="ignore")
                eda_preview = EDAAgent("preview").run(analysis, target=target)
                suggestions = recommend_target(preview_meta, target)
                q = eda_preview.quality_score or {}
                score = q.get("score", 0)
                grade = q.get("grade", "?")
                grade_class = {
                    "excellent": "ams-grade-excellent",
                    "bon": "ams-grade-bon",
                    "moyen": "ams-grade-moyen",
                    "faible": "ams-grade-faible",
                }.get(str(grade).lower(), "")
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
                    f"<div style='font-size:32px;font-weight:800;color:var(--ink);'>{score}<span style='font-size:14px;color:var(--muted);'>/100</span></div>"
                    f"<span class='ams-grade {grade_class}'>{grade.upper()}</span></div>",
                    unsafe_allow_html=True,
                )
                if id_names:
                    st.markdown(f"🆔 **Identifiants détectés** : `{'`, `'.join(id_names)}` (exclus)")
                if q.get("issues"):
                    st.markdown("**⚠️ Problèmes identifiés :**")
                    for issue in q["issues"]:
                        st.markdown(f"- {issue}")
                if suggestions:
                    st.markdown("**🎯 Cibles suggérées :**")
                    for s in suggestions[:3]:
                        reasons = "; ".join(s.reasons)
                        st.markdown(f"- `{s.column}` → **{s.problem_type.value}** _(score {s.score})_ — {reasons}")
                    if target is None:
                        apply = st.checkbox(
                            f"✅ Utiliser `{suggestions[0].column}` comme cible (problème {suggestions[0].problem_type.value})",
                            value=True,
                            key="apply_suggestion",
                        )
                        if apply:
                            suggested_target = suggestions[0].column
                breakdown = q.get("breakdown", []) or []
                if breakdown:
                    with st.expander("🧮 Décomposition du score", expanded=False):
                        st.caption("Pénalités appliquées (triées par impact décroissant) :")
                        for delta, label, evidence in breakdown:
                            st.markdown(f"- **−{delta} pts** : `{label}` — {evidence}")
            except Exception as exc:
                st.warning(f"Diagnostic indisponible : {exc}")
            if suggested_target and target is None:
                target = suggested_target

        run_button = st.button("🚀 Lancer le pipeline AutoML", type="primary", use_container_width=True)
    else:
        target = None

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
    st.markdown('''<div class="ams-section-title" style="margin-top:32px;"><span class="ams-section-icon">2</span>Résultats du pipeline</div>''', unsafe_allow_html=True)

    tabs = st.tabs(["📊 Vue d'ensemble", "🔬 EDA", "🧹 Prétraitement", "🏆 Leaderboard", "🧠 Explicabilité", "💬 Q&A", "📥 Exports"])

    with tabs[0]:
        col_a, col_b, col_c, col_d = st.columns(4)
        for col, label, value in [
            (col_a, "Lignes", f"{result.dataset.rows:,}"),
            (col_b, "Colonnes", f"{result.dataset.columns}"),
            (col_c, "Type de problème", result.problem_type.value),
            (col_d, "Meilleur modèle", result.best_model.name),
        ]:
            col.markdown(
                f'''<div class="ams-card"><h3>{label}</h3><div class="ams-value">{value}</div></div>''',
                unsafe_allow_html=True,
            )
        st.markdown("#### Métriques du meilleur modèle")
        metric_rows = ""
        for k, v in (result.best_model.metrics or {}).items():
            metric_rows += f'''<div class="ams-card" style="margin:6px 0;"><h3>{k}</h3><div class="ams-value">{v:.4f}</div></div>'''
        st.markdown(metric_rows, unsafe_allow_html=True)

    with tabs[1]:
        if not result.eda:
            st.info("Lance d'abord le pipeline.")
        else:
            eda = result.eda
            quality = eda.quality_score or {}
            score = quality.get("score", 0)
            grade = quality.get("grade", "?")
            grade_class = {
                "excellent": "ams-grade-excellent",
                "bon": "ams-grade-bon",
                "moyen": "ams-grade-moyen",
                "faible": "ams-grade-faible",
            }.get(str(grade).lower(), "")
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:14px;margin:6px 0 18px 0;">
                <div style="font-size:42px;font-weight:800;color:var(--ink);line-height:1;">{score}<span style="font-size:14px;color:var(--muted);">/100</span></div>
                <span class="ams-grade {grade_class}">{grade.upper()}</span>
                <div style="color:var(--muted);font-size:13px;">Qualité globale du dataset</div>
                </div>""",
                unsafe_allow_html=True,
            )
            with st.expander("⚠️ Problèmes identifiés", expanded=True):
                issues = quality.get("issues", [])
                if issues:
                    for issue in issues:
                        st.markdown(f"- {issue}")
                else:
                    st.markdown("_Aucun problème détecté._")
            breakdown = quality.get("breakdown", []) or []
            if breakdown:
                with st.expander("🧮 Décomposition du score", expanded=False):
                    st.markdown("Le score part de 100 et applique les pénalités ci-dessous (triées par impact décroissant) :")
                    for delta, label, evidence in breakdown:
                        st.markdown(f"- **−{delta} pts** : `{label}` — {evidence}")
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


st.markdown(
    '''<div class="ams-footer">AutoML Studio · 10 agents spécialisés · EDA · Prétraitement · Modélisation · Explicabilité · Déploiement</div>''',
    unsafe_allow_html=True,
)
