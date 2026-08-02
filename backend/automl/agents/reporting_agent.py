"""Reporting Agent: writes an executive HTML report, a PDF summary and a Jupyter
notebook that reproduces the pipeline end-to-end.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from ..core.config import get_settings
from ..core.schema import PipelineResult


HTML_TEMPLATE = Template("""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<title>Rapport AutoML — {{ result.dataset.name }}</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;background:#f8fafc;color:#0f172a;}
  header{background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff;padding:36px 48px;}
  h1{margin:0;font-size:30px;}
  main{padding:32px 48px;max-width:1100px;margin:auto;}
  section{background:#fff;border-radius:14px;padding:24px 28px;margin-bottom:24px;box-shadow:0 1px 3px rgba(15,23,42,0.08);}
  h2{color:#1d4ed8;border-bottom:2px solid #e0e7ff;padding-bottom:6px;}
  table{border-collapse:collapse;width:100%;margin-top:8px;}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #e2e8f0;font-size:14px;}
  th{background:#eef2ff;}
  .insights li{margin:6px 0;}
  figure img{max-width:100%;border-radius:8px;border:1px solid #e2e8f0;}
  .badge{display:inline-block;padding:4px 10px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:12px;margin-right:6px;}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;}
  .card{background:#f8fafc;border-radius:10px;padding:16px;border:1px solid #e2e8f0;}
  .card h3{margin:0 0 6px 0;font-size:14px;color:#475569;text-transform:uppercase;letter-spacing:.05em;}
  .card p{margin:0;font-size:22px;font-weight:600;color:#1d4ed8;}
</style>
</head>
<body>
<header>
  <h1>Rapport AutoML Studio</h1>
  <p>Dataset <strong>{{ result.dataset.name }}</strong> · {{ result.dataset.rows }} lignes · {{ result.dataset.columns }} colonnes</p>
  <p>Problème détecté : <span class="badge">{{ result.problem_type.value }}</span> Généré le {{ now }}</p>
</header>
<main>
  <section>
    <h2>Vue d'ensemble</h2>
    <div class="grid">
      <div class="card"><h3>Meilleur modèle</h3><p>{{ result.best_model.name }}</p></div>
      <div class="card"><h3>Métrique principale</h3><p>{{ primary_metric }} = {{ primary_value }}</p></div>
      <div class="card"><h3>Temps d'entraînement</h3><p>{{ total_time }} s</p></div>
      <div class="card"><h3>Features post-préprocessing</h3><p>{{ result.preprocessing.drop_columns|length }} supprimées</p></div>
    </div>
  </section>

  <section>
    <h2>Insights EDA</h2>
    {% if result.eda %}
    <ul class="insights">
      {% for insight in result.eda.insights %}<li>{{ insight }}</li>{% endfor %}
    </ul>
    {% else %}<p>Aucun EDA disponible.</p>{% endif %}
  </section>

  <section>
    <h2>Stratégie de prétraitement</h2>
    <ul>
      {% for r in result.preprocessing.rationale %}<li>{{ r }}</li>{% endfor %}
    </ul>
    <p><strong>Imputation :</strong> {{ result.preprocessing.imputation }}</p>
    <p><strong>Encodage :</strong> {{ result.preprocessing.encoders }}</p>
    <p><strong>Scaling :</strong> {{ result.preprocessing.scaling }}</p>
  </section>

  <section>
    <h2>Leaderboard</h2>
    <table>
      <thead><tr><th>Modèle</th><th>Métrique principale</th><th>Autres métriques</th><th>Temps (s)</th></tr></thead>
      <tbody>
        {% for m in result.leaderboard %}
        <tr>
          <td>{{ m.name }}</td>
          <td>{{ "%.4f"|format(m.metrics.get(primary_metric, 0.0)) }}</td>
          <td>{{ m.metrics }}</td>
          <td>{{ m.training_time_s }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Explicabilité (SHAP)</h2>
    {% if result.explainability %}
    <p>Méthode : {{ result.explainability.method }}</p>
    <ol>{% for name, value in result.explainability_top %}
      <li><strong>{{ name }}</strong> — importance {{ "%.4f"|format(value) }}</li>
    {% endfor %}</ol>
    {% else %}<p>Aucune explicabilité calculée.</p>{% endif %}
  </section>

  <section>
    <h2>Figures</h2>
    {% if result.eda %}
      {% for fig in result.eda.figures %}
      <figure><img src="../{{ fig }}" alt="{{ fig }}" /></figure>
      {% endfor %}
    {% endif %}
    {% if result.explainability %}
      {% for fig in result.explainability.figures %}
      <figure><img src="../{{ fig }}" alt="{{ fig }}" /></figure>
      {% endfor %}
    {% endif %}
  </section>
</main>
</body>
</html>
""")


class ReportingAgent:
    name = "reporting"

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.settings = get_settings()
        self.report_dir = self.settings.report_root / dataset_id
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write_html(self, result: PipelineResult) -> Path:
        primary_metric = self._primary_metric(result)
        primary_value = result.best_model.metrics.get(primary_metric, 0.0)
        total_time = sum(m.training_time_s for m in result.leaderboard)
        explainability_top = []
        if result.explainability:
            explainability_top = list(result.explainability.feature_importance.items())[:10]
        html = HTML_TEMPLATE.render(
            result=result,
            now=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            primary_metric=primary_metric,
            primary_value=f"{primary_value:.4f}",
            total_time=f"{total_time:.1f}",
            explainability_top=explainability_top,
        )
        path = self.report_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        return path

    def write_pdf(self, result: PipelineResult) -> Path | None:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
        except Exception:
            return None
        path = self.report_dir / "report.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 50
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, "Rapport AutoML Studio")
        y -= 40
        c.setFont("Helvetica", 11)
        lines = [
            f"Dataset : {result.dataset.name}",
            f"Type de probleme : {result.problem_type.value}",
            f"Lignes x Colonnes : {result.dataset.rows} x {result.dataset.columns}",
            "",
            f"Meilleur modele : {result.best_model.name}",
        ]
        for metric, value in result.best_model.metrics.items():
            lines.append(f"  - {metric}: {value:.4f}")
        for line in lines:
            c.drawString(40, y, line)
            y -= 16
        c.showPage()
        c.save()
        return path

    def write_notebook(self, result: PipelineResult) -> Path:
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Reproduction AutoML Studio\n",
                    f"\n",
                    f"Dataset : **{result.dataset.name}**\n",
                    f"Type : `{result.problem_type.value}`\n",
                    f"Meilleur modèle : **{result.best_model.name}**\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "from automl import AutoMLPipeline\n",
                    f"pipeline = AutoMLPipeline('data/{result.dataset.name}', target={result.dataset.target_column!r})\n",
                    "result = pipeline.run()\n",
                    "result.leaderboard.head()\n",
                ],
            },
        ]
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = self.notebook_root() / "reproduce.ipynb"
        path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        return path

    def write_executive(self, result: PipelineResult) -> Path:
        primary_metric = self._primary_metric(result)
        primary_value = result.best_model.metrics.get(primary_metric, 0.0)
        path = self.report_dir / "executive_summary.md"
        path.write_text(
            f"""# Résumé exécutif — {result.dataset.name}

- **Type de problème** : {result.problem_type.value}
- **Dataset** : {result.dataset.rows} lignes × {result.dataset.columns} colonnes
- **Meilleur modèle** : {result.best_model.name}
- **Métrique principale** : {primary_metric} = {primary_value:.4f}

## Top 3 modèles
""" + "\n".join(
                f"{i+1}. **{m.name}** — {primary_metric} = {m.metrics.get(primary_metric, 0):.4f}"
                for i, m in enumerate(result.leaderboard[:3])
            ),
            encoding="utf-8",
        )
        return path

    def notebook_root(self) -> Path:
        path = self.settings.notebook_root / self.dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _primary_metric(self, result: PipelineResult) -> str:
        return {
            "classification": "f1_weighted",
            "regression": "r2",
            "clustering": "silhouette",
            "anomaly_detection": "n_anomalies",
            "time_series": "mape",
        }.get(result.problem_type.value, "score")




