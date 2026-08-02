"""Generate the demo notebook."""
import json
from pathlib import Path

NB = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AutoML Studio demo\n",
                "\n",
                "Reproduction du pipeline complet.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.insert(0, '../backend')\n",
                "from automl.pipeline import AutoMLPipeline\n",
                "from automl.agents.qa_agent import QAAgent\n",
                "import pandas as pd\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pipeline = AutoMLPipeline(source='../data/samples/titanic.csv', target='Survived', optuna_trials=5, cv_folds=3)\n",
                "result = pipeline.run()\n",
                "result.leaderboard[0].name\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pd.DataFrame([{'model': m.name, 'metric': m.metrics, 'time_s': m.training_time_s} for m in result.leaderboard])\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "qa = QAAgent(pd.read_csv('../data/samples/titanic.csv'), target='Survived', leaderboard=result.leaderboard)\n",
                "qa.ask('Pourquoi la cible est-elle desequilibree ?')\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    target = Path(__file__).resolve().parent / "demo.ipynb"
    target.write_text(json.dumps(NB, indent=2), encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
