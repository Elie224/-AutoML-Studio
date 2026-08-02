"""Generate small sample datasets using only the standard library.

This script produces two CSVs (Titanic-style classification and Housing-style
regression) without requiring numpy/pandas to be installed at design time.
"""
from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent / 'samples'


def generate_titanic(n: int = 800, seed: int = 42) -> list[dict]:
    random.seed(seed)
    rows = []
    for i in range(1, n + 1):
        pclass = random.choices([1, 2, 3], weights=[0.25, 0.20, 0.55])[0]
        sex = random.choices(["male", "female"], weights=[0.65, 0.35])[0]
        age = max(0.5, min(80.0, random.gauss(30, 14)))
        if random.random() < 0.20:
            age = ""
        fare = max(0.0, min(600.0, random.gauss(50, 30)))
        sibsp = random.choices([0, 1, 2, 3, 4], weights=[0.55, 0.25, 0.10, 0.05, 0.05])[0]
        parch = random.choices([0, 1, 2, 3], weights=[0.65, 0.20, 0.10, 0.05])[0]
        embarked = random.choices(["C", "Q", "S"], weights=[0.25, 0.10, 0.65])[0]

        logit = -1.5 + 1.2 * (pclass == 3) + 2.5 * (sex == "male") - 0.04 * (age if isinstance(age, float) else 30) + 0.005 * fare - 0.4 * sibsp
        prob = 1.0 / (1.0 + math.exp(-logit))
        survived = 1 if random.random() < prob else 0
        rows.append(
            {
                "PassengerId": i,
                "Pclass": pclass,
                "Sex": sex,
                "Age": age if age != "" else "",
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": round(fare, 2),
                "Embarked": embarked,
                "Survived": survived,
            }
        )
    return rows


def generate_housing(n: int = 600, seed: int = 7) -> list[dict]:
    random.seed(seed)
    rows = []
    for _ in range(n):
        size = max(40, min(350, random.gauss(120, 40)))
        rooms = random.randint(2, 7)
        house_age = max(1, min(120, random.gauss(25, 18)))
        distance = max(0.5, min(30, random.gauss(8, 4)))
        price = 80_000 + 3000 * size + 20_000 * rooms - 1200 * house_age - 5_000 * distance + random.gauss(0, 40_000)
        rows.append(
            {
                "size_m2": round(size, 1),
                "rooms": rooms,
                "house_age": round(house_age, 1),
                "distance_km": round(distance, 2),
                "price": int(round(price)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_csv(HERE / "titanic.csv", generate_titanic())
    write_csv(HERE / "housing.csv", generate_housing())
    print(f"Wrote {len(list(HERE.glob('*.csv')))} sample CSV files in {HERE}")


if __name__ == "__main__":
    main()
