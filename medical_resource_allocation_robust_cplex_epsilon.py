"""
Epsilon-constraint analysis for the medical resource allocation model.

This script reuses the main model implementation in `medical_resource_allocation_robust_cplex_fuzzy_programming.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

import medical_resource_allocation_robust_cplex_fuzzy_programming as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "epsilon_main_experiment_output"
WORKBOOK_PATH = OUTPUT_DIR / "epsilon_constraint_results.xlsx"
EPSILON_GRID_SIZE = 100


def _epsilon_values(start: float, stop: float) -> list[float]:
    """Return evenly spaced epsilon values from start to stop inclusive."""
    if EPSILON_GRID_SIZE <= 1:
        return [float(start)]
    step = (stop - start) / (EPSILON_GRID_SIZE - 1)
    return [start + step * k for k in range(EPSILON_GRID_SIZE)]


def solve_epsilon_constraint(
    epsilon: float,
) -> Dict:
    """Minimize objective 1 subject to objective 2 <= epsilon."""
    parts = base.build_base_model(
        name=f"epsilon_constraint_{epsilon:.6f}",
    )
    parts.model.add_constraint(
        parts.eta_shortage_expr <= epsilon,
        ctname="epsilon_shortage_bound",
    )
    parts.model.minimize(parts.distance_expr)
    parts.model.solve(log_output=False)
    sol = base.extract_solution(parts)
    sol["model_type"] = "epsilon_constraint"
    sol["epsilon"] = epsilon
    return sol


def run_epsilon_grid(
    epsilon_min: float,
    epsilon_max: float,
) -> pd.DataFrame:
    """Solve the epsilon-constraint model over a shortage epsilon grid."""
    rows = []
    for idx, epsilon in enumerate(_epsilon_values(epsilon_min, epsilon_max), start=1):
        print(f"Solving epsilon {idx}/{EPSILON_GRID_SIZE}")
        try:
            sol = solve_epsilon_constraint(
                epsilon=epsilon,
            )
        except (RuntimeError, ValueError):
            continue
        row = base._build_main_sheet_row(sol)
        row.pop("model_type", None)
        row.pop("mu", None)
        rows.append({"epsilon": epsilon, **row})
    return pd.DataFrame(rows)


def main() -> None:
    base.configure_primary_clusters()

    sol_distance = base.solve_single_objective(
        "distance",
    )

    sol_shortage = base.solve_single_objective(
        "shortage",
    )

    epsilon_min = sol_shortage["f2_shortage_objective"]
    epsilon_max = sol_distance["f2_shortage_objective"]

    epsilon_df = run_epsilon_grid(
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.write_xlsx_standard_library(
        WORKBOOK_PATH,
        {"epsilon": epsilon_df},
    )
    print(f"Excel workbook saved to: {WORKBOOK_PATH}")


if __name__ == "__main__":
    main()