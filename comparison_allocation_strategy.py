"""
Compare the proposed strategy with two benchmark strategies.

Benchmark strategies
----------
1. Primary-first strategy:
   - Fully satisfy all primary demand points' nominal demand and minimize weighted primary allocation distance.
   - Then use remaining resources to maximize the minimum secondary demand satisfaction rate and finally minimize weighted secondary allocation distance.

2. Equal-ratio allocation strategy:
   - Allocate total available resources to all primary and secondary demand points in proportion to demand
   - Then minimize weighted primary allocation distance and weighted secondary allocation distance lexicographically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

import medical_resource_allocation_robust_cplex_fuzzy_programming as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "comparison_allocation_strategy_output"
WORKBOOK_PATH = OUTPUT_DIR / "comparison_allocation_strategy.xlsx"


def _solve_or_raise(parts: base.ModelParts) -> None:
    """Solve a model and raise a consistent error if no solution is available."""
    parts.model.solve(log_output=False)
    if parts.model.solution is None:
        raise RuntimeError(
            f"No usable solution. CPLEX status = {parts.model.solve_details.status}"
        )


# Primary-first strategy
def solve_primary_first_strategy() -> Dict:
    """Solve the Primary-first strategy with lexicographic tie-breaking."""
    tol = 1e-6
    parts = base.build_base_model(theta=0.0, name="primary_first_strategy")

    for o in base.PRIMARY_DEMANDS:
        parts.model.add_constraint(
            parts.model.sum(parts.x[i, o] for i in base.FACILITIES)
            == base.PRIMARY_DEMAND[o],
            ctname=f"primary_first_full_primary[{o}]",
        )

    parts.model.minimize(parts.distance_expr)
    _solve_or_raise(parts)
    best_primary_distance = float(parts.distance_expr.solution_value)

    parts.model.add_constraint(
        parts.distance_expr <= best_primary_distance + tol,
        ctname="primary_first_fix_primary_distance",
    )
    secondary_min_rate = parts.model.continuous_var(
        lb=0.0,
        ub=1.0,
        name="secondary_min_satisfaction_rate",
    )
    for j in base.DEMANDS:
        parts.model.add_constraint(
            parts.model.sum(parts.z[i, j] for i in base.FACILITIES)
            >= secondary_min_rate * base.SECONDARY_DEMAND[j],
            ctname=f"primary_first_secondary_min_sat[{j}]",
        )

    parts.model.maximize(secondary_min_rate)
    _solve_or_raise(parts)
    best_secondary_min_rate = float(secondary_min_rate.solution_value)

    parts.model.add_constraint(
        secondary_min_rate >= best_secondary_min_rate - tol,
        ctname="primary_first_fix_secondary_min_satisfaction",
    )
    parts.model.minimize(parts.secondary_distance_expr)
    _solve_or_raise(parts)

    sol = base.extract_solution(parts)
    sol["model_type"] = "Primary-first strategy"
    sol["strategy"] = "Primary-first strategy"
    sol["primary_distance_stage_value"] = best_primary_distance
    sol["secondary_min_satisfaction_stage_value"] = best_secondary_min_rate
    sol["secondary_distance_stage_value"] = sol["secondary_distance"]
    return sol


# Equal-ratio allocation strategy
def solve_equal_ratio_allocation_strategy() -> Dict:
    """Solve the Equal-ratio allocation strategy with lexicographic objectives."""
    tol = 1e-6
    parts = base.build_base_model(theta=0.0, name="equal_ratio_allocation_strategy")

    total_capacity = sum(base.CAPACITY.values())
    total_demand = sum(base.PRIMARY_DEMAND.values()) + sum(
        base.SECONDARY_DEMAND.values()
    )
    allocation_ratio = total_capacity / total_demand

    for o in base.PRIMARY_DEMANDS:
        parts.model.add_constraint(
            parts.model.sum(parts.x[i, o] for i in base.FACILITIES)
            == allocation_ratio * base.PRIMARY_DEMAND[o],
            ctname=f"equal_ratio_primary[{o}]",
        )
    for j in base.DEMANDS:
        parts.model.add_constraint(
            parts.model.sum(parts.z[i, j] for i in base.FACILITIES)
            == allocation_ratio * base.SECONDARY_DEMAND[j],
            ctname=f"equal_ratio_secondary[{j}]",
        )

    parts.model.minimize(parts.distance_expr)
    _solve_or_raise(parts)
    best_primary_distance = float(parts.distance_expr.solution_value)

    parts.model.add_constraint(
        parts.distance_expr <= best_primary_distance + tol,
        ctname="equal_ratio_fix_primary_distance",
    )
    parts.model.minimize(parts.secondary_distance_expr)
    _solve_or_raise(parts)

    sol = base.extract_solution(parts)
    sol["model_type"] = "Equal-ratio allocation strategy"
    sol["strategy"] = "Equal-ratio allocation strategy"
    sol["allocation_ratio"] = allocation_ratio
    sol["primary_distance_stage_value"] = best_primary_distance
    sol["secondary_distance_stage_value"] = sol["secondary_distance"]
    return sol


def _strategy_output_row(sol: Dict) -> Dict:
    """Build one workbook row using the main experiment's output style."""
    row = base._build_main_sheet_row(sol)
    row.pop("model_type", None)
    row.pop("mu", None)
    ordered_row = {"strategy": sol.get("strategy")}
    for key, value in row.items():
        ordered_row[key] = value
        if key == "f2_shortage_objective":
            ordered_row["tie_breaking_criterion"] = sol.get("secondary_distance")
    return ordered_row


def main() -> None:
    base.configure_primary_clusters()

    print("Solving strategy: Equitable strategy in this paper")
    equitable_sol = base.solve_fuzzy()
    equitable_sol["model_type"] = "Equitable strategy in this paper"
    equitable_sol["strategy"] = "Equitable strategy in this paper"

    print("Solving strategy: Primary-first strategy")
    primary_first_sol = solve_primary_first_strategy()

    print("Solving strategy: Equal-ratio allocation strategy")
    equal_ratio_sol = solve_equal_ratio_allocation_strategy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame([
        _strategy_output_row(equitable_sol),
        _strategy_output_row(primary_first_sol),
        _strategy_output_row(equal_ratio_sol),
    ])
    base.write_xlsx_standard_library(
        WORKBOOK_PATH,
        {"comparison": comparison_df},
    )
    print(f"Excel workbook saved to: {WORKBOOK_PATH}")


if __name__ == "__main__":
    main()