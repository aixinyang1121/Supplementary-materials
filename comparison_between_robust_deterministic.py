"""
Compare the robust model with the deterministic model.

The script solves both models, evaluates their fixed allocation decisions under nominal parameters and 3000 sampled out-of-sample uncertainty scenarios, and exports the comparison workbook.
"""

from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import pandas as pd

import medical_resource_allocation_robust_cplex_fuzzy_programming as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "robust_deterministic_comparison_output"
WORKBOOK_PATH = OUTPUT_DIR / "robust_deterministic_comparison.xlsx"
DEFAULT_N_SCENARIOS = 3000
RANDOM_SEED = 2026
FEASIBILITY_TOL = 1e-6
NOMINAL_SCENARIO = {
    "primary_demand": base.PRIMARY_DEMAND,
    "secondary_demand": base.SECONDARY_DEMAND,
    "primary_distance": base.PRIMARY_DISTANCE,
}

def _sample_uncertainty_scenarios() -> Iterator[Dict]:
    """Sample demand and primary-distance realizations from their uncertainty sets."""
    rng = np.random.default_rng(RANDOM_SEED)
    primary_distance_keys = [
        (i, o) for i in base.FACILITIES for o in base.PRIMARY_DEMANDS
    ]
    for scenario_id in range(1, DEFAULT_N_SCENARIOS + 1):
        primary_demand = {
            o: base.PRIMARY_DEMAND[o] * (1.0 + base.DEFAULT_LAMBDA_W * rng.random())
            for o in base.PRIMARY_DEMANDS
        }
        secondary_demand = {
            j: base.SECONDARY_DEMAND[j] * (1.0 + base.DEFAULT_LAMBDA_D * rng.random())
            for j in base.DEMANDS
        }

        distance_xi_values = rng.uniform(-1.0, 1.0, len(primary_distance_keys))
        primary_distance = {
            key: base.PRIMARY_DISTANCE[key]
            * (1.0 + base.DEFAULT_LAMBDA_DISTANCE * float(distance_xi_values[idx]))
            for idx, key in enumerate(primary_distance_keys)
        }
        yield {
            "scenario_id": scenario_id,
            "primary_demand": primary_demand,
            "secondary_demand": secondary_demand,
            "primary_distance": primary_distance,
        }


def _evaluate_solution_under_scenario(
    sol: Dict,
    scenario: Dict,
    model_label: str,
) -> Dict:
    """Evaluate one fixed solution under one sampled deterministic scenario."""
    primary_alloc_by_demand = sol["primary_alloc_by_demand"]
    secondary_alloc_by_demand = sol["secondary_alloc_by_demand"]
    primary_demand = scenario["primary_demand"]
    secondary_demand = scenario["secondary_demand"]
    primary_total_demand = sum(primary_demand.values())
    secondary_total_demand = sum(secondary_demand.values())
    primary_alloc = sol["primary_alloc"]
    secondary_alloc = sol["secondary_alloc"]

    primary_satisfaction_by_demand = {
        o: primary_alloc_by_demand[o] / primary_demand[o]
        for o in base.PRIMARY_DEMANDS
    }
    secondary_satisfaction_by_demand = {
        j: secondary_alloc_by_demand[j] / secondary_demand[j]
        for j in base.DEMANDS
    }
    R_P = primary_alloc / primary_total_demand
    R_S = secondary_alloc / secondary_total_demand
    R_P_min = min(primary_satisfaction_by_demand.values())
    R_S_min = min(secondary_satisfaction_by_demand.values())

    objective_1 = sum(
        scenario["primary_distance"][i, o] * sol["x"].get((i, o), 0.0)
        for i in base.FACILITIES for o in base.PRIMARY_DEMANDS
    )
    objective_2 = (
        base.DEFAULT_ETA * (primary_total_demand - primary_alloc)
        + (secondary_total_demand - secondary_alloc)
    )
    checks = {
        "primary_min_ok": all(
            primary_alloc_by_demand[o] + FEASIBILITY_TOL
            >= base.DEFAULT_THETA * primary_demand[o]
            for o in base.PRIMARY_DEMANDS
        ),
        "secondary_min_ok": all(
            secondary_alloc_by_demand[j] + FEASIBILITY_TOL
            >= base.DEFAULT_THETA * secondary_demand[j]
            for j in base.DEMANDS
        ),
        "primary_equity_ok": all(
            primary_alloc_by_demand[o] + FEASIBILITY_TOL
            >= (primary_alloc / primary_total_demand - base.DEFAULT_DELTA)
            * primary_demand[o]
            and primary_alloc_by_demand[o]
            <= (primary_alloc / primary_total_demand + base.DEFAULT_DELTA)
            * primary_demand[o]
            + FEASIBILITY_TOL
            for o in base.PRIMARY_DEMANDS
        ),
        "secondary_equity_ok": all(
            secondary_alloc_by_demand[j] + FEASIBILITY_TOL
            >= (secondary_alloc / secondary_total_demand - base.DEFAULT_DELTA)
            * secondary_demand[j]
            and secondary_alloc_by_demand[j]
            <= (secondary_alloc / secondary_total_demand + base.DEFAULT_DELTA)
            * secondary_demand[j]
            + FEASIBILITY_TOL
            for j in base.DEMANDS
        ),
    }
    checks["feasible"] = all(checks.values())

    row = {
        "model": model_label,
        **checks,
        "objective_1_value": objective_1,
        "objective_2_value": objective_2,
        "R_P": R_P,
        "R_S": R_S,
        "R_P_min": R_P_min,
        "R_S_min": R_S_min,
        "Gap_R": abs(R_P - R_S),
        "Gap_min": abs(R_P_min - R_S_min),
        "Gap_P": abs(R_P - R_P_min),
        "Gap_S": abs(R_S - R_S_min),
    }
    return row


def export_workbook(
    solutions: Dict[str, Dict],
    nominal_evaluations: Dict[str, Dict],
    rows: list[Dict],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_order = ("Robust model", "Deterministic model")
    scenario_df = pd.DataFrame(rows)
    scenario_df["objective_1_difference_from_nominal"] = (
        scenario_df["objective_1_value"]
        - scenario_df["model"].map(
            {
                model: evaluation["objective_1_value"]
                for model, evaluation in nominal_evaluations.items()
            }
        )
    )
    mean_indicators = [
        ("Mean Objective 1 value", "objective_1_value"),
        ("Mean Delta f1 value", "objective_1_difference_from_nominal"),
        ("Mean R_P value", "R_P"),
        ("Mean R_S value", "R_S"),
        ("Mean R_P^min value", "R_P_min"),
        ("Mean R_S^min value", "R_S_min"),
        ("Mean Gap_R value", "Gap_R"),
        ("Mean Gap_min value", "Gap_min"),
        ("Mean Gap_P value", "Gap_P"),
        ("Mean Gap_S value", "Gap_S"),
    ]
    minimum_indicators = [
        ("Minimum R_P value over scenarios", "R_P"),
        ("Minimum R_S value over scenarios", "R_S"),
        ("Minimum R_P^min value over scenarios", "R_P_min"),
        ("Minimum R_S^min value over scenarios", "R_S_min"),
    ]
    maximum_indicators = [
        ("Maximum Gap_R value over scenarios", "Gap_R"),
        ("Maximum Gap_min value over scenarios", "Gap_min"),
        ("Maximum Gap_P value over scenarios", "Gap_P"),
        ("Maximum Gap_S value over scenarios", "Gap_S"),
    ]
    out_of_sample_rows = []
    for model in model_order:
        model_df = scenario_df.loc[scenario_df["model"] == model]
        metric_df = model_df.loc[model_df["feasible"]]
        if metric_df.empty:
            raise RuntimeError(
                f"No feasible sampled scenarios for {model}; cannot summarize "
                "indicators over feasible scenarios."
            )
        out_of_sample_rows.append(
            {
                "model": model,
                "Constraint violation probability": (
                    1.0 - float(model_df["feasible"].mean())
                ),
                "Minimum demand satisfaction rate violation probability": float(
                    (
                        (~model_df["primary_min_ok"])
                        | (~model_df["secondary_min_ok"])
                    ).mean()
                ),
                "Intra-group equity violation probability": float(
                    (
                        (~model_df["primary_equity_ok"])
                        | (~model_df["secondary_equity_ok"])
                    ).mean()
                ),
                "Primary equity violation probability": float(
                    (~model_df["primary_equity_ok"]).mean()
                ),
                "Secondary equity violation probability": float(
                    (~model_df["secondary_equity_ok"]).mean()
                ),
                **{
                    indicator: float(metric_df[column].mean())
                    for indicator, column in mean_indicators
                },
                **{
                    indicator: float(metric_df[column].min())
                    for indicator, column in minimum_indicators
                },
                **{
                    indicator: float(metric_df[column].max())
                    for indicator, column in maximum_indicators
                },
            }
        )

    nominal_rows = []
    for model in model_order:
        sol = solutions[model]
        evaluation = nominal_evaluations[model]
        evaluated_sol = {
            **sol,
            "model_type": model,
            "f1_distance_objective": evaluation["objective_1_value"],
            "f2_shortage_objective": evaluation["objective_2_value"],
            "primary_satisfaction": evaluation["R_P"],
            "primary_satisfaction_by_demand": {
                o: sol["primary_alloc_by_demand"][o] / base.PRIMARY_DEMAND[o]
                for o in base.PRIMARY_DEMANDS
            },
            "primary_min_satisfaction": evaluation["R_P_min"],
            "secondary_satisfaction": evaluation["R_S"],
            "secondary_satisfaction_by_demand": {
                j: sol["secondary_alloc_by_demand"][j] / base.SECONDARY_DEMAND[j]
                for j in base.DEMANDS
            },
            "secondary_min_satisfaction": evaluation["R_S_min"],
        }
        main_row = base._build_main_sheet_row(evaluated_sol)
        main_row.pop("model_type")
        main_row.pop("mu")
        output_row = {"model": model}
        for key, value in main_row.items():
            output_row[key] = value
            if key == "f2_shortage_objective":
                output_row["tie_breaking_criterion"] = sol["secondary_distance"]
        nominal_rows.append(output_row)

    base.write_xlsx_standard_library(
        WORKBOOK_PATH,
        {
            "nominal": pd.DataFrame(nominal_rows),
            "out_of_sample": pd.DataFrame(out_of_sample_rows),
        },
    )
    print(f"Excel workbook saved to: {WORKBOOK_PATH}")


def main() -> None:
    print("Solving robust model")
    solutions = {"Robust model": base.solve_fuzzy()}
    print("Solving deterministic model")
    solutions["Deterministic model"] = base.solve_fuzzy(
        robust_distance=False,
        gamma=0.0,
        lambda_distance=0.0,
        robust_demand=False,
        lambda_W=0.0,
        lambda_D=0.0,
    )
    nominal_evaluations = {
        model: _evaluate_solution_under_scenario(
            sol,
            NOMINAL_SCENARIO,
            model,
        )
        for model, sol in solutions.items()
    }

    rows = []
    for scenario in _sample_uncertainty_scenarios():
        print(f"Evaluating out-of-sample sample {scenario['scenario_id']}/{DEFAULT_N_SCENARIOS}")
        for model, sol in solutions.items():
            rows.append(
                _evaluate_solution_under_scenario(
                    sol,
                    scenario,
                    model,
                )
            )
    export_workbook(
        solutions,
        nominal_evaluations,
        rows,
    )


if __name__ == "__main__":
    main()