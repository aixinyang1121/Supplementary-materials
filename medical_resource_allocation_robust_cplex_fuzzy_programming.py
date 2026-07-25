"""
Medical resource allocation robust optimization with fuzzy programming.
Python + CPLEX implementation.

1. Single-objective models:
   - Minimize weighted primary allocation distance.
   - Minimize weighted unmet medical demand.
2. Fuzzy multi-objective model:
   - Maximize membership degree mu.
   - Fuzzy bounds are computed from the two single-objective models.
3. Sensitivity analysis over the ethical parameter eta, the equity tolerance parameter delta and the minimum demand satisfaction rate theta.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, Literal, Optional, Tuple
import zipfile
from xml.sax.saxutils import escape

import population_weighted_Kmeans as kmeans

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "main_experiment_output"
WORKBOOK_PATH = OUTPUT_DIR / "main_experiment_results.xlsx"
CLUSTER_PLOT_PATH = OUTPUT_DIR / "population_weighted_kmeans_k3_clusters.png"


import numpy as np
import pandas as pd
from docplex.mp.model import Model


# ---------------------------------------------------------------------
# 1. Data input
# ---------------------------------------------------------------------

FACILITIES = list(range(1, 28))  # 1,2,...,27
DEMANDS = list(range(1, 13))     # 1,2,...,12 corresponding to A,B,...,L
PRIMARY_CLUSTER_COUNT = 3
PRIMARY_DEMANDS = list(range(1, PRIMARY_CLUSTER_COUNT + 1))
DEFAULT_ETA = 1.2
DEFAULT_DELTA = 0.1
DEFAULT_THETA = 0.4
DEFAULT_LAMBDA_DISTANCE = 0.1
DEFAULT_LAMBDA_W = 0.1
DEFAULT_LAMBDA_D = 0.1

# Information of representative demand points.
DEMAND_NAMES = {
    1: "A_Wusong_Street",
    2: "B_Songnan_Town",
    3: "C_Dachang_Town",
    4: "D_Gucun_Town",
    5: "E_Miaohang_Town",
    6: "F_Gaojing_Town",
    7: "G_Youyi_Road_Street",
    8: "H_Zhangmiao_Street",
    9: "I_Luojing_Town",
    10: "J_Luodian_Town",
    11: "K_Yanghang_Town",
    12: "L_Yuepu_Town",
}

# Nominal secondary medical demand \bar D_j.
SECONDARY_DEMAND = {
    1: 384.0,
    2: 392.0,
    3: 885.0,
    4: 613.0,
    5: 225.0,
    6: 414.0,
    7: 390.0,
    8: 673.0,
    9: 171.0,
    10: 398.0,
    11: 300.0,
    12: 325.0,
}

# Healthcare facility capacity c_i.
CAPACITY = {
    1: 93.0,
    2: 539.0,
    3: 890.0,
    4: 124.0,
    5: 800.0,
    6: 1146.0,
    7: 76.0,
    8: 122.0,
    9: 140.0,
    10: 1028.0,
    11: 251.0,
    12: 154.0,
    13: 1104.0,
    14: 360.0,
    15: 670.0,
    16: 163.0,
    17: 130.0,
    18: 133.0,
    19: 526.0,
    20: 158.0,
    21: 143.0,
    22: 124.0,
    23: 99.0,
    24: 143.0,
    25: 72.0,
    26: 184.0,
    27: 129.0,
}

# Healthcare facility locations.
FACILITY_LOCATION = {
    1: [121.332, 31.484],
    2: [121.346, 31.415],
    3: [121.428, 31.414],
    4: [121.426, 31.419],
    5: [121.473, 31.411],
    6: [121.482, 31.412],
    7: [121.482, 31.412],
    8: [121.356, 31.386],
    9: [121.483, 31.385],
    10: [121.492, 31.380],
    11: [121.440, 31.373],
    12: [121.404, 31.354],
    13: [121.369, 31.348],
    14: [121.482, 31.346],
    15: [121.438, 31.334],
    16: [121.485, 31.344],
    17: [121.424, 31.328],
    18: [121.458, 31.325],
    19: [121.388, 31.309],
    20: [121.413, 31.289],
    21: [121.491, 31.404],
    22: [121.440, 31.342],
    23: [121.372, 31.300],
    24: [121.377, 31.319],
    25: [121.399, 31.454],
    26: [121.367, 31.354],
    27: [121.454, 31.339],
}

# Distance matrix from facilities to secondary demand points.
SECONDARY_DISTANCE_ROWS = {
    1:  [19.4, 20.5, 20.3, 14.9, 19.1, 22.1, 16.9, 19.5, 1.3, 8.7, 15.0, 9.4],
    2:  [14.6, 14.7, 12.7, 7.3, 11.9, 15.7, 13.1, 12.8, 6.8, 1.1, 9.4, 6.8],
    3:  [7.5, 8.8, 12.2, 8.1, 9.2, 10.8, 5.4, 8.6, 10.6, 7.1, 3.5, 2.7],
    4:  [7.9, 9.3, 12.6, 8.4, 9.7, 11.3, 5.6, 9.1, 10.2, 7.1, 4.0, 2.2],
    5:  [4.3, 7.0, 13.7, 11.0, 10.0, 9.6, 1.3, 8.5, 14.4, 11.4, 4.5, 6.2],
    6:  [4.0, 7.1, 14.3, 11.7, 10.5, 9.7, 1.0, 8.9, 15.0, 12.3, 5.2, 7.0],
    7:  [4.0, 7.1, 14.3, 11.7, 10.5, 9.7, 1.0, 8.9, 15.1, 12.3, 5.2, 7.0],
    8:  [13.0, 12.4, 9.3, 4.0, 8.9, 13.0, 12.2, 10.1, 10.1, 2.4, 7.8, 7.7],
    9:  [1.3, 4.1, 12.1, 10.6, 8.2, 6.8, 2.0, 6.3, 16.8, 12.6, 4.3, 8.6],
    10: [0.4, 3.7, 12.3, 11.1, 8.4, 6.4, 2.7, 6.4, 17.7, 13.5, 5.1, 9.6],
    11: [5.1, 4.6, 8.4, 6.2, 4.8, 6.2, 5.3, 3.9, 14.8, 9.1, 1.3, 7.4],
    12: [8.8, 7.1, 5.2, 2.5, 3.2, 7.3, 9.2, 4.3, 14.8, 7.7, 4.6, 9.1],
    13: [12.2, 10.4, 5.0, 1.1, 5.6, 10.2, 12.4, 7.4, 14.5, 6.7, 7.6, 10.6],
    14: [3.6, 0.3, 9.4, 9.9, 5.7, 2.5, 6.3, 3.5, 19.6, 14.0, 5.9, 11.8],
    15: [7.1, 4.2, 5.1, 6.1, 1.3, 3.5, 8.8, 0.9, 18.2, 11.4, 5.5, 11.4],
    16: [3.7, 0.7, 9.6, 10.2, 5.9, 2.4, 6.6, 3.7, 20.0, 14.3, 6.3, 12.2],
    17: [8.5, 5.6, 3.7, 5.3, 0.3, 4.7, 10.0, 2.3, 18.2, 11.1, 6.3, 11.9],
    18: [6.6, 3.2, 6.5, 8.3, 3.3, 1.4, 8.9, 1.8, 20.1, 13.5, 6.8, 12.9],
    19: [12.5, 9.7, 0.5, 5.2, 4.3, 8.3, 13.9, 6.4, 19.2, 11.5, 9.6, 14.3],
    20: [12.4, 9.1, 2.8, 8.0, 4.8, 7.0, 14.3, 6.3, 21.9, 14.3, 10.8, 16.2],
    21: [3.0, 6.3, 14.1, 12.0, 10.3, 8.9, 0.8, 8.4, 16.2, 13.1, 5.5, 8.1],
    22: [6.4, 3.8, 5.8, 6.0, 1.9, 3.7, 7.9, 0.7, 17.6, 11.0, 4.7, 10.7],
    23: [14.3, 11.4, 2.2, 6.1, 6.0, 10.0, 15.5, 8.2, 19.8, 12.0, 11.2, 15.5],
    24: [12.7, 10.2, 1.9, 4.0, 4.6, 9.1, 13.7, 6.8, 17.8, 10.1, 9.2, 13.4],
    25: [12.4, 14.0, 16.2, 11.2, 13.8, 16.0, 9.8, 13.6, 5.9, 6.8, 8.6, 2.5],
    26: [12.2, 10.6, 5.6, 1.0, 6.0, 10.6, 12.3, 7.7, 13.8, 6.1, 7.5, 10.1],
    27: [5.7, 2.6, 6.7, 7.4, 2.9, 2.4, 7.7, 0.7, 18.6, 12.2, 5.2, 11.4],
}
SECONDARY_DISTANCE = {
    (i, j): float(SECONDARY_DISTANCE_ROWS[i][j - 1])
    for i in FACILITIES for j in DEMANDS
}

# Values of intermediate variables a_ij.
COVERAGE_ROWS = {
    1:  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    2:  [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    3:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    4:  [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    5:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    6:  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    7:  [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    8:  [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    9:  [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    10: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    11: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    12: [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    13: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    14: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    15: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    16: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
    17: [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    18: [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    19: [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    20: [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    21: [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    22: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    23: [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    24: [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    25: [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
    26: [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    27: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
}
COVERAGE = {(i, j): int(COVERAGE_ROWS[i][j - 1]) for i in FACILITIES for j in DEMANDS}

# Nominal total primary medical demand \bar W.
W_PRIMARY = 8205


def build_primary_demand_clusters(
    n_clusters: int = PRIMARY_CLUSTER_COUNT,
) -> tuple[dict[int, float], dict[int, list[float]], dict[int, list[int]]]:
    """Build primary demand points from population-weighted K-means."""
    demand_ids = kmeans.DEMAND_IDS
    points = kmeans.DEMAND_POINTS
    weights = kmeans.POPULATION_WEIGHTS
    labels, centers, _ = kmeans.weighted_kmeans(
        points,
        weights,
        n_clusters=n_clusters,
    )

    population_by_cluster = {}
    location_by_cluster = {}
    members_by_cluster = {}
    for cluster_id in range(n_clusters):
        member_indices = np.where(labels == cluster_id)[0]
        member_ids = [demand_ids[idx] for idx in member_indices]
        center = centers[cluster_id]
        demand_id = cluster_id + 1
        population_by_cluster[demand_id] = float(weights[member_indices].sum())
        location_by_cluster[demand_id] = [float(center[0]), float(center[1])]
        members_by_cluster[demand_id] = member_ids
    return population_by_cluster, location_by_cluster, members_by_cluster


PRIMARY_DEMAND_POPULATION, PRIMARY_DEMAND_LOCATION, PRIMARY_DEMAND_MEMBERS = (
    build_primary_demand_clusters(PRIMARY_CLUSTER_COUNT)
)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in kilometers between two lon/lat points."""
    radius_km = 6371.0088
    lon1_rad, lat1_rad = math.radians(lon1), math.radians(lat1)
    lon2_rad, lat2_rad = math.radians(lon2), math.radians(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _refresh_primary_demand_parameters() -> None:
    """Refresh primary demand quantities and facility-to-primary distances."""
    global TOTAL_PRIMARY_CLUSTER_POPULATION
    global PRIMARY_DEMAND
    global PRIMARY_DISTANCE

    TOTAL_PRIMARY_CLUSTER_POPULATION = sum(PRIMARY_DEMAND_POPULATION.values())
    PRIMARY_DEMAND = {
        o: W_PRIMARY * PRIMARY_DEMAND_POPULATION[o] / TOTAL_PRIMARY_CLUSTER_POPULATION
        for o in PRIMARY_DEMANDS
    }
    PRIMARY_DISTANCE = {
        (i, o): haversine_km(
            FACILITY_LOCATION[i][0],
            FACILITY_LOCATION[i][1],
            PRIMARY_DEMAND_LOCATION[o][0],
            PRIMARY_DEMAND_LOCATION[o][1],
        )
        for i in FACILITIES for o in PRIMARY_DEMANDS
    }


_refresh_primary_demand_parameters()


# Primary demand point information.
def configure_primary_clusters() -> None:
    """Configure primary demand points from population-weighted K-means."""
    global PRIMARY_CLUSTER_COUNT
    global PRIMARY_DEMANDS
    global PRIMARY_DEMAND_POPULATION
    global PRIMARY_DEMAND_LOCATION
    global PRIMARY_DEMAND_MEMBERS

    PRIMARY_DEMANDS = list(range(1, PRIMARY_CLUSTER_COUNT + 1))
    PRIMARY_DEMAND_POPULATION, PRIMARY_DEMAND_LOCATION, PRIMARY_DEMAND_MEMBERS = (
        build_primary_demand_clusters(PRIMARY_CLUSTER_COUNT)
    )
    _refresh_primary_demand_parameters()


# Plot primary demand point location.
def export_primary_cluster_plot() -> None:
    """Export the k=3 population-weighted K-means clustering result."""
    labels, centers, _ = kmeans.weighted_kmeans(
        kmeans.DEMAND_POINTS,
        kmeans.POPULATION_WEIGHTS,
        n_clusters=PRIMARY_CLUSTER_COUNT,
    )
    kmeans.plot_cluster_map(
        kmeans.DEMAND_IDS,
        kmeans.DEMAND_POINTS,
        kmeans.POPULATION_WEIGHTS,
        labels,
        centers,
        CLUSTER_PLOT_PATH,
    )
    print(f"k=3 cluster plot saved to: {CLUSTER_PLOT_PATH}")


@dataclass
class FuzzyBounds:
    distance_best: float
    distance_worst: float
    shortage_best: float
    shortage_worst: float

    @property
    def f1_scale(self) -> float:
        return self.distance_worst - self.distance_best

    @property
    def f2_scale(self) -> float:
        return self.shortage_worst - self.shortage_best

    @property
    def distance_scale(self) -> float:
        return self.f1_scale

    @property
    def shortage_scale(self) -> float:
        return self.f2_scale


@dataclass
class ModelParts:
    model: Model
    x: Dict[Tuple[int, int], object]
    z: Dict[Tuple[int, int], object]
    eta_shortage_expr: object
    distance_expr: object
    secondary_distance_expr: object
    W_eff: Dict[int, float]
    D_eff: Dict[int, float]
    capacity: Dict[int, float]


# ---------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------

def build_base_model(
    eta: float = DEFAULT_ETA,
    delta: float = DEFAULT_DELTA,
    theta: float = DEFAULT_THETA,
    include_secondary_shortage: bool = True,
    lambda_W: float = DEFAULT_LAMBDA_W,
    lambda_D: float = DEFAULT_LAMBDA_D,
    gamma: Optional[float] = None,
    lambda_distance: float = DEFAULT_LAMBDA_DISTANCE,
    name: str = "medical_resource_allocation",
    **model_options,
) -> ModelParts:


    robust_demand = bool(model_options.pop("robust_demand", True))
    robust_distance = bool(model_options.pop("robust_distance", True))
    if model_options:
        unexpected = ", ".join(sorted(model_options))
        raise TypeError(f"Unexpected build_base_model option(s): {unexpected}")

    if gamma is None:
        gamma = 0.5 * len(FACILITIES) * len(PRIMARY_DEMANDS)

    m = Model(name)
    capacity = dict(CAPACITY)

    # Maximum medical demand within the uncertainty sets.
    W_eff = {
        o: PRIMARY_DEMAND[o] * (1.0 + lambda_W if robust_demand else 1.0)
        for o in PRIMARY_DEMANDS
    }
    D_eff = {
        j: SECONDARY_DEMAND[j] * (1.0 + lambda_D if robust_demand else 1.0)
        for j in DEMANDS
    }

    # Decision variables.
    # x_i,o: amount of medical resources reserved at facility i for primary demand point o.
    x = m.continuous_var_dict(
        [(i, o) for i in FACILITIES for o in PRIMARY_DEMANDS],
        lb=0.0,
        name="x",
    )

    # z_i,j: amount of medical resources reserved at facility i for secondary demand point j.
    z = m.continuous_var_dict(
        [(i, j) for i in FACILITIES for j in DEMANDS],
        lb=0.0,
        name="z",
    )

    # Constraints.
    # Facility capacity constraint.
    for i in FACILITIES:
        m.add_constraint(
            m.sum(x[i, o] for o in PRIMARY_DEMANDS)
            + m.sum(z[i, j] for j in DEMANDS)
            <= capacity[i],
            ctname=f"capacity[{i}]",
        )

    # Demand fulfillment constraints.
    for o in PRIMARY_DEMANDS:
        m.add_constraint(
            m.sum(x[i, o] for i in FACILITIES) <= PRIMARY_DEMAND[o],
            ctname=f"primary_demand_ub[{o}]",
        )
    for j in DEMANDS:
        m.add_constraint(
            m.sum(z[i, j] for i in FACILITIES) <= SECONDARY_DEMAND[j],
            ctname=f"secondary_demand_ub[{j}]",
        )

    # Coverage constraint.
    for i in FACILITIES:
        for j in DEMANDS:
            m.add_constraint(
                z[i, j] <= COVERAGE[i, j] * 10000,
                ctname=f"coverage[{i},{j}]",
            )

    # Minimum demand satisfaction constraints.
    for o in PRIMARY_DEMANDS:
        m.add_constraint(
            m.sum(x[i, o] for i in FACILITIES) >= theta * W_eff[o],
            ctname=f"primary_min_sat[{o}]",
        )
    for j in DEMANDS:
        m.add_constraint(
            m.sum(z[i, j] for i in FACILITIES) >= theta * D_eff[j],
            ctname=f"secondary_min_sat[{j}]",
        )

    primary_alloc = m.sum(x[i, o] for i in FACILITIES for o in PRIMARY_DEMANDS)
    secondary_alloc = m.sum(z[i, j] for i in FACILITIES for j in DEMANDS)
    total_primary_nominal_demand = sum(PRIMARY_DEMAND.values())
    total_secondary_nominal_demand = sum(SECONDARY_DEMAND.values())
    total_primary_effective_demand = sum(W_eff.values())
    total_secondary_effective_demand = sum(D_eff.values())

    # Intra-group equity constraints.
    for o in PRIMARY_DEMANDS:
        primary_o_alloc = m.sum(x[i, o] for i in FACILITIES)
        m.add_constraint(
            primary_o_alloc
            >= (primary_alloc / total_primary_nominal_demand - delta) * W_eff[o],
            ctname=f"primary_equity[{o}]_1",
        )
        m.add_constraint(
            primary_o_alloc
            <= (primary_alloc / total_primary_effective_demand + delta)
            * PRIMARY_DEMAND[o],
            ctname=f"primary_equity[{o}]_2",
        )

    for j in DEMANDS:
        secondary_j_alloc = m.sum(z[i, j] for i in FACILITIES)
        m.add_constraint(
            secondary_j_alloc
            >= (secondary_alloc / total_secondary_nominal_demand - delta)
            * D_eff[j],
            ctname=f"secondary_equity[{j}]_1",
        )
        m.add_constraint(
            secondary_j_alloc
            <= (secondary_alloc / total_secondary_effective_demand + delta)
            * SECONDARY_DEMAND[j],
            ctname=f"secondary_equity[{j}]_2",
        )

    primary_unmet = total_primary_nominal_demand - primary_alloc
    secondary_unmet = total_secondary_nominal_demand - secondary_alloc
    eta_primary_unmet = eta * primary_unmet
    eta_shortage_expr = (
        eta_primary_unmet + secondary_unmet
        if include_secondary_shortage
        else eta_primary_unmet
    )

    if robust_distance:
        # Robust conterpart for accessibility uncertainty.
        f = m.continuous_var(lb=0.0, name="robust_distance_f")
        q = m.continuous_var(lb=0.0, name="robust_distance_q")
        p = m.continuous_var_dict(
            [(i, o) for i in FACILITIES for o in PRIMARY_DEMANDS],
            lb=0.0,
            name="robust_distance_p",
        )
        m.add_constraint(
            m.sum(
                PRIMARY_DISTANCE[i, o] * x[i, o]
                for i in FACILITIES for o in PRIMARY_DEMANDS
            )
            + gamma * q
            + m.sum(p[i, o] for i in FACILITIES for o in PRIMARY_DEMANDS)
            <= f,
            ctname="robust_distance_epigraph",
        )
        for i in FACILITIES:
            for o in PRIMARY_DEMANDS:
                d_hat_i_o = lambda_distance * PRIMARY_DISTANCE[i, o]
                m.add_constraint(
                    q + p[i, o] >= d_hat_i_o * x[i, o],
                    ctname=f"robust_distance_dual[{i},{o}]",
                )
        distance_expr = f
    else:
        distance_expr = m.sum(
            PRIMARY_DISTANCE[i, o] * x[i, o]
            for i in FACILITIES for o in PRIMARY_DEMANDS
        )

    # Tie-breaking criterion.
    secondary_distance_expr = m.sum(
        SECONDARY_DISTANCE[i, j] * z[i, j] for i in FACILITIES for j in DEMANDS
    )

    return ModelParts(
        model=m,
        x=x,
        z=z,
        eta_shortage_expr=eta_shortage_expr,
        distance_expr=distance_expr,
        secondary_distance_expr=secondary_distance_expr,
        W_eff=W_eff,
        D_eff=D_eff,
        capacity=capacity,
    )


# ---------------------------------------------------------------------
# 3. Model solution and report
# ---------------------------------------------------------------------


def extract_solution(parts: ModelParts, mu_var=None, tol: float = 1e-6) -> Dict:
    m = parts.model
    if m.solution is None:
        raise RuntimeError(f"No usable solution. CPLEX status = {m.solve_details.status}")

    x_sol = {
        (i, o): parts.x[i, o].solution_value
        for i in FACILITIES for o in PRIMARY_DEMANDS
        if parts.x[i, o].solution_value > tol
    }
    z_sol = {
        (i, j): parts.z[i, j].solution_value
        for i in FACILITIES for j in DEMANDS
        if parts.z[i, j].solution_value > tol
    }

    total_capacity = sum(parts.capacity.values())

    primary_alloc_by_demand = {
        o: sum(parts.x[i, o].solution_value for i in FACILITIES)
        for o in PRIMARY_DEMANDS
    }
    primary_alloc = sum(primary_alloc_by_demand.values())
    total_primary_demand = sum(parts.W_eff.values())
    primary_satisfaction_by_demand = {
        o: primary_alloc_by_demand[o] / parts.W_eff[o]
        for o in PRIMARY_DEMANDS
    }
    primary_satisfaction = primary_alloc / total_primary_demand
    primary_min_satisfaction = min(primary_satisfaction_by_demand.values())

    secondary_alloc_by_demand = {
        j: sum(parts.z[i, j].solution_value for i in FACILITIES)
        for j in DEMANDS
    }
    secondary_alloc = sum(secondary_alloc_by_demand.values())
    total_secondary_demand = sum(parts.D_eff.values())
    secondary_satisfaction_by_demand = {
        j: secondary_alloc_by_demand[j] / parts.D_eff[j]
        for j in DEMANDS
    }
    secondary_satisfaction = secondary_alloc / total_secondary_demand
    secondary_min_satisfaction = min(secondary_satisfaction_by_demand.values())

    return {
        "mu": None if mu_var is None else float(mu_var.solution_value),
        "f1_distance_objective": float(parts.distance_expr.solution_value),
        "f2_shortage_objective": float(parts.eta_shortage_expr.solution_value),
        "secondary_distance": float(parts.secondary_distance_expr.solution_value),
        "primary_alloc": primary_alloc,
        "primary_satisfaction": primary_satisfaction,
        "primary_alloc_by_demand": primary_alloc_by_demand,
        "primary_satisfaction_by_demand": primary_satisfaction_by_demand,
        "primary_min_satisfaction": primary_min_satisfaction,
        "secondary_alloc": secondary_alloc,
        "secondary_satisfaction": secondary_satisfaction,
        "secondary_alloc_by_demand": secondary_alloc_by_demand,
        "secondary_satisfaction_by_demand": secondary_satisfaction_by_demand,
        "secondary_min_satisfaction": secondary_min_satisfaction,
        "total_alloc": primary_alloc + secondary_alloc,
        "total_capacity": total_capacity,
        "x": x_sol,
        "z": z_sol,
    }


# Single-objective model solution.
def solve_single_objective(
    objective: Literal["shortage", "distance"],
    **kwargs,
) -> Dict:
    """Solve one of the two single-objective LPs with lexicographic screening."""
    if objective == "shortage":
        shortage_kwargs = dict(kwargs)
        shortage_kwargs["robust_distance"] = False
        parts = build_base_model(
            name=f"single_{objective}",
            **shortage_kwargs,
        )
        parts.model.minimize(parts.eta_shortage_expr)
        parts.model.solve(log_output=False)
        if parts.model.solution is None:
            raise RuntimeError(f"No usable solution. CPLEX status = {parts.model.solve_details.status}")
        best_shortage = float(parts.eta_shortage_expr.solution_value)

        eval_parts = build_base_model(
            name=f"single_{objective}_evaluate_robust_distance",
            **kwargs,
        )
        eval_parts.model.add_constraint(
            eval_parts.eta_shortage_expr <= best_shortage + 1e-8,
            ctname="fix_shortage_optimum",
        )
        eval_parts.model.minimize(eval_parts.distance_expr)
        eval_parts.model.solve(log_output=False)
        sol = extract_solution(eval_parts)
        sol["model_type"] = f"single_{objective}"
        return sol

    if objective == "distance":
        parts = build_base_model(
            name=f"single_{objective}",
            **kwargs,
        )
        parts.model.minimize(parts.distance_expr)
        parts.model.solve(log_output=False)
        if parts.model.solution is None:
            raise RuntimeError(f"No usable solution. CPLEX status = {parts.model.solve_details.status}")
        best_distance = float(parts.distance_expr.solution_value)

        eval_parts = build_base_model(
            name=f"single_{objective}_evaluate_shortage",
            **kwargs,
        )
        eval_parts.model.add_constraint(
            eval_parts.distance_expr <= best_distance + 1e-8,
            ctname="fix_distance_optimum",
        )
        eval_parts.model.minimize(eval_parts.eta_shortage_expr)
        eval_parts.model.solve(log_output=False)
        sol = extract_solution(eval_parts)
        sol["model_type"] = f"single_{objective}"
        return sol

    raise ValueError("objective must be 'shortage' or 'distance'")


# Multi-objective model solution with fuzzy programming.
def solve_fuzzy(
    **kwargs,
) -> Dict:
   
    sol_short = solve_single_objective(
        "shortage", **kwargs,
    )
    sol_distance = solve_single_objective(
        "distance", **kwargs,
    )
    bounds = FuzzyBounds(
        distance_best=sol_distance["f1_distance_objective"],
        distance_worst=sol_short["f1_distance_objective"],
        shortage_best=sol_short["f2_shortage_objective"],
        shortage_worst=sol_distance["f2_shortage_objective"],
    )
    if bounds.shortage_scale <= 0 or bounds.distance_scale <= 0:
        raise ValueError(f"Invalid fuzzy bounds: {bounds}")

    parts = build_base_model(
        name="fuzzy_multiobjective",
        **kwargs,
    )
    m = parts.model
    mu = m.continuous_var(lb=0.0, ub=1.0, name="mu")

    m.add_constraint(
        parts.distance_expr + bounds.distance_scale * mu <= bounds.distance_worst,
        ctname="fuzzy_f1_distance",
    )
    m.add_constraint(
        parts.eta_shortage_expr + bounds.shortage_scale * mu <= bounds.shortage_worst,
        ctname="fuzzy_f2_shortage",
    )

    m.maximize(mu)
    m.solve(log_output=False)

    # Tie-breaking criterion.
    if m.solution is not None:
        mu_star = mu.solution_value
        m.add_constraint(mu >= mu_star - 1e-8, ctname="fix_mu_for_tiebreak")
        m.minimize(parts.secondary_distance_expr)
        m.solve(log_output=False)

    sol = extract_solution(parts, mu_var=mu)
    sol["model_type"] = "fuzzy_multiobjective"
    return sol


def _management_indicator_columns(sol: Dict) -> Dict:
    """Build management indicator columns for workbook outputs."""
    total_capacity = sol["total_capacity"]
    utilization = sol["total_alloc"] / total_capacity
    primary_nominal_distance = sum(
        PRIMARY_DISTANCE[i, o] * sol["x"].get((i, o), 0.0)
        for i in FACILITIES for o in PRIMARY_DEMANDS
    )
    primary_efficiency = primary_nominal_distance / sol["primary_alloc"]
    secondary_efficiency = sol["secondary_distance"] / sol["secondary_alloc"]
    return {
        "R_P": sol["primary_satisfaction"],
        "R_S": sol["secondary_satisfaction"],
        "R_P_min": sol["primary_min_satisfaction"],
        "R_S_min": sol["secondary_min_satisfaction"],
        "Gap_R": (
            abs(sol["primary_satisfaction"] - sol["secondary_satisfaction"])
        ),
        "Gap_min": abs(
            sol["primary_min_satisfaction"] - sol["secondary_min_satisfaction"]
        ),
        "Gap_P": abs(sol["primary_satisfaction"] - sol["primary_min_satisfaction"]),
        "Gap_S": abs(
            sol["secondary_satisfaction"] - sol["secondary_min_satisfaction"]
        ),
        "omega": utilization,
        "E_P": primary_efficiency,
        "E_S": secondary_efficiency,
    }


def _build_main_sheet_row(sol: Dict) -> Dict:
    """Build the strictly ordered main-sheet row requested for the workbook."""
    indicators = _management_indicator_columns(sol)
    row = {
        "model_type": sol.get("model_type"),
        "mu": sol.get("mu"),
        "f1_distance_objective": sol.get("f1_distance_objective"),
        "f2_shortage_objective": sol.get("f2_shortage_objective"),
        "R_P": indicators["R_P"],
        "R_S": indicators["R_S"],
        "R_P^min": indicators["R_P_min"],
        "R_S^min": indicators["R_S_min"],
        "Gap_R": indicators["Gap_R"],
        "Gap_min": indicators["Gap_min"],
        "Gap_P": indicators["Gap_P"],
        "Gap_S": indicators["Gap_S"],
        "omega": indicators["omega"],
        "E_P": indicators["E_P"],
        "E_S": indicators["E_S"],
        "primary_alloction": sol["primary_alloc"],
        "secondary_alloction": sol["secondary_alloc"],
        "total_alloction": sol["total_alloc"],
    }
    for o in PRIMARY_DEMANDS:
        label = chr(96 + o)
        row[f"primary_{label}_alloction"] = sol["primary_alloc_by_demand"][o]
        row[f"primary_{label}_satisfaction"] = sol["primary_satisfaction_by_demand"][o]
    for j in DEMANDS:
        label = chr(64 + j)
        row[f"secondary_{label}_alloction"] = sol["secondary_alloc_by_demand"][j]
        row[f"secondary_{label}_satisfaction"] = sol["secondary_satisfaction_by_demand"][j]
    return row


def _xlsx_cell_xml(row_idx: int, col_idx: int, value) -> str:
    """Build one XLSX cell using only standard-library XML."""
    column_name = ""
    while col_idx:
        col_idx, remainder = divmod(col_idx - 1, 26)
        column_name = chr(65 + remainder) + column_name
    cell_ref = f"{column_name}{row_idx}"
    if value is None or pd.isna(value):
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return f'<c r="{cell_ref}"/>'
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _worksheet_xml(df: pd.DataFrame) -> str:
    """Build minimal worksheet XML for a DataFrame."""
    rows = []
    header_cells = [
        _xlsx_cell_xml(1, col_idx, column)
        for col_idx, column in enumerate(df.columns, start=1)
    ]
    rows.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_idx, values in enumerate(df.itertuples(index=False, name=None), start=2):
        cells = [
            _xlsx_cell_xml(row_idx, col_idx, value)
            for col_idx, value in enumerate(values, start=1)
        ]
        rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx_standard_library(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """Write a simple multi-sheet XLSX file without openpyxl/xlsxwriter."""
    sheet_items = list(sheets.items())
    workbook_sheets = []
    workbook_rels = []
    content_overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    for idx, (sheet_name, _) in enumerate(sheet_items, start=1):
        safe_name = escape(sheet_name[:31])
        workbook_sheets.append(
            f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(content_overrides)}'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        '</workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rels)}'
        '</Relationships>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", root_rels)
        xlsx.writestr("xl/workbook.xml", workbook_xml)
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for idx, (_, df) in enumerate(sheet_items, start=1):
            xlsx.writestr(f"xl/worksheets/sheet{idx}.xml", _worksheet_xml(df))


def export_experiment_workbook(
    base_solutions: Iterable[Dict],
    secondary_mortality_comparison_solutions: Iterable[Dict],
    eta_df: pd.DataFrame,
    theta_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    """Export the main experiment and sensitivity analyses to one XLSX file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_solutions = list(base_solutions)
    secondary_mortality_comparison_solutions = list(secondary_mortality_comparison_solutions)

    main_rows = []
    for sol in base_solutions:
        main_rows.append(_build_main_sheet_row(sol))
    main_df = pd.DataFrame(main_rows)

    comparison_rows = []
    for model_type, sol in zip(
        ("with_secondary_mortality", "without_secondary_mortality"),
        secondary_mortality_comparison_solutions,
    ):
        display_sol = {**sol, "model_type": model_type}
        comparison_rows.append(_build_main_sheet_row(display_sol))
    comparison_df = pd.DataFrame(comparison_rows)

    write_xlsx_standard_library(
        WORKBOOK_PATH,
        {
            "main": main_df,
            "secondary_mortality_comparison": comparison_df,
            "eta": eta_df,
            "delta": delta_df,
            "theta": theta_df,
        },
    )
    print(f"Excel workbook saved to: {WORKBOOK_PATH}")


def run_sensitivity(parameter_name: str, values: Iterable[float]) -> pd.DataFrame:
    """Run one fuzzy-model sensitivity sweep."""
    rows = []
    for value in values:
        try:
            sol = solve_fuzzy(**{parameter_name: value})
        except (RuntimeError, ValueError) as exc:
            message = str(exc).lower()
            if any(
                text in message
                for text in ("infeasible", "no usable solution", "invalid fuzzy bounds")
            ):
                print(
                    f"Stopping {parameter_name} sensitivity at "
                    f"{parameter_name}={value:.6g}: {exc}"
                )
                break
            raise

        row = _build_main_sheet_row(sol)
        row.pop("model_type", None)
        rows.append({parameter_name: value, **row})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4. Main experiment run
# ---------------------------------------------------------------------

def main() -> None:
    configure_primary_clusters()

    export_primary_cluster_plot()

    sol_distance = solve_single_objective("distance")

    sol_shortage = solve_single_objective("shortage")

    sol_fuzzy = solve_fuzzy()
    print("Completed main experiment")

    sol_distance_no_secondary = solve_single_objective(
        "distance",
        include_secondary_shortage=False,
    )
    sol_distance_no_secondary["model_type"] = "single_distance_no_secondary_shortage"

    sol_shortage_no_secondary = solve_single_objective(
        "shortage",
        include_secondary_shortage=False,
    )
    sol_shortage_no_secondary["model_type"] = "single_shortage_no_secondary_shortage"

    sol_fuzzy_no_secondary = solve_fuzzy(
        include_secondary_shortage=False,
    )
    sol_fuzzy_no_secondary["model_type"] = "fuzzy_no_secondary_shortage"
    print("Completed without secondary mortality model")

    sens_eta = run_sensitivity(
        "eta",
        (round(1.0 + 0.1 * k, 1) for k in range(11)),
    )
    print("Completed sensitivity analysis for eta")

    sens_delta = run_sensitivity(
        "delta",
        (round(0.1 * k, 1) for k in range(1, 11)),
    )
    print("Completed sensitivity analysis for delta")

    sens_theta = run_sensitivity(
        "theta",
        (round(0.30 + 0.05 * k, 10) for k in count()),
    )
    print("Completed sensitivity analysis for theta")

    export_experiment_workbook(
        base_solutions=(sol_distance, sol_shortage, sol_fuzzy),
        secondary_mortality_comparison_solutions=(
            sol_fuzzy,
            sol_fuzzy_no_secondary,
        ),
        eta_df=sens_eta,
        delta_df=sens_delta,
        theta_df=sens_theta,
    )

if __name__ == "__main__":
    main()