"""
Simulate the SEIR model and export daily average compartment populations.
"""

from __future__ import annotations

import math
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "seir_output"
WORKBOOK_PATH = OUTPUT_DIR / "seir_daily_average_compartments.xlsx"
FIGURE_PATH = OUTPUT_DIR / "seir_simulation.png"


# SEIR model parameters.
BETA1 = 0.1575
BETA2 = 0.7874
R1 = 2.0
R2 = 1.0
SIGMA = 0.1429
GAMMA = 0.1538

# Simulation horizon and initial conditions.
DAYS = 150
DT = 0.05
N0 = 1.05
E0 = 1e-8
I0 = 1e-8
R0 = 0.0
S0 = N0 - E0 - I0 - R0
PEOPLE_PER_MILLION = 1_000_000


def seir_rhs(y: np.ndarray) -> np.ndarray:
    """SEIR model ODE system."""
    s, e, i, r = y
    n = s + e + i + r

    infectious_transmission = R1 * BETA1 * i * s / n
    latent_transmission = R2 * BETA2 * e * s / n

    ds = - infectious_transmission - latent_transmission
    de = infectious_transmission + latent_transmission - SIGMA * e
    di = SIGMA * e - GAMMA * i
    dr = GAMMA * i
    return np.array([ds, de, di, dr], dtype=float)


def simulate() -> tuple[np.ndarray, np.ndarray]:
    """Integrate the SEIR model with a fourth-order Runge-Kutta method."""
    t = np.arange(0.0, DAYS + DT, DT)
    y = np.zeros((len(t), 4), dtype=float)
    y[0] = np.array([S0, E0, I0, R0], dtype=float)

    for k in range(len(t) - 1):
        current = y[k]
        k1 = seir_rhs(current)
        k2 = seir_rhs(current + 0.5 * DT * k1)
        k3 = seir_rhs(current + 0.5 * DT * k2)
        k4 = seir_rhs(current + DT * k3)
        y[k + 1] = current + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


def daily_average_compartments(t: np.ndarray, y: np.ndarray, days: range) -> pd.DataFrame:
    """Return rounded daily average S/E/I/R populations in people."""
    rows = []
    for day in days:
        mask = (t >= day) & (t < day + 1)
        daily_mean = np.mean(y[mask], axis=0) * PEOPLE_PER_MILLION
        rows.append({
            "day": day,
            "S_average_people": int(round(float(daily_mean[0]))),
            "E_average_people": int(round(float(daily_mean[1]))),
            "I_average_people": int(round(float(daily_mean[2]))),
            "R_average_people": int(round(float(daily_mean[3]))),
        })
    return pd.DataFrame(rows)


def peak_population(t: np.ndarray, y: np.ndarray, column: int) -> tuple[float, float]:
    """Return the peak day and peak population for one SEIR compartment."""
    values = y[:, column]
    peak_idx = int(np.argmax(values))
    return float(t[peak_idx]), float(values[peak_idx])


def plot_results(t: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """Plot S, E, I, R curves and save the figure."""
    s, e, i, r = y.T

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(8.8, 5.3), dpi=600)
    ax.plot(t, s, color="#72c9f4", lw=2.0, label="Susceptible (S)")
    ax.plot(t, e, color="#3f86d0", lw=2.0, label="Exposed (E)")
    ax.plot(t, i, color="#ffc85c", lw=2.0, label="Infectious (I)")
    ax.plot(t, r, color="#7a7a7a", lw=2.0, label="Recovered (R)")

    ax.set_xlim(0, DAYS)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Day", fontsize=18)
    ax.set_ylabel("Population (million people)", fontsize=18)
    ax.set_xticks(np.arange(0, DAYS + 1, 20))
    ax.set_yticks(np.arange(0, 1.21, 0.2))
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle=(0, (4, 4)), color="#d9d9d9", linewidth=1.0)
    ax.tick_params(which="both", direction="in", top=True, right=True, width=1.2)

    legend = ax.legend(
        loc="center right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=16,
        borderpad=0.25,
        handlelength=1.8,
        handletextpad=0.3,
    )
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _xlsx_cell_xml(row_idx: int, col_idx: int, value) -> str:
    column_name = ""
    while col_idx:
        col_idx, remainder = divmod(col_idx - 1, 26)
        column_name = chr(65 + remainder) + column_name
    cell_ref = f"{column_name}{row_idx}"
    if value is None or pd.isna(value):
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return f'<c r="{cell_ref}"/>'
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _worksheet_xml(df: pd.DataFrame) -> str:
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


def write_xlsx_standard_library(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
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


def main() -> None:
    t, y = simulate()
    exposed_peak_day, exposed_peak_value = peak_population(t, y, column=1)
    infectious_peak_day, infectious_peak_value = peak_population(t, y, column=2)
    print(
        "Exposed (E) peak: "
        f"around {round(exposed_peak_day):.0f}, "
        f"{exposed_peak_value * PEOPLE_PER_MILLION:.0f} people"
    )
    print(
        "Infectious (I) peak: "
        f"around {round(infectious_peak_day):.0f}, "
        f"{infectious_peak_value * PEOPLE_PER_MILLION:.0f} people"
    )
    daily_averages = daily_average_compartments(t, y, range(0, DAYS + 1))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_xlsx_standard_library(WORKBOOK_PATH, {"daily_average": daily_averages})
    plot_results(t, y, FIGURE_PATH)
    print(f"Excel workbook saved to: {WORKBOOK_PATH}")
    print(f"SEIR simulation figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()