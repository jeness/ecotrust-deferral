#!/usr/bin/env python3
"""Generate the three paper result figures from canonical experiment artifacts.

The script uses only the Python standard library.  It writes standalone
PGFPlots/TikZ sources to ``results/``; run latexmk on those sources to obtain
vector PDFs for inclusion in the IEEE paper.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[1]
RESULTS = ARTIFACTS / "results"


COMMON = r"""
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{arrows.meta}
\pgfplotsset{compat=1.18}
\definecolor{cbBlue}{HTML}{0072B2}
\definecolor{cbOrange}{HTML}{D55E00}
\definecolor{cbGreen}{HTML}{009E73}
\definecolor{cbPurple}{HTML}{CC79A7}
\definecolor{cbSky}{HTML}{56B4E9}
\definecolor{cbGold}{HTML}{E69F00}
\pgfplotsset{
  paperaxis/.style={
    axis line style={black!65,line width=0.45pt},
    tick style={black!65,line width=0.45pt},
    tick label style={font=\footnotesize},
    label style={font=\small},
    title style={font=\small\bfseries,yshift=-1pt},
    grid=major,
    grid style={black!10,line width=0.35pt},
    line width=1.05pt,
    mark size=1.8pt,
  }
}
"""


def read_json(name: str):
    with (RESULTS / name).open() as handle:
        return json.load(handle)


def write_tex(name: str, body: str) -> None:
    path = RESULTS / name
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path.relative_to(ARTIFACTS)}")


def table_rows(rows, columns):
    header = " ".join(columns) + r"\\"
    data = [" ".join(str(row[col]) for col in columns) + r"\\" for row in rows]
    return "\n".join([header] + data)


def coordinates(rows, x_key: str, y_key: str, x_scale: float = 1.0) -> str:
    return " ".join(
        f"({float(row[x_key]) * x_scale:.6g},{float(row[y_key]):.6g})"
        for row in rows
    )


def make_main_comparison_figure() -> None:
    """Plot the two operational trade-offs hidden by the main result table."""
    decision = read_json("decision_learning_results.json")
    rows = decision["E1_main_comparison"]["rows"]
    by_code = {row["method"].split()[0]: row for row in rows}

    families = [
        ("Automatic", ["A0", "A1"], "black!60", "square*"),
        ("Random / heuristic", ["B0", "B1", "B2"], "cbGold", "diamond*"),
        ("Confidence rule", ["B3", "B4"], "cbOrange", "triangle*"),
        ("L2D", ["B5", "B6", "B7"], "cbGreen", "pentagon*"),
        ("EcoTrust-DL", ["C1", "C2", "C3"], "cbBlue", "*"),
    ]

    def point_coordinates(codes, y_key, y_scale=1.0):
        return " ".join(
            f"({100 * by_code[code]['escalation_rate']:.3f},{y_scale * by_code[code][y_key]:.4f})"
            for code in codes
        )

    cost_plots = []
    coverage_plots = []
    for family, codes, color, mark in families:
        cost_plots.append(
            f"\\addplot[only marks,{color},mark={mark},mark size=2.4pt] coordinates "
            f"{{{point_coordinates(codes, 'deployment_cost')}}};\n"
            f"\\addlegendentry{{{family}}}"
        )
        review_codes = [code for code in codes if code not in {"A0", "A1"}]
        if review_codes:
            coverage_plots.append(
                f"\\addplot[only marks,{color},mark={mark},mark size=2.4pt] coordinates "
                f"{{{point_coordinates(review_codes, 'error_coverage', y_scale=100)}}};"
            )

    cost_offsets = {
        "A0": (5, -2, "west"),
        "A1": (5, 0, "west"),
        "B0": (0, 6, "south"),
        "B1": (-5, -1, "east"),
        "B2": (0, 5, "south"),
        "B3": (-5, 7, "south east"),
        "B4": (5, 0, "west"),
        "B5": (5, 0, "west"),
        "B6": (0, 7, "south"),
        "B7": (-5, 5, "south east"),
        "C1": (-5, 4, "south east"),
    }
    coverage_offsets = {
        "B0": (0, 7, "south"),
        "B1": (-5, -1, "east"),
        "B2": (5, 0, "west"),
        "B3": (5, 0, "west"),
        "B4": (4, 0, "west"),
        "B5": (5, 6, "south west"),
        "B6": (-6, 0, "east"),
        "B7": (5, -6, "north west"),
        "C1": (6, -4, "west"),
    }

    label_text = {"C1": "C1--C3"}

    def labels(y_key, offsets, codes, y_scale=1.0):
        output = []
        for code in codes:
            row = by_code[code]
            dx, dy, anchor = offsets[code]
            x = 100 * row["escalation_rate"]
            y = y_scale * row[y_key]
            text = label_text.get(code, code)
            output.append(
                f"\\node[font=\\footnotesize,anchor={anchor},fill=white,"
                f"fill opacity=.84,text opacity=1,inner sep=.6pt] at "
                f"([xshift={dx}pt,yshift={dy}pt]axis cs:{x:.3f},{y:.4f}) {{{text}}};"
            )
        return "\n".join(output)

    body = r"""\documentclass[tikz,border=2pt]{standalone}
@@COMMON@@
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=1 by 2,vertical sep=1.65cm},
  paperaxis,width=8.55cm,height=4.05cm,
]
\nextgroupplot[
  title={(a) Cost--review trade-off},
  xmin=-3,xmax=82,ymin=0,ymax=3.05,
  xlabel={Escalation rate (\%)},ylabel={Deployment cost/day},
  xtick={0,20,40,60,80},ytick={0,0.5,1,2,3},
  legend to name=methodlegend,
  legend columns=2,
  legend style={font=\footnotesize,draw=none,column sep=5pt},
]
@@COST_PLOTS@@
@@COST_LABELS@@

\nextgroupplot[
  title={(b) Error capture per review budget},
  xmin=18,xmax=82,ymin=20,ymax=101,
  xlabel={Escalation rate (\%)},ylabel={Error coverage (\%)},
  xtick={20,40,60,80},ytick={20,40,60,80,100},
]
\addplot[black!50,densely dotted,domain=18:82,samples=2] {x};
@@COVERAGE_PLOTS@@
@@COVERAGE_LABELS@@
\end{groupplot}
\node[anchor=north] at ([yshift=-1.05cm]group c1r2.south) {\pgfplotslegendfromname{methodlegend}};
\end{tikzpicture}
\end{document}
"""
    replacements = {
        "@@COMMON@@": COMMON,
        "@@COST_PLOTS@@": "\n".join(cost_plots),
        "@@COST_LABELS@@": labels(
            "deployment_cost", cost_offsets,
            ["A0", "A1", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "C1"]
        ),
        "@@COVERAGE_PLOTS@@": "\n".join(coverage_plots),
        "@@COVERAGE_LABELS@@": labels(
            "error_coverage", coverage_offsets,
            ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "C1"], y_scale=100
        ),
    }
    for token, value in replacements.items():
        body = body.replace(token, value)
    write_tex("main_method_comparison.tex", body)


def make_mechanism_figure() -> None:
    decision = read_json("decision_learning_results.json")
    loo = decision["E2_signal_contribution"]["leave_one_out"]

    with (RESULTS / "backend_bank_plot_data.csv").open(newline="") as handle:
        backend = list(csv.DictReader(handle))
    real = [r for r in backend if r["is_strawman"] == "0"]
    control = [r for r in backend if r["is_strawman"] == "1"]

    real_table = table_rows(
        [
            {
                "ece": r["ece"],
                "gain": r["cost_advantage_dl"],
                "err": r["cost_advantage_dl_sd"],
            }
            for r in real
        ],
        ["ece", "gain", "err"],
    )
    control_table = table_rows(
        [
            {
                "ece": r["ece"],
                "gain": r["cost_advantage_dl"],
                "err": r["cost_advantage_dl_sd"],
            }
            for r in control
        ],
        ["ece", "gain", "err"],
    )

    label = {
        "conflict": "Agent conflict",
        "entropy": "Entropy",
        "confidence": "Confidence",
        "conformal": "Conformal",
        "novelty": "Novelty",
        "variance": "Ens. variance",
    }
    ordered = sorted(loo, key=lambda row: row["delta_cost_vs_full"], reverse=True)
    ypos = {row["dropped"]: len(ordered) - i for i, row in enumerate(ordered)}
    # PGFPlots maps the first yticklabel to y=1 (the bottom row), whereas the
    # most consequential omission is deliberately placed at y=6 (the top).
    yticklabels = ",".join(label[row["dropped"]] for row in reversed(ordered))

    cost_stems = []
    cost_marks = []
    cost_nodes = []
    auc_stems = []
    auc_pos = []
    auc_neg = []
    auc_nodes = []
    for row in ordered:
        y = ypos[row["dropped"]]
        dc = row["delta_cost_vs_full"]
        da = row["delta_auc_vs_full"]
        cost_stems.append(f"\\addplot[black!35,line width=.55pt] coordinates {{(0,{y}) ({dc:.4f},{y})}};")
        cost_marks.append(f"({dc:.4f},{y})")
        cost_nodes.append(
            f"\\node[font=\\footnotesize,anchor=west] at (axis cs:{dc:.4f},{y}) {{{dc:+.3f}}};"
        )
        auc_stems.append(f"\\addplot[black!35,line width=.55pt] coordinates {{(0,{y}) ({da:.4f},{y})}};")
        (auc_pos if da >= 0 else auc_neg).append(f"({da:.4f},{y})")
        # Place labels to the right of their markers so negative values remain
        # inside the axis instead of being clipped at the left boundary.
        anchor = "west"
        auc_nodes.append(
            f"\\node[font=\\footnotesize,anchor={anchor}] at (axis cs:{da:.4f},{y}) {{{da:+.3f}}};"
        )

    body = r"""\documentclass[tikz,border=2pt]{standalone}
@@COMMON@@
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  paperaxis,
  name=eceplot,
  width=7.5cm,height=5.25cm,
  xmin=0,xmax=0.285,ymin=-0.35,ymax=4.35,
  xlabel={Backend ECE},
  ylabel={Cost gain (Chow $-$ learned risk)},
  title={(a) Calibration predicts learned-risk value},
  legend style={font=\footnotesize,draw=none,fill=none,at={(0.98,0.98)},anchor=north east,row sep=-1pt},
]
\addplot[black!55,dashed,domain=0:0.285,samples=2] {@@FIT_SLOPE@@*x @@FIT_INTERCEPT@@};
\addlegendentry{fit to real backends}
\addplot[only marks,mark=*,cbBlue,fill=cbBlue,error bars/.cd,y dir=both,y explicit]
table[row sep=\\,x=ece,y=gain,y error=err]{
@@REAL_TABLE@@
};
\addlegendentry{real backend}
\addplot[only marks,mark=triangle*,cbOrange,fill=cbOrange,error bars/.cd,y dir=both,y explicit]
table[row sep=\\,x=ece,y=gain,y error=err]{
@@CONTROL_TABLE@@
};
\addlegendentry{over-confident control}
\addplot[black!55,densely dotted,domain=0:0.285,samples=2] {0};
\end{axis}

\begin{axis}[
  paperaxis,
  % Reserve a real label gutter: the longest row name (Agent conflict) must
  % not intrude into panel (a), especially after the figure is scaled in IEEE.
  at={(8.70cm,0)},anchor=south west,
  width=5.15cm,height=5.25cm,
  xmin=-0.014,xmax=0.009,ymin=0.5,ymax=6.5,
  xlabel={$\Delta$Cost after removal},
  title={(b) Decision contribution},
  ytick={1,2,3,4,5,6},
  yticklabels={@@YTICKLABELS@@},
  xtick={-0.01,-0.005,0,0.005},
  scaled x ticks=false,
  /pgf/number format/fixed,
  /pgf/number format/precision=3,
]
\addplot[black!55,densely dotted] coordinates {(0,0.5) (0,6.5)};
@@COST_STEMS@@
\addplot[only marks,mark=*,cbBlue,fill=cbBlue] coordinates {@@COST_MARKS@@};
@@COST_NODES@@
\end{axis}

\begin{axis}[
  paperaxis,
  at={(13.70cm,0)},anchor=south west,
  width=5.15cm,height=5.25cm,
  xmin=-0.0205,xmax=0.0145,ymin=0.5,ymax=6.5,
  xlabel={$\Delta$Risk AUC after removal},
  title={(c) Discrimination contribution},
  ytick={1,2,3,4,5,6},yticklabels={,,,,,},
  xtick={-0.01,0,0.01},
  scaled x ticks=false,
  /pgf/number format/fixed,
  /pgf/number format/precision=3,
]
\addplot[black!55,densely dotted] coordinates {(0,0.5) (0,6.5)};
@@AUC_STEMS@@
\addplot[only marks,mark=*,cbBlue,fill=cbBlue] coordinates {@@AUC_POS@@};
\addplot[only marks,mark=*,cbOrange,fill=cbOrange] coordinates {@@AUC_NEG@@};
@@AUC_NODES@@
\end{axis}
\end{tikzpicture}
\end{document}
"""
    # Fit line computed from the plotted real backends so a regeneration can
    # never restore a stale hard-coded slope/intercept. Ordinary least squares
    # in pure Python (no numpy dependency in this generator).
    _ex = [float(r["ece"]) for r in real]
    _ay = [float(r["cost_advantage_dl"]) for r in real]
    _n = len(_ex)
    _mx = sum(_ex) / _n
    _my = sum(_ay) / _n
    _sxx = sum((x - _mx) ** 2 for x in _ex)
    _sxy = sum((x - _mx) * (y - _my) for x, y in zip(_ex, _ay))
    _slope = _sxy / _sxx
    _intercept = _my - _slope * _mx

    replacements = {
        "@@COMMON@@": COMMON,
        "@@FIT_SLOPE@@": f"{_slope:.4f}",
        "@@FIT_INTERCEPT@@": (f"+ {_intercept:.4f}" if _intercept >= 0
                              else f"- {abs(_intercept):.4f}"),
        "@@REAL_TABLE@@": real_table,
        "@@CONTROL_TABLE@@": control_table,
        "@@YTICKLABELS@@": yticklabels,
        "@@COST_STEMS@@": "\n".join(cost_stems),
        "@@COST_MARKS@@": " ".join(cost_marks),
        "@@COST_NODES@@": "\n".join(cost_nodes),
        "@@AUC_STEMS@@": "\n".join(auc_stems),
        "@@AUC_POS@@": " ".join(auc_pos),
        "@@AUC_NEG@@": " ".join(auc_neg),
        "@@AUC_NODES@@": "\n".join(auc_nodes),
    }
    for token, value in replacements.items():
        body = body.replace(token, value)
    write_tex("mechanism_evidence.tex", body)


def make_operating_figure() -> None:
    decision = read_json("decision_learning_results.json")
    reviewer_results = read_json("reviewer_sensitivity_results.json")
    reviewer = reviewer_results["rows"]
    sweep = decision["E3_cost_sensitivity"]
    fixed = decision["E8_fixed_review_rate"]["curves"]
    automatic = next(
        row for row in decision["E1_main_comparison"]["rows"]
        if row["method"].startswith("A0 ")
    )
    automatic_cost = automatic["deployment_cost"]
    automatic_accuracy = reviewer_results["raw_accuracy"]
    breakeven_pct = 100 * reviewer_results["predicted_breakeven"]

    curve_specs = [
        ("EcoTrust-DL (learned risk)", "cbBlue", "*", "solid", "EcoTrust"),
        ("Chow / self-confidence", "cbOrange", "square*", "dashed", "Chow"),
        ("L2D rejector (raw feats)", "cbGreen", "triangle*", "dashdotted", "Raw L2D"),
        ("Selective classification", "cbPurple", "diamond*", "dotted", "Selective"),
        ("Hand-weighted score u", "black!55", "pentagon*", "densely dashed", "Hand score"),
    ]
    budget_plots = []
    for key, color, mark, style, legend in curve_specs:
        coords = coordinates(fixed[key], "budget", "deployment_cost", x_scale=100)
        budget_plots.append(
            f"\\addplot[{color},{style},mark={mark}] coordinates {{{coords}}};\n"
            f"\\addlegendentry{{{legend}}}"
        )

    acc_tables = {}
    for prefix in ("unaware", "aware"):
        acc_tables[prefix] = table_rows(
            [
                {
                    "a": 100 * row["reviewer_acc"],
                    "acc": row[f"{prefix}_accuracy"],
                    "err": row[f"{prefix}_accuracy_sd"],
                }
                for row in reviewer
            ],
            ["a", "acc", "err"],
        )

    body = r"""\documentclass[tikz,border=2pt]{standalone}
@@COMMON@@
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 2,horizontal sep=1.25cm,vertical sep=2.25cm},
  paperaxis,width=8.55cm,height=4.7cm,
]
\nextgroupplot[
  title={(a) Miss-cost sensitivity},
  xmode=log,ymode=log,log ticks with fixed point,
  xmin=0.85,xmax=590,ymin=0.12,ymax=16,
  xlabel={False-negative cost $c_{FN}$ (log scale)},
  ylabel={Deployment cost/day (log scale)},
  xtick={1,5,10,50,100,500},
  legend style={font=\footnotesize,draw=none,fill=none,at={(0.02,0.98)},anchor=north west,row sep=-1pt},
]
\path[fill=black!5] (axis cs:5,0.12) rectangle (axis cs:10,16);
\addplot[cbBlue,solid,mark=*] coordinates {@@SWEEP_ECO@@};
\addlegendentry{EcoTrust}
\addplot[black!60,dashed,mark=square*] coordinates {@@SWEEP_AUTO@@};
\addlegendentry{Automatic}
\addplot[cbOrange,dashdotted,mark=triangle*] coordinates {@@SWEEP_CHOW@@};
\addlegendentry{Chow}

\nextgroupplot[
  title={(b) Fixed review rate (exactly $k$)},
  xmin=3,xmax=52,ymin=0.25,ymax=2.82,
  xlabel={Review rate (\%)},ylabel={Deployment cost/day},
  xtick={5,10,20,30,40,50},
  legend style={font=\footnotesize,draw=none,fill=white,fill opacity=.86,text opacity=1,at={(0.98,0.98)},anchor=north east,row sep=-1pt},
]
@@BUDGET_PLOTS@@

\nextgroupplot[
  title={(c) Cost under reviewer noise},
  xmin=68,xmax=102,ymin=0.48,ymax=3.3,
  xlabel={Reviewer accuracy $a$ (\%)},ylabel={Deployment cost/day},
  xtick={70,80,90,100},
  legend style={font=\footnotesize,draw=none,fill=none,at={(0.98,0.98)},anchor=north east,row sep=-1pt},
]
\addplot[cbOrange,dashed,mark=square*] coordinates {@@COST_UNAWARE@@};
\addlegendentry{Oracle-unaware}
\addplot[cbBlue,solid,mark=*] coordinates {@@COST_AWARE@@};
\addlegendentry{Reviewer-aware}
\addplot[black!55,densely dotted] coordinates {(70,@@AUTO_COST@@) (100,@@AUTO_COST@@)};
\addlegendentry{Automatic}

\nextgroupplot[
  title={(d) Accuracy and predicted breakeven},
  xmin=68,xmax=102,ymin=0.80,ymax=0.98,
  xlabel={Reviewer accuracy $a$ (\%)},ylabel={System accuracy},
  xtick={70,80,90,100},ytick={0.80,0.85,0.90,0.95},
]
\addplot[cbOrange,dashed,mark=square*,error bars/.cd,y dir=both,y explicit]
table[row sep=\\,x=a,y=acc,y error=err]{
@@ACC_UNAWARE@@
};
\addplot[cbBlue,solid,mark=*,error bars/.cd,y dir=both,y explicit]
table[row sep=\\,x=a,y=acc,y error=err]{
@@ACC_AWARE@@
};
\addplot[black!55,densely dotted] coordinates {(70,@@AUTO_ACCURACY@@) (100,@@AUTO_ACCURACY@@)};
\draw[black!60,dashdotted] (axis cs:@@BREAKEVEN@@,0.80) -- (axis cs:@@BREAKEVEN@@,0.98);
\node[font=\footnotesize,anchor=north east,fill=white,inner sep=1pt] at (axis cs:@@BREAKEVEN@@,0.978) {$a^*=@@BREAKEVEN@@\%$};
\end{groupplot}
\end{tikzpicture}
\end{document}
"""
    replacements = {
        "@@COMMON@@": COMMON,
        "@@SWEEP_ECO@@": coordinates(sweep, "c_fn", "deployment_cost"),
        "@@SWEEP_AUTO@@": coordinates(sweep, "c_fn", "deployment_cost_auto"),
        "@@SWEEP_CHOW@@": coordinates(sweep, "c_fn", "chow_deployment_cost"),
        "@@BUDGET_PLOTS@@": "\n".join(budget_plots),
        "@@COST_UNAWARE@@": coordinates(reviewer, "reviewer_acc", "unaware_cost", x_scale=100),
        "@@COST_AWARE@@": coordinates(reviewer, "reviewer_acc", "aware_cost", x_scale=100),
        "@@ACC_UNAWARE@@": acc_tables["unaware"],
        "@@ACC_AWARE@@": acc_tables["aware"],
        "@@AUTO_COST@@": f"{automatic_cost:.3f}",
        "@@AUTO_ACCURACY@@": f"{automatic_accuracy:.4f}",
        "@@BREAKEVEN@@": f"{breakeven_pct:.2f}",
    }
    for token, value in replacements.items():
        body = body.replace(token, value)
    write_tex("operating_regimes.tex", body)


def blend(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def cost_color(value: float) -> str:
    blue = (44, 123, 182)
    white = (255, 255, 245)
    red = (215, 48, 39)
    if value <= 1.0:
        t = max(0.0, min(1.0, (value - 0.90) / 0.10))
        rgb = blend(blue, white, t)
    else:
        t = max(0.0, min(1.0, (value - 1.0) / 1.40))
        rgb = blend(white, red, t)
    return "".join(f"{channel:02X}" for channel in rgb)


def make_transfer_figure() -> None:
    sites = read_json("multibasin_results.json")["sites"]
    short_names = {
        "14150000": "MF Willamette",
        "14316500": "N Umpqua",
        "13317000": "Salmon",
        "12500450": "Yakima",
        "14162500": "McKenzie",
        "14339000": "Rogue",
        "12181000": "Skagit",
        "12210700": "Nooksack",
        "13340000": "Clearwater",
        "13269000": "Snake",
    }
    methods = [
        ("Auto", lambda s: s["cost_auto"]),
        ("Chow", lambda s: s["cost_chow"]),
        ("Ungated", lambda s: s["cost_ungated"]),
        ("L2D-MS", lambda s: s["l2d_ms"]["deployment_cost"]),
        ("L2D-OvA", lambda s: s["l2d_ova"]["deployment_cost"]),
        ("Gate", lambda s: s["cost_gated"]),
    ]

    definitions = []
    rows = []
    cell_index = 0
    for i, site in enumerate(sites):
        y = -0.58 * i
        error = site["true_error_rate"]
        name = short_names[site["site_no"]]
        rows.append(
            f"\\node[font=\\footnotesize,anchor=east] at (3.25,{y:.3f}) {{{name}}};"
        )
        rows.append(
            f"\\path[fill=black!7] (3.45,{y - 0.08:.3f}) rectangle (4.90,{y + 0.08:.3f});"
        )
        rows.append(
            f"\\path[fill=cbGreen!75] (3.45,{y - 0.08:.3f}) rectangle ({3.45 + 1.45 * error:.3f},{y + 0.08:.3f});"
        )
        rows.append(
            f"\\node[font=\\footnotesize,anchor=east] at (4.84,{y:.3f}) {{{100 * error:.1f}\\%}};"
        )
        values = [getter(site) for _, getter in methods]
        minimum = min(values)
        for j, value in enumerate(values):
            x = 6.05 + 1.68 * j
            color_name = f"cell{cell_index}"
            definitions.append(f"\\definecolor{{{color_name}}}{{HTML}}{{{cost_color(value)}}}")
            is_min = abs(value - minimum) < 5e-4
            outline = "draw=black,line width=0.9pt" if is_min else "draw=black!22,line width=0.35pt"
            font = r"\footnotesize\bfseries" if is_min else r"\footnotesize"
            rows.append(
                f"\\draw[fill={color_name},{outline}] ({x - 0.80:.3f},{y - 0.25:.3f}) rectangle ({x + 0.80:.3f},{y + 0.25:.3f});"
            )
            rows.append(
                f"\\node[font={font}] at ({x:.3f},{y:.3f}) {{{value:.3f}}};"
            )
            cell_index += 1

    headers = []
    for j, (name, _) in enumerate(methods):
        x = 6.05 + 1.68 * j
        headers.append(f"\\node[font=\\small\\bfseries] at ({x:.3f},0.62) {{{name}}};")

    body = r"""\documentclass[tikz,border=2pt]{standalone}
\usepackage{xcolor}
\definecolor{cbGreen}{HTML}{009E73}
@@DEFINITIONS@@
\begin{document}
\begin{tikzpicture}
\node[font=\small\bfseries,anchor=east] at (3.25,0.62) {USGS station};
\node[font=\small\bfseries] at (4.18,0.62) {Error};
\node[font=\small\bfseries] at (10.25,1.03) {Deployment cost per day};
@@HEADERS@@
@@ROWS@@
\node[font=\footnotesize,anchor=west] at (5.25,-5.82) {blue: $<1$};
\node[font=\footnotesize,anchor=west] at (7.35,-5.82) {white: one review/day};
\node[font=\footnotesize,anchor=west] at (11.25,-5.82) {red: $>1$};
\draw[black,line width=.9pt] (14.0,-5.99) rectangle (14.45,-5.65);
\node[font=\footnotesize,anchor=west] at (14.55,-5.82) {row minimum};
\end{tikzpicture}
\end{document}
"""
    replacements = {
        "@@DEFINITIONS@@": "\n".join(definitions),
        "@@HEADERS@@": "\n".join(headers),
        "@@ROWS@@": "\n".join(rows),
    }
    for token, value in replacements.items():
        body = body.replace(token, value)
    write_tex("transfer_heatmap.tex", body)


def main() -> None:
    make_main_comparison_figure()
    make_mechanism_figure()
    make_operating_figure()
    make_transfer_figure()


if __name__ == "__main__":
    main()
