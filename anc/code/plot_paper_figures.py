"""Rebuild four manuscript figures using only portable scalar result records.

From the companion directory, run:
    python code/plot_paper_figures.py --output-dir ../images

The plots retain the manuscript's units and geometry. No experiment runner,
raw benchmark record, or completed-table certificate is imported or opened.
Each plot is accompanied by its plotted values and rendered panel geometry.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re


def pin_one_cpu(requested: int | None) -> int:
    """Use one available logical CPU, leaving logical CPUs 0--3 unused."""
    import psutil
    process = psutil.Process()
    allowed = process.cpu_affinity()
    candidates = [cpu for cpu in allowed if cpu >= 4]
    if not candidates:
        raise RuntimeError("No available logical CPU outside reserved CPUs 0--3")
    selected = requested if requested is not None else (10 if 10 in candidates else candidates[-1])
    if selected not in candidates:
        raise ValueError(f"CPU {selected} is unavailable or reserved; choose from {candidates}")
    process.cpu_affinity([selected])
    return selected


def read_json(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


def save_figure(fig, stem: str, output: Path, evidence: dict) -> dict:
    import matplotlib.pyplot as plt
    from audit_panel_alignment import require_matplotlib_panel_alignment
    output.mkdir(parents=True, exist_ok=True)
    qa = output / "figure_qa"
    qa.mkdir(exist_ok=True)
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=qa / f"{stem}.alignment.json",
        overlay_svg=qa / f"{stem}.alignment.svg", strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
    )
    prefix = output / stem
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04)
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=.04)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.04)
    evidence.update(raw_records_opened=False, experiment_solvers_run=False,
                    alignment_status=alignment["verdict"])
    (qa / f"{stem}.values.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    plt.close(fig)
    return {"figure": stem, "alignment": alignment["verdict"], "pdf": prefix.with_suffix(".pdf").name}


def plot_structure(root: Path, output: Path) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    controls = read_json(root, "energy_mechanism/curvature_controls.json")
    structure = read_json(root, "structure/structure_results.json")
    assert controls["status"] == "PASS" and controls["fixed_p"] == 65
    scales = controls["scales"]
    x = [row["scale"] for row in scales]
    assert x == [1, 2, 4, 16, 64, 256]
    # Bounds have normalized-objective units; this panel displays 10^-6 units.
    values = {key: [float(Fraction(row[key])) * 1e6 for row in scales]
              for key in ("actual_omission", "energy", "enhanced_scalar", "metric_moment")}
    assert all(np.isfinite(v) and v > 0 for series in values.values() for v in series)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained")
    for key, label, color, marker in (
        ("actual_omission", "Actual omission", "#35434e", "x"),
        ("energy", "Energy", "#168c82", "o"),
        ("enhanced_scalar", "Enhanced scalar", "#e39722", "s"),
        ("metric_moment", "Preconditioned moment", "#2465a6", "^"),
    ):
        ax[0].plot(x, values[key], label=label, color=color, marker=marker, markersize=3, lw=1.1)
    ax[0].set(xscale="log", yscale="log", xlabel="Nuisance scale s",
              ylabel=r"Additive error bound ($10^{-6}$)")
    ax[0].set_title("A  Curvature-aware controls, fixed p=65", pad=28)
    ax[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(-.02, 1.01),
                 ncol=2, fontsize=6.3, handlelength=1.5, columnspacing=1.0)
    ax[0].set_xticks(x, labels=list(map(str, x)))
    rows = sorted([row for row in structure["results"]
                   if row["alpha"] == "1/32" and row["method"] == "residual_energy"],
                  key=lambda row: row["p"])
    ps = [row["p"] for row in rows]
    assert ps == [17, 65, 257]
    setup = [sum(row[key] for key in ("construction_seconds", "anchor_setup_seconds", "method_setup_seconds"))
             for row in rows]
    search = [row["search_wall_seconds"] for row in rows]
    total = [a + b for a, b in zip(setup, search)]
    assert np.allclose(total, [row["method_end_to_end_seconds"] for row in rows], rtol=1e-10, atol=1e-10)
    for series, label, color, marker in (
        (setup, "Setup", "#2465a6", "o"), (search, "Search", "#168c82", "^"),
        (total, "Total", "#35434e", "s"),
    ):
        assert all(v > 0 and np.isfinite(v) for v in series)
        ax[1].plot(ps, series, label=label, color=color, marker=marker, markersize=3, lw=1.1)
    ax[1].set(yscale="log", xlabel="Original parameters p", ylabel="Wall time (s)")
    ax[1].set_title("B  Complete cost as dimension grows", pad=28)
    ax[1].set_xticks(ps)
    ax[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(-.02, 1.01),
                 ncol=3, fontsize=6.3, handlelength=1.4, columnspacing=1.0)
    return save_figure(fig, "structure_main_v82", output,
                       dict(sources=["energy_mechanism/curvature_controls.json", "structure/structure_results.json"],
                            scales=x, bounds_1e6=values, p=ps, setup_seconds=setup, search_seconds=search))


def decision_counts(root: Path) -> dict:
    conditions = read_json(root, "energy_decisions/conditions.json")
    lookup = {(row["pool"], row["seed"]): row for row in conditions}
    assert len(lookup) == 15
    counts = {}
    for key, name in (("R", "frozen_decisions.json"), ("M", "decisions_baseline_extended.json")):
        rows = read_json(root, f"refinement_v92/{name}")
        assert len(rows) == 15
        seen, by_pool = set(), [0, 0, 0]
        for row in rows:
            match = re.fullmatch(r"pool([0-2])_(\d+)_adaptive_energy", row["instance"])
            if match is None:
                raise ValueError(f"Unexpected stored decision instance: {row['instance']}")
            pool, seed = map(int, match.groups())
            assert (pool, seed) not in seen
            seen.add((pool, seed))
            complete = row.get("complete_seconds", row.get("total_seconds"))
            condition = lookup[pool, seed]
            stop = (complete <= 120 and Fraction(row["upper"]) <=
                    Fraction(51, 50) * Fraction(condition["midpoint_lower"]))
            by_pool[pool] += int(stop)
        counts[key] = by_pool
    recorded = read_json(root, "refinement_v92/extended_decision_comparison.json")
    assert counts == recorded["counts"] == {"R": [5, 5, 5], "M": [0, 5, 0]}
    return counts


def plot_public(root: Path, output: Path) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    aggregate = read_json(root, "contraction_real/aggregate.json")
    order = ["superconductivity", "parkinsons", "wine_white", "yacht"]
    by_name = {row["dataset"]: row for row in aggregate}
    assert set(by_name) == set(order)
    aggregate = [by_name[name] for name in order]
    counts = decision_counts(root)
    original = read_json(root, "energy_decisions/aggregate.json")
    energy = next(row for row in original["methods"] if row["method"] == "contraction_energy")["stops_by_pool"]
    assert energy == [5, 5, 3]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(4)
    colors = ["#257c8c", "#dc8a35", "#868d9c"]
    plotted = []
    for j, (method, label) in enumerate((("energy", "Adaptive energy"),
                                       ("moment_centered", "Centered moment"), ("full", "Full search"))):
        offset = (j - 1) * .22
        medians = []
        for k, row in enumerate(aggregate):
            result = row["methods"][method]
            widths = result["widths"]
            assert len(widths) == 5 and all(np.isfinite(v) and v > 0 for v in widths)
            assert np.isclose(np.median(widths), result["median_width"], rtol=1e-12, atol=0)
            medians.append(result["median_width"])
            axes[0].scatter(np.full(5, x[k] + offset) + np.linspace(-.035, .035, 5), widths,
                            s=9, color=colors[j], edgecolor="white", linewidth=.3, zorder=3)
            plotted.append(dict(dataset=row["dataset"], method=method, widths=widths,
                                median_width=result["median_width"]))
        axes[0].bar(x + offset, medians, width=.19, color=colors[j], label=label, alpha=.88)
    axes[0].axhline(1e-5, color="#b55252", ls="--", lw=.9)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, ["Supercond.", "Parkinsons", "Wine", "Yacht"])
    axes[0].set_ylabel("Certified interval width")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.26), ncol=3, frameon=False, fontsize=7.5)
    bottom = np.zeros(3)
    for pool, color in enumerate(("#257c8c", "#6a9f9d", "#aec9c5")):
        values = [energy[pool], counts["R"][pool], counts["M"][pool]]
        axes[1].bar(np.arange(3), values, bottom=bottom, color=color, width=.62, label=f"Pool {pool + 1}")
        bottom += values
    axes[1].set_xticks(np.arange(3), ["Energy\n10 seconds", "Refined\n120 seconds", "Moment\n120 seconds"])
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].set_ylim(0, 16.1)
    axes[1].set_yticks([0, 5, 10, 15])
    axes[1].set_ylabel("Certified STOPs\n(15 conditions)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, 1.26), ncol=3, frameon=False, fontsize=7.5)
    for i, value in enumerate(bottom):
        axes[1].text(i, value + .23, str(int(value)), ha="center", fontsize=8)
    for ax, label in zip(axes, "AB"):
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
        ax.text(-.15, 1.13, label, transform=ax.transAxes, fontweight="bold", fontsize=12)
    fig.subplots_adjust(left=.08, right=.99, bottom=.23, top=.76, wspace=.32)
    return save_figure(fig, "public_energy_v84", output,
                       dict(sources=["contraction_real/aggregate.json", "energy_decisions/aggregate.json",
                                     "energy_decisions/conditions.json", "refinement_v92/frozen_decisions.json",
                                     "refinement_v92/decisions_baseline_extended.json"],
                            widths=plotted, stops_by_pool=dict(energy=energy, refined=counts["R"], moment=counts["M"]),
                            deadlines_seconds=[10, 120, 120], target_width=1e-5))


def plot_calibration(root: Path, output: Path) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    records = read_json(root, "dimension_and_decisions/calibration_endpoints.json")
    sensitivity = read_json(root, "dimension_and_decisions/calibration_sensitivity.json")
    methods = ("anchor", "gradient", "metric_pod", "metric_greedy", "full")
    endpoints = {(row["horizon_seconds"], row["job"], row["method"]): row["endpoint"]
                 for row in records["endpoints"]}
    assert len(records["conditions"]) == 12 and len(endpoints) == 120
    grid = set(np.linspace(0, 1, 501).tolist() + [.25])
    for endpoint in endpoints.values():
        if endpoint is not None:
            for value in (endpoint["lower_K2"], endpoint["upper_K2"]):
                if 0 <= value <= 1:
                    grid.update([value, np.nextafter(value, -np.inf), np.nextafter(value, np.inf)])
    grid = sorted(value for value in grid if 0 <= value <= 1)
    assert len(grid) == sensitivity["grid_points"]
    curves = []
    for method in methods:
        rows = []
        for threshold in grid:
            labels = []
            for job in range(12):
                endpoint = endpoints[10.0, job, method]
                labels.append("UNRESOLVED" if endpoint is None else
                              "STOP" if endpoint["upper_K2"] <= threshold else
                              "VIOLATING" if endpoint["lower_K2"] > threshold else "UNRESOLVED")
            counts = Counter(labels)
            rows.append(dict(threshold_K2=float(threshold), STOP=counts["STOP"],
                             definite=counts["STOP"] + counts["VIOLATING"]))
        primary = rows[grid.index(.25)]
        expected = next(row for row in sensitivity["primary_counts"] if row["method"] == method)
        assert primary["STOP"] == expected["STOP"] and primary["definite"] == expected["definite"]
        curves.append(dict(method=method, rows=rows))
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), layout="constrained")
    colors = {"anchor": "#888888", "gradient": "#0072b2", "metric_pod": "#d55e00",
              "metric_greedy": "#009e73", "full": "#7a5195"}
    labels = {"anchor": "Anchor", "gradient": "Gradient", "metric_pod": "Metric POD",
              "metric_greedy": "Metric greedy", "full": "Full"}
    for curve in curves:
        method = curve["method"]
        for ax, key in zip(axes, ("definite", "STOP")):
            ax.step(grid, [row[key] for row in curve["rows"]], where="post", label=labels[method],
                    color=colors[method], ls="--" if method == "full" else "-")
    for ax, title in zip(axes, ("A  Definite decisions", "B  STOP decisions")):
        ax.set(xlim=(0, 1), ylim=(-.2, 12.5), xlabel=r"Training-objective tolerance ($\mathrm{K}^2$)",
               ylabel="Conditions / 12", title=title)
        ax.axvline(.25, color="black", ls=":", lw=.8)
    axes[1].legend(loc="lower right", frameon=False, ncol=2, fontsize=6.5)
    return save_figure(fig, "calibration_threshold_v83", output,
                       dict(sources=["dimension_and_decisions/calibration_endpoints.json",
                                     "dimension_and_decisions/calibration_sensitivity.json"],
                            horizon_seconds=10, primary_threshold_K2=.25, curves=curves))


def plot_zorro(root: Path, output: Path) -> dict:
    import matplotlib.pyplot as plt
    import numpy as np
    curves = read_json(root, "matched_external/anytime_curves.json")
    datasets = {"superconductivity": "Supercond.", "parkinsons": "Parkinsons",
                "wine_white": "Wine Quality", "yacht": "Yacht"}
    styles = (("adaptive_energy", "#257c8c", "Adaptive energy"),
              ("moment_centered", "#dc8a35", "Centered moment"),
              ("ZORRO_refined", "#726891", "ZORRO refined"))
    assert len(curves) == 12
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2))
    plotted = []
    for ax, (dataset, title) in zip(axes.flat, datasets.items()):
        for method, color, label in styles:
            rows = [row for row in curves if row["dataset"] == dataset and row["method"] == method]
            assert len(rows) == 1
            trace = rows[0]["trace"]
            assert trace and all(0 <= time <= 10 and np.isfinite(width) and width > 0 for time, width in trace)
            assert all(left[0] <= right[0] for left, right in zip(trace, trace[1:]))
            if trace[-1][0] < 10:
                trace = trace + [[10, trace[-1][1]]]
            ax.step([row[0] for row in trace], [row[1] for row in trace], where="post",
                    color=color, label=label, lw=1.2)
            plotted.append(dict(dataset=dataset, method=method, trace=trace))
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xlim(0, 10)
        ax.grid(alpha=.15)
        ax.set_xlabel("Complete seconds")
        ax.set_ylabel("Interval width")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "zorro_refinement_v91", output,
                       dict(sources=["matched_external/anytime_curves.json"], complete_cap_seconds=10,
                            width_units="normalized objective", curves=plotted))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"))
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--figures", nargs="+", choices=("structure", "public", "calibration", "zorro"),
                        default=["structure", "public", "calibration", "zorro"])
    args = parser.parse_args()
    cpu = pin_one_cpu(args.cpu)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    builders = {"structure": plot_structure, "public": plot_public,
                "calibration": plot_calibration, "zorro": plot_zorro}
    reports = []
    for name in args.figures:
        # The two later figures retain the original 9-point base typography.
        with plt.rc_context({"font.size": 9 if name in ("public", "zorro") else 8}):
            reports.append(builders[name](args.results_dir, args.output_dir))
    print(json.dumps(dict(status="PASS", logical_cpu=cpu, numerical_threads=1,
                          raw_records_opened=False, experiment_solvers_run=False, figures=reports)))


if __name__ == "__main__":
    main()
