#!/usr/bin/env python3
"""
Plot upper-bound convergence of the BnB solver against AnyCSP.

For every instance file <x>.out in path_bnb:
  - extract the UB-improvement history from lines like
        c UB hist: t=3.213, UB=498, solutionCost,UB,fixedCostBySearch,derivedCost: (0,  79,  0,  419)
    and plot UB (y) over t (x).
  - if path_anycsp contains a file with the same name that says
    's SATISFIABLE', extract the best upper bound from its 'o <value>'
    line and draw it as a red horizontal reference line.
  - save the figure as <x>.png in path_plots.

Instances without any 'c UB hist' lines are skipped.

HOW TO USE:
python plot_ub.py --path_bnb bnb_assignments/test --path_anycsp assignments/N25_limit1_1000steps --path_plots plots
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe (no display needed, e.g. on the cluster)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

UB_HIST_RE = re.compile(r"c\s+UB\s+hist:\s*t=([-+0-9.eE]+)\s*,\s*UB=([-+0-9.eE]+)")
SAT_RE = re.compile(r"^s\s+SATISFIABLE")
ANYCSP_UB_RE = re.compile(r"^o\s+([-+0-9.eE]+)")


def parse_ub_history(path: Path):
    """Return (t, ub) as numpy arrays from 'c UB hist' lines (empty arrays if none)."""
    ts, ubs = [], []
    with path.open() as f:
        for line in f:
            m = UB_HIST_RE.search(line)
            if m:
                ts.append(float(m.group(1)))
                ubs.append(float(m.group(2)))
    return np.array(ts), np.array(ubs)


def parse_anycsp_ub(path: Path):
    """Return the best UB from an AnyCSP output file, or None if not solved."""
    solved = False
    best = None
    with path.open() as f:
        for line in f:
            if SAT_RE.match(line):
                solved = True
            m = ANYCSP_UB_RE.match(line)
            if m:  # keep the last 'o' line -> best bound found
                best = float(m.group(1))
    return best if solved else None


def plot_instance(stem, t, ub, ub_anycsp, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, ub, color="tab:blue", linewidth=1.2, marker="o", markersize=3,
            label="BnB UB history")
    # alternative: ax.step(t, ub, where="post", ...) for a proper step function
    if ub_anycsp is not None:
        ax.plot((t[0], t[-1]), (ub_anycsp, ub_anycsp), color="red", linewidth=1.8, linestyle="--",
                   label=f"AnyCSP UB = {ub_anycsp:g}")
        #ax.axhline(ub_anycsp, color="red", linewidth=1.8, linestyle="--",
        #           label=f"AnyCSP UB = {ub_anycsp:g}")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel("upper bound")
    ax.set_title(f"instance {stem}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    # ax.set_xscale("log")  # uncomment if runtimes span orders of magnitude
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path_bnb", default="bnb_assignments/test")
    p.add_argument("--path_anycsp", default="assignments/N25_limit1_1000steps")
    p.add_argument("--path_plots", default="plots")
    args = p.parse_args()

    path_bnb = Path(args.path_bnb)
    path_anycsp = Path(args.path_anycsp)
    path_plots = Path(args.path_plots)

    if not path_bnb.is_dir():
        sys.exit(f"error: {path_bnb} is not a directory")
    path_plots.mkdir(parents=True, exist_ok=True)

    n_plotted = n_skipped = n_no_ref = 0
    for bnb_file in sorted(path_bnb.glob("*.out")):
        stem = bnb_file.stem

        t, ub = parse_ub_history(bnb_file)
        if t.size == 0:  # checker: no 'c UB hist' lines -> skip this instance
            print(f"[skip] {bnb_file.name}: no 'c UB hist' lines")
            n_skipped += 1
            continue

        anycsp_file = path_anycsp / bnb_file.name
        ub_anycsp = parse_anycsp_ub(anycsp_file) if anycsp_file.is_file() else None
        if ub_anycsp is None:
            n_no_ref += 1

        plot_instance(stem, t, ub, ub_anycsp, path_plots / f"{stem}.png")
        n_plotted += 1
        msg = f"[plot] {stem}: {t.size} BnB UB improvements"
        msg += f", AnyCSP UB = {ub_anycsp:g}, UB gap = {ub_anycsp-ub[-1]:g}" if ub_anycsp is not None else ", no AnyCSP reference"
        print(msg)

    print(f"\ndone: {n_plotted} plot(s) saved to {path_plots}, "
          f"{n_skipped} skipped (no UB hist), {n_no_ref} without AnyCSP reference")


if __name__ == "__main__":
    main()
