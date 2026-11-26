#!/usr/bin/env python3
"""
Add the learnt (conflict) clauses dumped by WMaxCDCL as hard clauses to the instances.

For every <name>.wcnf in `data`:
  - read logs/<name>.out and collect all dumped clauses ("c CORE ...",
    "c TIER2 ...", "c LOCAL ..." lines) over all dumps of the run
  - clauses with variables outside the instance (PB helpers, only in old
    unfiltered logs) are skipped
  - rank the unique clauses by
      1. tier: core, then tier2, then local (best tier the clause was ever in)
      2. lbd: low first (smallest lbd the clause was dumped with)
      3. survival: clauses that appear in more dumps first
      4. position: first appearance in the log
  - add the first N = round(max_clauses * <original #clauses>) of them
    (all of them if max_clauses is None or there are fewer)
  - write output/<name>.wcnf = original instance + the selected clauses as hard
    clauses (weight = top), with the clause count in the "p wcnf" line updated

All dumped clauses are satisfied by every optimal solution (they are dumped when
the upper bound improves), so the optimum and its cost do not change.
The selection is deterministic.

Usage (run from the repo root):
  python3 add_conflict_clauses.py
  python3 add_conflict_clauses.py DATA LOGS OUTPUT [--max-clauses F] [-j N]
  (use 'none' for no limit, e.g. --max-clauses none)
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---- settings (can be overridden on the command line) ----
data = "data_paper_clean/rcpsp/val"
logs = "logs_paper/rcpsp/val"
output = "data_paper_clean/rcpsp/val_clauses5"
max_clauses = 0.05    # add max_clauses * <original #clauses> clauses (None = all)

TIER_RANK = {"CORE": 0, "TIER2": 1, "LOCAL": 2}
CLAUSE_LINE = re.compile(r"^c (CORE|TIER2|LOCAL) (?!begin|end)(.*)$")


def read_header(wcnf: Path):
    """Return (num_vars, num_clauses, top) from the 'p wcnf' line."""
    with wcnf.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("p "):
                tok = line.split()
                if tok[1] != "wcnf" or len(tok) < 5:
                    raise ValueError(f"{wcnf}: expected 'p wcnf <vars> <clauses> <top>', got: {line.strip()}")
                return int(tok[2]), int(tok[3]), tok[4]
    raise ValueError(f"{wcnf}: no 'p wcnf' line (run clean_data_paper.py first)")


def collect_clauses(log: Path, num_vars: int):
    """{sorted literal tuple: [best tier rank, smallest lbd, #dumps, first position]}"""
    info = {}
    out_of_range = 0
    with log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("c "):
                continue
            m = CLAUSE_LINE.match(line)
            if not m:
                continue
            tok = m.group(2).replace(",", " ").split()
            if not tok or tok[-1] != "0":
                raise ValueError(f"{log}: clause not terminated by 0: {line.strip()[:100]}")
            rank = TIER_RANK[m.group(1)]
            lbd = int(tok[0].split("=")[-1])  # "lbd=5" (new) or "5" (first version)
            lits = tuple(sorted(int(x) for x in tok[1:-1]))
            if any(abs(l) > num_vars for l in lits):
                out_of_range += 1
                continue
            r = info.get(lits)
            if r is None:
                info[lits] = [rank, lbd, 1, len(info)]
            else:  # a clause is in one tier per dump, so each line = one more dump survived
                r[0] = min(r[0], rank)
                r[1] = min(r[1], lbd)
                r[2] += 1
    return info, out_of_range


def select(info: dict, num_clauses: int, max_clauses):
    """Return (selected clauses, how many of them per tier)."""
    ranked = sorted(info, key=lambda c: (info[c][0], info[c][1], -info[c][2], info[c][3]))
    if max_clauses is not None:
        ranked = ranked[: round(max_clauses * num_clauses)]
    per_tier = [0, 0, 0]
    for c in ranked:
        per_tier[info[c][0]] += 1
    return ranked, per_tier


def process(wcnf: Path, log: Path, dst: Path, max_clauses):
    num_vars, num_clauses, top = read_header(wcnf)
    info, out_of_range = collect_clauses(log, num_vars)
    clauses, per_tier = select(info, num_clauses, max_clauses)

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with wcnf.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as out:
        for line in src:
            if line.startswith("p "):
                out.write(f"p wcnf {num_vars} {num_clauses + len(clauses)} {top}\n")
            else:
                out.write(line)
        for lits in clauses:
            out.write(f"{top} " + " ".join(map(str, lits)) + " 0\n")
    tmp.replace(dst)  # atomic: no half-written files if interrupted
    return len(info), per_tier, out_of_range


def optional(type_):
    """argparse type that also accepts 'none'."""
    return lambda s: None if s.lower() == "none" else type_(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=data)
    ap.add_argument("logs", nargs="?", default=logs)
    ap.add_argument("output", nargs="?", default=output)
    ap.add_argument("--max-clauses", type=optional(float), default=max_clauses,
                    help="add MAX_CLAUSES * <original #clauses> clauses ('none' = all)")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    args = ap.parse_args()

    data_dir, logs_dir, out_dir = Path(args.data), Path(args.logs), Path(args.output)
    instances = sorted(data_dir.glob("*.wcnf"))
    if not instances:
        sys.exit(f"no .wcnf files found in {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    todo, missing = [], []
    for wcnf in instances:
        log = logs_dir / f"{wcnf.stem}.out"
        if log.is_file():
            todo.append((wcnf, log, out_dir / wcnf.name))
        else:
            missing.append(wcnf.name)

    print(f"max clauses: {args.max_clauses} x original")
    print(f"{len(instances)} instances, {len(todo)} with log, {len(missing)} without log (skipped)")

    unique_total = skipped_total = 0
    added = [0, 0, 0]
    errors = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process, w, l, d, args.max_clauses): w
                for w, l, d in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                unique, per_tier, out_of_range = fut.result()
            except Exception as e:  # keep going, report at the end
                errors.append((futs[fut].name, e))
                continue
            unique_total += unique
            added = [a + b for a, b in zip(added, per_tier)]
            skipped_total += out_of_range
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    print(f"done: {len(todo) - len(errors)} files written to {out_dir}")
    print(f"unique dumped clauses: {unique_total}, added: {sum(added)} "
          f"(core={added[0]}, tier2={added[1]}, local={added[2]})")
    if skipped_total:
        print(f"clause lines skipped (variables outside the instance): {skipped_total}")
    for name in missing[:20]:
        print(f"  no log: {name}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more without log")
    for name, e in errors:
        print(f"  ERROR {name}: {e}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
