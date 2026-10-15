"""
Parses a Slurm-style .out log containing lines like:

    5226: s OPTIMUM FOUND (took 1s)
    2247: s NOSTATUS (took 7240s)

determines which corresponding <number>.wcnf instances were solved
(OPTIMUM FOUND) vs not (NOSTATUS), and then randomly splits:

    - solved instances       -> data/training/<data_dir> and data/validation/<data_dir>
    - unsolved (NOSTATUS)    -> data/test/<data_dir>

according to (integers) train_instances / val_instances / test_instances.

To avoid data leakage between train and validation, the solved pool is
first partitioned into two DISJOINT sub-pools, sized proportionally to
train_instances : val_instances. Each sub-pool (and the unsolved pool,
for test) is then independently filled via oversampling: if the
requested count exceeds the sub-pool size, the full sub-pool is copied
repeatedly (each repeat pass suffixed with "_passK" to avoid
overwriting on disk) until fewer than one full sub-pool remains, then
the remainder is filled with a random sample without replacement.


    HOW TO USE:
    python process_bnb_results.py \
    --bnb_output bnb_output.out \
    --data_dir j60 \
    --train_instances 900 \
    --val_instances 200 \
    --test_instances 100
    
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import numpy as np

LINE_PATTERN = re.compile(r"^\s*(\S+):\s*s\s+(OPTIMUM FOUND|NOSTATUS)")


def parse_out_file(out_path):
    """Parse the .out file and return an (N, 2) object array.

    Column 0: instance number as string.
    Column 1: bool, True for OPTIMUM FOUND, False for NOSTATUS.
    """
    out_path = Path(out_path)
    rows = []
    with out_path.open("r") as f:
        for line in f:
            match = LINE_PATTERN.match(line)
            if match is None:
                continue
            number_str, status = match.group(1), match.group(2)
            found = status == "OPTIMUM FOUND"
            rows.append((number_str, found))

    if not rows:
        raise ValueError(f"No matching status lines found in {out_path}")

    return np.array(rows, dtype=object)


def find_wcnf_files(status_array, data_dir):
    """Resolve <number>.wcnf paths for solved (True) and unsolved (False) instances.

    Returns (solved_paths, unsolved_paths), skipping any missing files with a warning.
    """
    data_dir = Path(data_dir)
    solved, unsolved = [], []

    for number_str, found in status_array:
        src = data_dir / f"{number_str}.wcnf"
        if not src.exists():
            print(f"Warning: missing file {src} (skipped)")
            continue
        (solved if bool(found) else unsolved).append(src)

    return solved, unsolved


def split_pool_by_ratio(files, ratio_a, ratio_b, rng):
    """Split files into two disjoint pools sized proportionally to ratio_a : ratio_b.

    Used to prevent data leakage: train and validation are drawn from
    non-overlapping sub-pools of the solved instances, instead of both
    being sampled independently from the full solved pool.
    """
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    total_ratio = ratio_a + ratio_b
    n_a = round(n * ratio_a / total_ratio) if total_ratio > 0 else 0
    pool_a = files[:n_a]
    pool_b = files[n_a:]
    return pool_a, pool_b


def _copy_with_pass_suffix(src: Path, dest_dir: Path, pass_idx: int) -> None:
    if pass_idx == 0:
        dest_name = src.name
    else:
        dest_name = f"{src.stem}_pass{pass_idx}{src.suffix}"
    shutil.copy2(src, dest_dir / dest_name)


def sample_with_oversampling(files, dest_dir, count, rng):
    """Copy `count` files into dest_dir, drawn from `files`.

    If count <= len(files): a plain random sample without replacement.
    If count > len(files): repeatedly copy the full pool (each repeat
    pass suffixed with "_passK" to avoid overwriting), tracking how many
    instances have been placed in `in_folder`, until the remaining
    quota (count - in_folder) is <= len(files); then fill the remainder
    with a random sample without replacement from the pool.

    Returns the total number of files copied (== count).
    """
    n_available = len(files)
    if count == 0:
        dest_dir.mkdir(parents=True, exist_ok=True)
        return 0
    if n_available == 0:
        raise ValueError("No files available to sample from.")
    dest_dir.mkdir(parents=True, exist_ok=True)

    in_folder = 0
    pass_idx = 0
    while count - in_folder > n_available:
        for src in files:
            _copy_with_pass_suffix(src, dest_dir, pass_idx)
        in_folder += n_available
        pass_idx += 1

    remaining = count - in_folder
    sampled = rng.sample(files, remaining)
    for src in sampled:
        _copy_with_pass_suffix(src, dest_dir, pass_idx)
    in_folder += remaining

    return in_folder


def main():
    parser = argparse.ArgumentParser(
        description="Sort BnB-solved/unsolved wcnf instances and split into train/val/test."
    )
    parser.add_argument("--bnb_output", type=str, required=True,
                         help='Name of the .out file inside the "bnb_outputs" folder, e.g. "bnb_output.out"')
    parser.add_argument("--data_dir", type=str, required=True,
                         help='Name of the folder inside "data_savilerow/" containing the <number>.wcnf instance files, e.g. "N25"')
    parser.add_argument("--train_instances", type=int, required=True)
    parser.add_argument("--val_instances", type=int, required=True)
    parser.add_argument("--test_instances", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (set to -1 for non-reproducible sampling)")
    args = parser.parse_args()

    out_path = Path("bnb_outputs") / args.bnb_output
    data_dir = Path("data_savilerow") / args.data_dir
    train_dir = Path("data/training") / args.data_dir
    val_dir = Path("data/validation") / args.data_dir
    test_dir = Path("data/test") / args.data_dir

    rng = random.Random(None if args.seed == -1 else args.seed)

    status_array = parse_out_file(out_path)
    print("========================================")
    print(f"Parsed {status_array.shape[0]} status lines from {out_path}")

    solved_files, unsolved_files = find_wcnf_files(status_array, data_dir)
    print(f"Found {len(solved_files)} OPTIMUM FOUND instances and {len(unsolved_files)} NOSTATUS instances in {data_dir}")

    # --- Train / validation: split the solved pool into disjoint sub-pools FIRST (prevents leakage),
    #     then oversample each sub-pool independently if its quota exceeds its sub-pool size. ---
    train_pool, val_pool = split_pool_by_ratio(
        solved_files, args.train_instances, args.val_instances, rng
    )
    print(
        f"Split solved pool into disjoint sub-pools: "
        f"{len(train_pool)} for training, {len(val_pool)} for validation"
    )

    if args.train_instances > len(train_pool):
        print(
            f"\nWarning: train_instances = {args.train_instances} exceeds "
            f"{len(train_pool)} available instances in the training sub-pool. Oversampling with repeated passes."
        )
    if args.val_instances > len(val_pool):
        print(
            f"Warning: val_instances = {args.val_instances} exceeds "
            f"{len(val_pool)} available instances in the validation sub-pool. Oversampling with repeated passes."
        )

    n_train_copied = sample_with_oversampling(train_pool, train_dir, args.train_instances, rng)
    n_val_copied = sample_with_oversampling(val_pool, val_dir, args.val_instances, rng)

    # --- Test: drawn from the unsolved pool (independent of train/val, no leakage risk there) ---
    n_unsolved = len(unsolved_files)
    if args.test_instances > n_unsolved:
        print(
            f"Warning: test_instances = {args.test_instances} exceeds "
            f"{n_unsolved} available NOSTATUS instances. Oversampling with repeated passes.\n"
        )
    n_test_copied = sample_with_oversampling(unsolved_files, test_dir, args.test_instances, rng)

    print(f"Copied {n_train_copied} files to {train_dir}")
    print(f"Copied {n_val_copied} files to {val_dir}")
    print(f"Copied {n_test_copied} files to {test_dir}")
    print("========================================")


if __name__ == "__main__":
    main()