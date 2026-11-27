"""
Parses instance files once and stores each parsed CSP_Data next to its source file
(e.g. J60_1.wcnf -> J60_1.wcnf.v1.pt), so training can load them quickly.
Training uses these caches when the dataset config has "cache": true.

Usage:
    python pickle_data.py "data_paper/rcpsp/train/*.wcnf" "data_paper/rcpsp/val/*.wcnf" --workers 32
"""
import os
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool

import torch
from tqdm import tqdm

from src.data.dataset import load_cached, cache_path


def work(file_path):
    torch.set_num_threads(1)  # one process per core; no extra threads inside each worker
    existed = os.path.exists(cache_path(file_path))
    load_cached(file_path)  # parses and writes the cache unless an up-to-date one exists
    return existed


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('patterns', nargs='+', help='Glob patterns of instance files (quote them)')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    for pattern in args.patterns:
        files = sorted(glob(pattern))
        if not files:
            raise SystemExit(f'No files match {pattern}')
        # largest files first, so one huge file does not hold up the end of the run
        files.sort(key=os.path.getsize, reverse=True)

        with Pool(args.workers) as pool:
            existed = list(tqdm(pool.imap_unordered(work, files), total=len(files), desc=pattern, mininterval=30))
        print(f'{pattern}: {len(files)} files, {len(files) - sum(existed)} newly cached, {sum(existed)} already had a cache', flush=True)
