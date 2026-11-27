#!/usr/bin/env bash
# -------------------------------------------------
#  Pickle (cache) the RCPSP training sets on Capella.
#  Only the CPU cores are used; one GPU is requested because Capella jobs need one.
#  Submit:  sbatch job-pickle.sh
#  Already cached files are skipped, so a resubmit continues where a job stopped.
# -------------------------------------------------
#SBATCH --job-name=pickle_rcpsp
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14              # a quarter of a node = the share of one GPU
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00                 # d-hh:mm:ss – max wall‑time
#SBATCH --output=outputs/%x-%j.out    # stdout
#SBATCH --error=errors/%x-%j.err      # stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=thanh-van.huynh@mailbox.tu-dresden.de

set -e
cd /data/horse/ws/thhu694i-anycsp
hostname
date

module purge
module load release/24.04 GCCcore/11.3.0 Python/3.10.4
source .venv-capella/bin/activate
export OMP_NUM_THREADS=1   # parallelism comes from the worker processes

# expected cache size: ~17x the .wcnf size, i.e. ~100 GB per training set
python pickle_data.py \
    "data_paper/rcpsp/train/*.wcnf" \
    "data_paper/rcpsp/train_clauses5/*.wcnf" \
    "data_paper/rcpsp/train_clauses10/*.wcnf" \
    "data_paper/rcpsp/train_clauses20/*.wcnf" \
    --workers $SLURM_CPUS_PER_TASK

du -sh data_paper/rcpsp/train*/
date
