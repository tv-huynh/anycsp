#!/usr/bin/env bash
# -------------------------------------------------
#  SLURM OPTIONS
# -------------------------------------------------
#SBATCH --job-name=paper_rcpsp_clauses5
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00                 # d-hh:mm:ss – max wall‑time
#SBATCH --output=outputs/%x-%j.out    # stdout
#SBATCH --error=errors/%x-%j.err      # stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=thanh-van.huynh@mailbox.tu-dresden.de

# -------------------------------------------------
#  LOAD modules & activate the virtual environment
# -------------------------------------------------
hostname
date

module purge
#module load release/24.04 GCCcore/12.2.0 Python/3.10.8
#source .venv/bin/activate
module load release/24.04 GCCcore/11.3.0 Python/3.10.4
source .venv-capella/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # tell PyTorch how many threads to use
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce memory fragmentation

# -------------------------------------------------
#  RUN the training script
# -------------------------------------------------
#python train.py --model_dir models/PWMAXSAT_weightfeatures --config configs/PWMAXSAT/PWMAXSAT_default.json

#python train.py --model_dir models/N25_limit1 --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/NSP/NSP_N25_limit1.json --no_load_opt_state
#python train.py --model_dir models/N25_nolimit --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/NSP/NSP_N25_nolimit.json --no_load_opt_state
#python train.py --model_dir models/N50_limit1 --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/NSP/NSP_N50_limit1.json --no_load_opt_state
#python train.py --model_dir models/N50_nolimit --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/NSP/NSP_N50_nolimit.json --no_load_opt_state

#python train.py --model_dir models/RCPSP --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/RCPSP/RCPSP_default.json --no_load_opt_state

#python train.py --model_dir models/RCPSP_clauses20 --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/RCPSP/RCPSP_clauses20.json --no_load_opt_state
#python train.py --model_dir models/RCPSP_clauses10 --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/RCPSP/RCPSP_clauses10.json --no_load_opt_state
python train.py --model_dir models/RCPSP_clauses5 --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/RCPSP/RCPSP_clauses5.json --no_load_opt_state

#python train.py --model_dir models/CA_arbitrary --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/CA/CA_arbitrary_default.json --no_load_opt_state
#python train.py --model_dir models/CA_regions --pretrained_dir models/PWMAXSAT_weightfeatures --config configs/CA/CA_regions_default.json --no_load_opt_state

#python train.py --model_dir models/NSP_wcnf150 --pretrained_dir models/PWMAXSAT --config configs/NSP/NSP_wcnf_default.json --no_load_opt_state
#python train.py --model_dir models/NSP_N25_bnb_assignment --pretrained_dir models/PWMAXSAT --config configs/NSP/NSP_N25_bnb.json --no_load_opt_state
#python train.py --model_dir models/NSP_N50_bnb --pretrained_dir models/PWMAXSAT --config configs/NSP/NSP_N50_bnb.json --no_load_opt_state

date
