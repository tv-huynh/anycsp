#!/usr/bin/env bash
# -------------------------------------------------
#  SLURM OPTIONS
# -------------------------------------------------
#SBATCH --job-name=nsp_n50_bnb
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00                 # hh:mm:ss – max wall‑time
#SBATCH --output=outputs/%x-%j.out    # stdout
#SBATCH --error=errors/%x-%j.err      # stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=thanh-van.huynh@mailbox.tu-dresden.de

# -------------------------------------------------
#  LOAD modules & activate the virtual environment
# -------------------------------------------------
#module purge
module purge
module load release/24.04 GCCcore/12.2.0 Python/3.10.8
source .venv/bin/activate
#module load release/24.04 GCCcore/11.3.0 Python/3.10.4
#source .venv_capella/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # tell PyTorch how many threads to use

# -------------------------------------------------
#  RUN the training script
# -------------------------------------------------
#python train.py --model_dir models/PWMAXSAT_mpweights --config configs/PWMAXSAT/PWMAXSAT_default.json
#python train.py --model_dir models/NSP --pretrained_dir models/PWMAXSAT --config configs/NSP/NSP_default.json --no_load_opt_state
#python train.py --model_dir models/NSP_wcnf150 --pretrained_dir models/PWMAXSAT --config configs/NSP/NSP_wcnf_default.json --no_load_opt_state
python train.py --model_dir models/NSP_N25_bnb_assignment --pretrained_dir models/PWMAXSAT --config configs/NSP/NSP_N25_bnb.json --no_load_opt_state
#python train.py --model_dir models/NSP_N50_bnb --pretrained_dir models/PWMAXSAT --config configs/NSP/NSP_N50_bnb.json --no_load_opt_state