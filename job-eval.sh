#!/usr/bin/env bash
# -------------------------------------------------
#  SLURM OPTIONS
# -------------------------------------------------
#SBATCH --job-name=inference_n25_limit1_1000steps_boost
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00                 # d-hh:mm:ss – max wall‑time
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

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # tell PyTorch how many threads to use

# -------------------------------------------------
#  RUN the training script
# -------------------------------------------------
python evaluate_pwmaxsat.py --model_dir models/N25_limit1 --checkpoint_name best --data_path 'data/validation/N25_limit1/*.wcnf' --output_dir assignments/N25_limit1_1000steps_boost --network_steps 1000 --num_boost 50
#python evaluate_pwmaxsat.py --model_dir models/N25_nolimit --checkpoint_name best --data_path 'data/training/N25_nolimit/*.wcnf' 'data/validation/N25_nolimit/*.wcnf' --output_dir assignments/N25_nolimit --network_steps 1
#python evaluate_pwmaxsat.py --model_dir models/N50_limit1 --checkpoint_name best --data_path 'data/training/N50_limit1/*.wcnf' 'data/validation/N50_limit1/*.wcnf' --output_dir assignments/N50_limit1 --network_steps 1
#python evaluate_pwmaxsat.py --model_dir models/N50_nolimit --checkpoint_name best --data_path 'data/training/N50_nolimit/*.wcnf' 'data/validation/N50_nolimit/*.wcnf' --output_dir assignments/N50_nolimit --network_steps 1

#python evaluate_pwmaxsat.py --model_dir models/RCPSP --checkpoint_name best --data_path 'data/training/RCPSP/*.wcnf' 'data/validation/RCPSP/*.wcnf' --output_dir assignments/RCPSP --network_steps 1

#python evaluate_pwmaxsat.py --model_dir models/CA_arbitrary --checkpoint_name best --data_path 'data/training/CA_arbitrary/*.wcnf' 'data/validation/CA_arbitrary/*.wcnf' --output_dir assignments/CA_arbitrary --network_steps 1
#python evaluate_pwmaxsat.py --model_dir models/CA_regions --checkpoint_name best --data_path 'data/training/CA_regions/*.wcnf' 'data/validation/CA_regions/*.wcnf' --output_dir assignments/CA_regions --network_steps 1

date