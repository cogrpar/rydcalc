#!/bin/bash
#SBATCH --job-name=rydberg
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=990G
#SBATCH --time=2:00:00


module purge
module load python/3.11

source ~/venvs/rydberg/bin/activate

python Yb_Rb_search.py