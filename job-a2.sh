#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name a2
#SBATCH --output=a2_%j.out

module load anaconda3/5.2.0
module load gcc/7.3.0
module load cmake 

make run
#python3 perfs_student.py
