#!/bin/bash

#SBATCH --output=res_%j.txt             # Output and error file name (%j expands to jobId)
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --partition=csc413                 # Partition name, if you have specific partitions
#SBATCH --gres=gpu                    # Request GPU generic resources; syntax depends on configuration
#SBATCH --time=08:00:00                 # Time limit hrs:min:sec

# # Load necessary modules
# module load python/3.8
# module load cuda/11.2

# # Activate your virtual environment, if necessary
# source /path/to/your/virtualenv/bin/activate

# module load cuda  # Load CUDA module, if available

# Run your Python training script
# /local/bin/python3 test.py
/local/bin/python3 train.py
# python3 train.py

# Deactivate virtual environment, if used
# deactivate