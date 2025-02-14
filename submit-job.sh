#!/bin/bash
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 2                    # Number of tasks (processes)
#SBATCH -c 1                    # CPU cores per task
#SBATCH --mem=12g              # Increased memory for 2 GPUs
#SBATCH -p qTRDGPUM            # Partition name
#SBATCH -t 1440                # Time limit in minutes
#SBATCH -J your-job-name  # Job name
#SBATCH -e error%A.err         # Error file
#SBATCH -o out%A.out           # Output file
#SBATCH -A psy53c17            # Account
#SBATCH --mail-type=ALL        # Mail notifications
#SBATCH --mail-user=your-email@email.com
#SBATCH --oversubscribe
#SBATCH --gres=gpu:1           # Request 2 GPUs

sleep 10s

echo $HOSTNAME >&2

conda activate $MYNEWENV

python curriculum_training.py

sleep 10s