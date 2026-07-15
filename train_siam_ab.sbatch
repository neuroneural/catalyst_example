#!/bin/bash
# =============================================================================
# 2x H100 DDP training for the Kervadec boundary-loss A/B (siam18).
#
# One PARAMETERIZED job: pass the config via --export=ALL,CONFIG=<name>. Catalyst
# picks DistributedDataParallelEngine automatically when >1 GPU is visible
# (curriculum_training.py get_engine) and mp.spawns one rank per GPU itself --
# so this is ONE task that sees both GPUs, NOT --ntasks=2.
#
# All outputs (checkpoints + tee'd stdout) go to the world-readable /data tree.
# Both A/B configs resume from the SAME frozen start.pth and write to their own
# logdir (set inside each yaml), so the only difference is the boundary term.
#
# One-time, freeze the shared start checkpoint:
#   mkdir -p /data/users4/splis/data/segmentation/synth18_boundary_ab/slurm
#   cp ./logs/tmp/synth18_gn_hdc_deep_turbo16/model.last.pth \
#      /data/users4/splis/data/segmentation/synth18_boundary_ab/start.pth
#
# Submit BOTH (control queued right after boundary on the same 2 H100s):
#   JID=$(sbatch --parsable --job-name=siam18_boundary \
#           --export=ALL,CONFIG=gn_hdc_deep_fast_turbo_siam18_boundary train_siam_ab.sbatch)
#   sbatch --dependency=afterany:$JID --job-name=siam18_ctrl \
#           --export=ALL,CONFIG=gn_hdc_deep_fast_turbo_siam18_ctrl   train_siam_ab.sbatch
# (Drop the --dependency line to let the scheduler order them itself.)
# =============================================================================
#SBATCH --job-name=siam18_ab
#SBATCH --partition=qTRDGPUH
#SBATCH --account=psy53c17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:2
#SBATCH --time=5-00:00:00
#SBATCH --output=/data/users4/splis/data/segmentation/synth18_boundary_ab/slurm/%x_%j.out
#SBATCH --error=/data/users4/splis/data/segmentation/synth18_boundary_ab/slurm/%x_%j.err
#SBATCH --verbose

set -euo pipefail

# world-readable outputs (dirs 755, files 644)
umask 0022

CONFIG="${CONFIG:-gn_hdc_deep_fast_turbo_siam18_boundary}"
REFINERY="${REFINERY:-/home/users/splis/craft/meshnet/refinery}"
VENV="${VENV:-/trdapps/linux-x86_64/envs/plis_venv/torch/bin/activate}"
SEGDIR="/data/users4/splis/data/segmentation/synth18_boundary_ab"

mkdir -p "${SEGDIR}/slurm"

# persistent_workers=true (base config) needs the glibc allocator tamed or the
# long-lived Mongo workers leak host RAM across the run.
export MALLOC_TRIM_THRESHOLD_=0 MALLOC_ARENA_MAX=2

echo "host=$(hostname)  job=${SLURM_JOB_ID}  config=${CONFIG}" >&2
nvidia-smi -L >&2

# shellcheck disable=SC1090
source "${VENV}"
cd "${REFINERY}"

python curriculum_training_fast.py \
    --config-dir=conf \
    --config-name="${CONFIG}" \
    2>&1 | tee "${SEGDIR}/slurm/${CONFIG}.${SLURM_JOB_ID}.log"
