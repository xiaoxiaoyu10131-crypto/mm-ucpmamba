#!/bin/bash
#SBATCH --job-name=xxy_async
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mm-ucpmamba-async.log
#SBATCH --error=mm-ucpmamba-async.err
set -Eeuo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate uvmrnn
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_mm_ucpmamba_async.sh"
