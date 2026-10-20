#!/bin/bash
#SBATCH --job-name=mm_utae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mm-utae-%j.log
#SBATCH --error=mm-utae-%j.err

set -e

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate uvmrnn

cd /share/home/xyxiao/xxy/expduoyuan/mm-ucpmamba108/utae

PROFILE="${PROFILE:-comparison}"
RES_DIR="${RES_DIR:-/share/home/xyxiao/dataset/expduoyuan/mm-utae_${PROFILE}}"

case "$PROFILE" in
  comparison) HEAD=legacy; AUX_WEIGHT=0; FINAL_BN_INIT=ones ;;
  paper) HEAD=logits; AUX_WEIGHT=0.5; FINAL_BN_INIT=legacy ;;
  *) echo "PROFILE must be comparison or paper" >&2; exit 2 ;;
esac

for fold in 2 3 4 5; do
  if [[ -e "$RES_DIR/Fold_${fold}/model.pth.tar" ]]; then
    echo "Checkpoint already exists: $RES_DIR/Fold_${fold}/model.pth.tar" >&2
    exit 2
  fi

  python -u train_semantic.py \
    --model mm_utae \
    --fold "${fold}" \
    --sar-orbits both \
    --fusion-mode concat \
    --batch_size 4 \
    --accumulation_steps 1 \
    --epochs 100 \
    --lr 0.001 \
    --rdm_seed 1 \
    --num_classes 20 \
    --ignore_index 19 \
    --ref_date 2018-09-01 \
    --decoder-norm batch \
    --mm-utae-head "$HEAD" \
    --mm-utae-aux-weight "$AUX_WEIGHT" \
    --final-bn-init "$FINAL_BN_INIT" \
    --temporal-dropout-s2 0.4 \
    --temporal-dropout-s1a 0.2 \
    --temporal-dropout-s1d 0.2 \
    --dataset_folder /share/home/xyxiao/dataset/expduoyuan/PASTIS-R \
    --res_dir "$RES_DIR"
done
