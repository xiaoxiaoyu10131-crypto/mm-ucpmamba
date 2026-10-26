#!/bin/bash
#SBATCH --job-name=mm-uptae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mm-uptae.log
#SBATCH --error=mm-uptae.err

set -e

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate uvmrnn

cd /share/home/xyxiao/xxy/expduoyuan/mm-ucpmamba1016/utae

for fold in 2 3
do
  python -u train_semantic.py \
    --model mm-uptae \
    --fold "$fold" \
    --sar-orbits both \
    --fusion-mode concat \
    --encoder_widths "[64,64,64,128]" \
    --decoder_widths "[32,32,64,128]" \
    --out_conv "[32,20]" \
    --encoder_norm group \
    --decoder-norm group \
    --decoder-groups 8 \
    --final-bn-init ones \
    --mid-num-bins 24 \
    --mid-start-day -200 \
    --mid-end-day 600 \
    --mid-low-memory \
    --async-radius 1 \
    --async-dim 16 \
    --async-trend-min-span 2 \
    --async-residual-init 0 \
    --async-dynamic \
    --async-interaction \
    --n_head 16 \
    --d_model 256 \
    --d_k 4 \
    --ref_date 2018-09-01 \
    --batch_size 4 \
    --accumulation_steps 1 \
    --epochs 100 \
    --lr 0.001 \
    --rdm_seed 1 \
    --temporal-dropout-s2 0.4 \
    --temporal-dropout-s1a 0.2 \
    --temporal-dropout-s1d 0.2 \
    --modality-dropout 0 \
    --num_classes 20 \
    --ignore_index 19 \
    --dataset_folder /share/home/xyxiao/dataset/expduoyuan/PASTIS-R \
    --res_dir /share/home/xyxiao/dataset/expduoyuan/mm-uptae \
    > "mm-uptae-fold_${fold}.log" 2>&1
done
