#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"
DATASET_FOLDER="${DATASET_FOLDER:-/share/home/xyxiao/dataset/expduoyuan/PASTIS-R}"
SEED="${SEED:-1}"
EXPERIMENT="${EXPERIMENT:-mm_ucpmamba_async_1016}"
RES_DIR="${RES_DIR:-./results/${EXPERIMENT}/seed_${SEED}}"
FOLDS="${FOLDS:-1 2 3 4 5}"
[[ -d "$DATASET_FOLDER" ]] || { echo "数据目录不存在：$DATASET_FOLDER" >&2; exit 2; }
read -r -a folds <<< "$FOLDS"
[[ ${#folds[@]} -gt 0 ]] || { echo 'FOLDS 不能为空' >&2; exit 2; }
seen=' '
for fold in "${folds[@]}"; do
  [[ "$fold" =~ ^[1-5]$ && "$seen" != *" $fold "* ]] || { echo '折编号错误或重复' >&2; exit 2; }
  seen+="$fold "
  [[ ! -e "$RES_DIR/Fold_${fold}" ]] || { echo "拒绝覆盖已有实验：$RES_DIR/Fold_${fold}" >&2; exit 2; }
done
# conf.json 由原训练入口写入；一次新实验使用独立目录。
[[ ! -e "$RES_DIR/conf.json" ]] || { echo '请使用新的 RES_DIR，避免混合配置' >&2; exit 2; }
mkdir -p "$RES_DIR/logs"
for fold in "${folds[@]}"; do
  "$PYTHON" -u run_mm_ucpmamba_async.py \
    --fold "$fold" --rdm_seed "$SEED" \
    --dataset_folder "$DATASET_FOLDER" --res_dir "$RES_DIR" \
    2>&1 | tee "$RES_DIR/logs/fold_${fold}.log"
done
if [[ ${#folds[@]} -eq 5 ]]; then
  "$PYTHON" -m src.mm_async_results "$RES_DIR"
fi
