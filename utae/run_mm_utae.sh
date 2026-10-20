#!/usr/bin/env bash
# 在已配置 PyTorch 的环境中手工运行；本脚本不自动提交 SLURM 作业。
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE="${PROFILE:-comparison}"
MODEL="${MODEL:-mm_utae}"
PYTHON="${PYTHON:-python}"
DATASET_FOLDER="${DATASET_FOLDER:-/share/home/xyxiao/dataset/expduoyuan/PASTIS-R}"
FOLDS="${FOLDS:-1 2 3 4 5}"
RES_DIR="${RES_DIR:-./results/${MODEL}_${PROFILE}_s2_s1a_s1d}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.001}"
SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

case "$MODEL" in
  mm_utae|mm_ucpmamba) ;;
  *) echo "MODEL 仅支持 mm_utae 或 mm_ucpmamba" >&2; exit 2 ;;
esac
case "$PROFILE" in
  comparison) ;;
  paper)
    if [[ "$MODEL" != mm_utae ]]; then
      echo "PROFILE=paper 仅支持 MODEL=mm_utae，避免误称论文配置" >&2
      exit 2
    fi
    ;;
  *) echo "PROFILE 仅支持 comparison 或 paper" >&2; exit 2 ;;
esac
if [[ ! -d "$DATASET_FOLDER" ]]; then
  echo "数据目录不存在：$DATASET_FOLDER" >&2
  exit 2
fi
for value in "$BATCH_SIZE" "$ACCUMULATION_STEPS" "$EPOCHS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE、ACCUMULATION_STEPS、EPOCHS 必须为正整数" >&2
    exit 2
  fi
done
if [[ ! "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "NUM_WORKERS 必须为非负整数" >&2
  exit 2
fi
read -r -a folds <<< "$FOLDS"
if [[ ${#folds[@]} -eq 0 ]]; then
  echo "FOLDS 不能为空" >&2
  exit 2
fi
seen=' '
for fold in "${folds[@]}"; do
  if [[ ! "$fold" =~ ^[1-5]$ || "$seen" == *" $fold "* ]]; then
    echo "FOLDS 应为不重复的官方测试折编号 1–5：$FOLDS" >&2
    exit 2
  fi
  seen+="$fold "
  if [[ -e "$RES_DIR/Fold_${fold}/model.pth.tar" ]]; then
    echo "已有检查点，停止以免覆盖：$RES_DIR/Fold_${fold}/model.pth.tar" >&2
    echo "请使用新的 RES_DIR，或仅选择尚未运行的 FOLDS。" >&2
    exit 2
  fi
done

profile_args=(--final-bn-init ones)
if [[ "$MODEL" == mm_utae ]]; then
  profile_args+=(--mm-utae-head legacy --mm-utae-aux-weight 0)
fi
if [[ "$PROFILE" == paper ]]; then
  profile_args=(--final-bn-init legacy --mm-utae-head logits --mm-utae-aux-weight 0.5)
fi
common_args=(
  --model "$MODEL" --dataset_folder "$DATASET_FOLDER" --res_dir "$RES_DIR"
  --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS"
  --epochs "$EPOCHS" --lr "$LR" --rdm_seed "$SEED"
  --num_workers "$NUM_WORKERS" --device "$DEVICE"
  --num_classes 20 --ignore_index 19 --ref_date 2018-09-01
  --optical-satellite S2 --sar-orbits both --fusion-mode concat
  --decoder-norm batch --modality-dropout 0
  --temporal-dropout-s2 0.4 --temporal-dropout-s1a 0.2 --temporal-dropout-s1d 0.2
)
# 续跑尚未完成的折时，禁止覆盖其他训练协议的根配置。
if [[ -f "$RES_DIR/conf.json" ]]; then
  "$PYTHON" - "$RES_DIR/conf.json" "${common_args[@]}" "${profile_args[@]}" <<'PY'
import json
import sys
from train_semantic import parser, list_args

with open(sys.argv[1], encoding="utf-8") as handle:
    saved = json.load(handle)
requested = vars(parser.parse_args(sys.argv[2:]))
for key in list_args:
    requested[key] = json.loads(requested[key])
runtime_keys = {"fold", "res_dir", "device", "num_workers", "display_step", "cache",
                "diagnostics", "diagnostic_class"}
keys = requested.keys() - runtime_keys
mismatches = [key for key in keys if saved.get(key, requested[key]) != requested[key]]
if mismatches:
    raise SystemExit("RES_DIR 已有不同配置，请使用新目录：" + ", ".join(mismatches))
PY
fi
mkdir -p "$RES_DIR/logs"
run_stamp="$(date +%Y%m%d_%H%M%S)_$$"
for fold in "${folds[@]}"; do
  log_file="$RES_DIR/logs/fold_${fold}_${run_stamp}.log"
  {
    printf 'PROFILE=%s MODEL=%s 官方测试折=%s\n' "$PROFILE" "$MODEL" "$fold"
    printf '命令：'; printf '%q ' "$PYTHON" -u train_semantic.py "${common_args[@]}" "${profile_args[@]}" --fold "$fold"; printf '\n'
    "$PYTHON" -u train_semantic.py "${common_args[@]}" "${profile_args[@]}" --fold "$fold"
  } 2>&1 | tee "$log_file"
done

# 只有本次完整执行五折才合并，防止把旧配置的折结果自动混入子集。
if [[ ${#folds[@]} -eq 5 ]]; then
  "$PYTHON" - "$RES_DIR" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from train_semantic import overall_performance

result_dir = Path(sys.argv[1])
with (result_dir / "conf.json").open(encoding="utf-8") as handle:
    config = SimpleNamespace(**json.load(handle))
config.res_dir = str(result_dir)
overall_performance(config)
PY
else
  echo "本次仅执行 $FOLDS；各折指标已保存，未自动生成五折 overall.json。"
fi
