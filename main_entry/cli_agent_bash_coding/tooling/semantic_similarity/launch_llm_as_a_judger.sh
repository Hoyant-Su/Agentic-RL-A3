#!/bin/bash

TP_SIZE=$(nvidia-smi -L | wc -l)
model_path="Qwen/Qwen3-8B"

python -m sglang.launch_server \
  --model-path $model_path \
  --host 0.0.0.0 \
  --port 30003 \
  --mem-fraction-static 0.03 \
  --tp-size $TP_SIZE \
  --cuda-graph-max-bs 16
