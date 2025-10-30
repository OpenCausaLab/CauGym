#!/bin/bash

MODEL_NAME="$1"

# --- 验证输入 ---
# 检查用户是否提供了参数
if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is required." >&2
    echo "Usage: $0 <MODEL_NAME>"
    exit 1
fi

python sample_evaluate.py --dataset outputs/$MODEL_NAME/base.jsonl --model_name $MODEL_NAME
python sample_evaluate.py --dataset outputs/$MODEL_NAME/deconfounding.jsonl --model_name $MODEL_NAME
python sample_insuff_evaluate.py --dataset outputs/$MODEL_NAME/insufficient.jsonl --model_name $MODEL_NAME
python sample_evaluate.py --dataset outputs/$MODEL_NAME/omitted.jsonl --model_name $MODEL_NAME
python sample_evaluate.py --dataset outputs/$MODEL_NAME/redundant.jsonl --model_name $MODEL_NAME
python sample_evaluate.py --dataset outputs/$MODEL_NAME/rephrased.jsonl --model_name $MODEL_NAME

