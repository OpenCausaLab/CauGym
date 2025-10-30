#!/bin/bash
MODEL_PATH="/mnt/shared-storage-user/chenjunqi/LLM/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_NAME="Test_model"
SAMPLE_NUM=1
MAX_TOKENS=8192

python sample_generate.py --model_name $MODEL_NAME --model_path $MODEL_PATH --test_name base --sample_num $SAMPLE_NUM --max_tokens $MAX_TOKENS
python sample_generate.py --model_name $MODEL_NAME --model_path $MODEL_PATH --test_name deconfounding --sample_num $SAMPLE_NUM --max_tokens $MAX_TOKENS
python sample_generate.py --model_name $MODEL_NAME --model_path $MODEL_PATH --test_name insufficient --instruction_file instructions/insuff_ins.txt --sample_num $SAMPLE_NUM --max_tokens $MAX_TOKENS
python sample_generate.py --model_name $MODEL_NAME --model_path $MODEL_PATH --test_name omitted --sample_num $SAMPLE_NUM --max_tokens $MAX_TOKENS
python sample_generate.py --model_name $MODEL_NAME --model_path $MODEL_PATH --test_name redundant --sample_num $SAMPLE_NUM --max_tokens $MAX_TOKENS
python sample_generate.py --model_name $MODEL_NAME --model_path $MODEL_PATH --test_name rephrased --sample_num $SAMPLE_NUM --max_tokens $MAX_TOKENS


