#!/bin/bash

#!/bin/bash
MODEL_PATH=${1}
MODEL_NAME=${2:-"Test_model"}
SAMPLE_NUM=${3:-5}
MAX_TOKENS=${4:-8192}


echo "--- Configuration ---"
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Sample Num: $SAMPLE_NUM"
echo "Max Tokens: $MAX_TOKENS"
echo "--- Starting Generation ---"


# 2. deconfounding
python sample_generate.py --model_name "$MODEL_NAME" --model_path $MODEL_PATH --test_name deconfounding --sample_num "$SAMPLE_NUM" --max_tokens "$MAX_TOKENS"

# 1. base
python sample_generate.py --model_name "$MODEL_NAME" --model_path $MODEL_PATH --test_name base --sample_num "$SAMPLE_NUM" --max_tokens "$MAX_TOKENS"


# 3. insufficient 
python sample_generate.py --model_name "$MODEL_NAME" --model_path $MODEL_PATH --test_name insufficient --instruction_file instructions/insuff_ins.txt --sample_num "$SAMPLE_NUM" --max_tokens "$MAX_TOKENS"

# 4. omitted
python sample_generate.py --model_name "$MODEL_NAME" --model_path $MODEL_PATH --test_name omitted --sample_num "$SAMPLE_NUM" --max_tokens "$MAX_TOKENS"

# 5. redundant
python sample_generate.py --model_name "$MODEL_NAME" --model_path $MODEL_PATH --test_name redundant --sample_num "$SAMPLE_NUM" --max_tokens "$MAX_TOKENS"

# 6. rephrased
python sample_generate.py --model_name "$MODEL_NAME" --model_path $MODEL_PATH --test_name rephrased --sample_num "$SAMPLE_NUM" --max_tokens "$MAX_TOKENS"

echo "--- Generation Script Completed ---"
