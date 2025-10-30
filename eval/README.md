# Quick Start
## installation
```
git clone https://github.com/whoiscjq/DeepCausa-eval.git
conda create -n DeepCausa-eval python=3.10 -y
conda activate DeepCausa-eval
pip install vllm
pip install datasets
```

## Run Models and Save Result
Using Huggingface Model
```
bash scripts/model_generate.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B 5 8192
```
Using Local Model
```
bash scripts/model_generate_local.sh /path/to/my/model Test_model 5 8192
```

## Evaluate the Result

Using Huggingface Model
```
bash scripts/model_evaluate.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```
Using Local Model
```
bash scripts/model_evaluate.sh Test_model
```
