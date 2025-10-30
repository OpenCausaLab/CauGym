# CAN LLMS SERVE AS CAUSAL INFERENCE AGENTS? A STUDY ON POST-TRAINING METHODS

This project introduces DeepCausa, a comprehensive dataset and benchmark designed to systematically investigate whether targeted Post-Training methods can transform Large Language Models (LLMs) into highly effective and robust Causal Inference Agents.

Our research provides the first systematic evidence that online RL enables a smaller-scale LLM (14B) to outperform larger models and exhibit superior generalization and robustness across complex causal tasks.


## DeepCausa Dataset Introduction

The DeepCausa dataset is composed of training datasets for reinforcing causal abilities and a testing datasets for comprehensive evaluation. 

1. Training Dataset 

The training datasetset is designed to systematically enhance LLM causal reasoning abilities, covering seven core causal inference tasks across interventional and counterfactual domains:

| **Causal Task** | **Abbreviation** | **Description** | 
 | ----- | ----- | ----- | 
| Average Treatment Effect | **ATE** | Assesses the average expected difference in outcomes had everyone received treatment versus no treatment. | 
| Controlled Direct Effect | **CDE** | Assesses the direct effect of the treatment on the outcome, while holding the mediator variable at a specific level. | 
| Effect of the Treatment on the Treated | **ETT** | Assesses the expected difference in outcomes for the subpopulation that actually received the treatment. | 
| Natural Direct Effect | **NDE** | Assesses the direct effect of the treatment on the outcome, with the mediator set to the value it would naturally take in the absence of treatment. | 
| Natural Indirect Effect | **NIE** | Assesses the effect on the outcome transmitted solely through the mediator, when the treatment is changed. | 
| Probability of Necessity | **PN** | Assesses the probability that the absence of treatment was a necessary condition for the outcome to be absent, given that treatment was received and the outcome occurred. | 
| Probability of Sufficiency | **PS** | Assesses the probability that receiving treatment was a sufficient condition for the outcome to occur, given that no treatment was received and the outcome did not occur. |


2. Testing Dataset 
We construct five unique testing datasetsets to evaluate the agents across three dimensions: Generalization, Knowledge Internalization, and Robustness:

| **Evaluation Dimension** | **Test Set Name** | **Purpose** | **Example (Partial)** | 
| ----- | ----- | ----- | ----- | 
| **Generalization** | `DeepCausa-rephrased` | Tests whether the model understands the semantic meaning of the question, rather than merely memorizing training phraseology. | | 
| **Internalization** | `DeepCausa-omitted` | Omits the explicit task instruction, testing the model's ability to independently recognize and apply causal theorems. | | 
| **Internalization** | `DeepCausa-deconfounding` | Contains problems solvable only by correctly applying the **Backdoor Criterion** to remove spurious correlations. | | 
| **Robustness** | `DeepCausa-redundant` | Introduces extraneous (correct but irrelevant) information to test the model's ability to disregard irrelevant interference. | | 
| **Robustness** | `DeepCausa-insufficient` | Removes necessary conditional probability information to test the model's ability to identify missing data and output `LACK_CONDITION`. | |

## Quick Start for DeepCausa Eval
 
### installation
```
git clone https://github.com/OpenCausaLab/DeepCausa.git
cd DeepCausa
cd eval
conda create -n DeepCausa-eval python=3.10 -y
conda activate DeepCausa-eval
pip install vllm
pip install datasets
```

### Run Models and Save Result
Using Huggingface Model
```
bash scripts/model_generate.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B 5 8192
```
Using Local Model
```
bash scripts/model_generate_local.sh /path/to/my/model Test_model 5 8192
```

### Evaluate the Result

Using Huggingface Model
```
bash scripts/model_evaluate.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```
Using Local Model
```
bash scripts/model_evaluate.sh Test_model

## Citation

If you use the DeepCausa dataset or reference our research on post-training methods, please cite our paper:
```
```