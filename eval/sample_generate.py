import multiprocessing
import json
import os
import torch
import re # Still useful for potentially extracting code from generation, though not for execution here

from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
# --- Configuration ---
# Your initial dataset (list of dictionaries)
# data_list = [...] # Keep your data_list here, as per your original script

# 5. vLLM Sampling Parameters

def main_processing(data_list, vllm_model_name, tokenizer_name, output_file_path,instruction,prompt_key):
    # --- Part 1: Prepare dataset and format prompts ---
    print("Step 1: Preparing dataset and formatting prompts...")
    hf_dataset = Dataset.from_list(data_list)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        return

    if tokenizer.chat_template is None:
        if hasattr(tokenizer, 'default_chat_template') and tokenizer.default_chat_template is not None:
            tokenizer.chat_template = tokenizer.default_chat_template
        else:
            print(f"Tokenizer '{tokenizer_name}' does not have a defined chat template. Exiting.")
            return

    def format_prompt_function(example):
        base_question = example.get(prompt_key, "")
        Reason=example.get("Reason", "")
        
        if not base_question:
            return {"formatted_prompt": f"Error: '{prompt_key}' field missing or empty."}

        
        messages = [
            {"role": "user", "content": f"{base_question}\n{instruction}"}
        ]
        try:
            formatted_prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {"formatted_prompt": formatted_prompt_str}
        except Exception as e:
            return {"formatted_prompt": f"Error applying chat template: {e}"}

    processed_hf_dataset = hf_dataset.map(format_prompt_function)
    print(f"Formatted {len(processed_hf_dataset)} prompts.")

    # --- Part 2: Extract formatted prompts for vLLM ---
    print("\nStep 2: Extracting prompts for vLLM...")
    prompts_for_vllm = []
    valid_indices = []
    
    for i, item in enumerate(processed_hf_dataset):
        prompt = item["formatted_prompt"]
        if "Error:" not in prompt:
            prompts_for_vllm.append(prompt)
            valid_indices.append(i)
        else:
            print(f"Skipping item idx {item.get('idx', 'Unknown')} due to prompt formatting error: {prompt}")

    if not prompts_for_vllm:
        print("No valid prompts to send to vLLM. Exiting.")
        processed_hf_dataset.to_json(output_file_path, orient="records", lines=True, force_ascii=False)
        print(f"Dataset with formatting information (and errors) saved to {output_file_path}")
        return
    
    print(f"Extracted {len(prompts_for_vllm)} valid prompts for vLLM.")

    # --- Part 3: Initialize vLLM ---
    print("\nStep 3: Initializing vLLM...")
    try:
        llm = LLM(model=vllm_model_name,
                  tensor_parallel_size=2,
                  gpu_memory_utilization=0.9,
                  dtype=torch.bfloat16,
                 )
        print("vLLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing vLLM with model '{vllm_model_name}': {e}")
        print("Please ensure vLLM is installed, CUDA is available, and the model name is correct.")
        return

    # --- Part 4: Generate answers with vLLM ---
    print(f"\nStep 4: Generating answers for {len(prompts_for_vllm)} prompts with vLLM...")
    print(f"vllm_model_name: {vllm_model_name}")
    try:
        vllm_outputs = llm.generate(prompts_for_vllm, sampling_params)
        print("Generation complete.")
    except Exception as e:
        print(f"Error during vLLM generation: {e}")
        return

    # Extract generated text and group by original prompt index
    # vLLM returns a list of RequestOutput, each RequestOutput has a list of SequenceOutput (n samples)
    grouped_generated_answers = {}
    for i, output in enumerate(vllm_outputs):
        original_data_index = valid_indices[i] # Get the original index from valid_indices
        # Store all 'n' generated texts for this original problem
        grouped_generated_answers[original_data_index] = [seq.text.strip() for seq in output.outputs]

    # --- Part 5: Add all generated answers back to the dataset ---
    print("\nStep 5: Adding all generated answers to the dataset...")
    
    # Initialize list for storing all 'n' raw generated answers for each problem
    all_raw_generated_answers = [None] * len(processed_hf_dataset) 
    
    for i in range(len(processed_hf_dataset)):
        if i in grouped_generated_answers:
            all_raw_generated_answers[i] = grouped_generated_answers[i]
        else:
            all_raw_generated_answers[i] = ["Error: No valid generation for this prompt."] # Indicate if a prompt was skipped
            
    # Add the new column to the Hugging Face Dataset
    processed_hf_dataset_with_answers = processed_hf_dataset.add_column("all_generated_answers", all_raw_generated_answers)

    # --- Part 6: Save to disk ---
    print(f"\nStep 6: Saving the complete dataset to {output_file_path}...")
    try:
        processed_hf_dataset_with_answers.to_json(
            output_file_path,
            orient="records",
            lines=True,
            force_ascii=False
        )
        print(f"Successfully saved dataset to {output_file_path}")
    except Exception as e:
        print(f"Error saving dataset to JSONL: {e}")
        # Fallback if to_json fails
        try:
            data_to_save = processed_hf_dataset_with_answers.to_list()
            with open(output_file_path.replace(".jsonl", "_fallback.json"), "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved dataset to {output_file_path.replace('.jsonl', '_fallback.json')} as a fallback.")
        except Exception as fallback_e:
            print(f"Fallback save also failed: {fallback_e}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Test_model")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_name",type=str, default="base")
    parser.add_argument("--instruction_file",type=str, default=None)
    parser.add_argument("--sample_num",type=int, default=5)
    parser.add_argument("--prompt_key",type=str, default="problem")
    parser.add_argument("--max_tokens",type=int, default=8192)
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_tokens,
        n=args.sample_num # <--- 关键：设置为你希望用于pass@k的样本数量
    )

    if args.instruction_file is not None:
        with open(args.instruction_file,"r") as f:
            instruction=f.read()
    else:
        instruction="""Let's think step by step."""
    print(f"You instruction is {instruction}")
    model_name=args.model_name
    path_model_name=model_name
    test_name=args.test_name
    model_path=args.model_path
    output_file_path = os.path.join("outputs", path_model_name, f"{test_name}.jsonl")
    
    if model_path is not None:
        vllm_model_name=model_path
        tokenizer_name=model_path
    else:
        vllm_model_name=model_name
        tokenizer_name=model_name

    
    print(f"Initial multiprocessing start method: {multiprocessing.get_start_method(allow_none=True)}")
    test_data_path=f"test_datasets/{test_name}_dataset.json"
    
    with open(test_data_path,"r") as f:
        all_data=json.load(f)
    data_list=all_data[:]

    try:
        multiprocessing.set_start_method('spawn', force=True)
        print(f"Successfully set start method to 'spawn'. Current: {multiprocessing.get_start_method()}")
    except RuntimeError as e:
        print(f"Could not set start method (it might already be set or other issues): {e}")
        print(f"Current start method after exception: {multiprocessing.get_start_method(allow_none=True)}")

    multiprocessing.freeze_support()
    print("Multiprocessing freeze_support() called.")


        
        # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    main_processing(data_list=data_list,vllm_model_name=vllm_model_name,tokenizer_name=tokenizer_name,output_file_path=output_file_path,instruction=instruction,prompt_key=args.prompt_key)