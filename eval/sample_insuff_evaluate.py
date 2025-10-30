import json
import re
from contextlib import redirect_stdout
from io import StringIO
from tqdm import tqdm
import multiprocessing
import argparse
import math 
import os # Import os for path manipulation
import datetime # Import datetime for timestamp


def check_single_response(generated_response_str: str, gt_answer_json_str: str) -> int:
    """
    Checks if a single generated code response produces the correct output.
    Returns 1 if correct, 0 otherwise.
    """
    if gt_answer_json_str in generated_response_str:
        return 1
    else:
        return 0    

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL file containing generated data.")
    parser.add_argument("--consistency_tolerance", type=float, default=0.0002,
                        help="Tolerance for considering two numerical answers as consistent.")
    parser.add_argument("--model_name", type=str, default="unknown_model", 
                        help="Name of the model being evaluated, for file naming.")
    args = parser.parse_args()
    
    jsonl_path = args.dataset
    consistency_tolerance = args.consistency_tolerance
    model_name = args.model_name
    
    # Determine output file path
    dataset_dir = os.path.dirname(jsonl_path)
    dataset_filename_base = os.path.splitext(os.path.basename(jsonl_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{dataset_filename_base}_eval.txt"
    output_filepath = os.path.join(dataset_dir, output_filename)

    # Load data
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Initialize dictionaries to store aggregated results by problem type
    type_corrects = {}
    type_sums = {}

    print(f"Evaluating dataset: {jsonl_path} for accuracy with tolerance {consistency_tolerance}")
    print(f"Results will be saved to: {output_filepath}")

    for sample in tqdm(data, desc="Processing samples"):
        problem_type = sample.get("Type", "Unknown")
        gt_answer_json_str = sample.get("answer", "{}")
        
        all_generated_answers = sample.get("all_generated_answers", [])
        
        if not all_generated_answers:
            continue 
        
        n_generated_candidates = len(all_generated_answers)
        

        correct_candidates_count = 0 

        for i, gen_answer_str in enumerate(all_generated_answers):
            is_correct = check_single_response(gen_answer_str, gt_answer_json_str)
            if is_correct:
                correct_candidates_count += 1

        type_corrects[problem_type] = type_corrects.get(problem_type, 0.0) + correct_candidates_count#pass_k_score_for_problem
        type_sums[problem_type] = type_sums.get(problem_type, 0) + n_generated_candidates

    # --- Prepare Results for Saving ---
    results_output = StringIO()

    # Print general info to results_output
    results_output.write("="*40 + "\n")
    results_output.write("--- Evaluation Parameters and Info ---\n")
    results_output.write(f"Dataset: {jsonl_path}\n")
    results_output.write(f"Model Name: {model_name}\n")
    results_output.write(f"Consistency Tolerance: {consistency_tolerance}\n")
    results_output.write(f"Evaluation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_output.write("="*40 + "\n\n")

    # Print Pass@k Results
    results_output.write("="*30 + "\n")
    results_output.write("--- Acc Results ---\n")
    results_output.write("="*30 + "\n")
    total_sums=0
    total_corrects=0
    for key in sorted(type_corrects.keys()):
        if type_sums[key] > 0:
            avg_pass = type_corrects[key] / type_sums[key]
            total_sums+=type_sums[key]
            total_corrects+=type_corrects[key]
            results_output.write(f"Type: {key} | Acc: {avg_pass:.3f}\n")
        else:
            results_output.write(f"Type: {key} | No data for Acc\n")

    results_output.write(f"All Accuracy: {(total_corrects/total_sums):.3f};.")
    print(f"All Accuracy: {(total_corrects/total_sums):.3f};.")

    # Save results to file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(results_output.getvalue())
    
    print(f"\nEvaluation complete. Results saved to: {output_filepath}")