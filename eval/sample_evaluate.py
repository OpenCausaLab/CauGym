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


def extract_prob(model_response):  #get last prob
    if not model_response:
        return None
    model_response=re.sub(r'\s+','',model_response)
    combined_pattern = r'\{\"(?:ANSWER|PROB)\"\:.*?\}' 
    matches = re.findall(combined_pattern, model_response, re.DOTALL)

    prob_str_value = None
    prob_str_values=[]
    # Iterate through all found matches to find the last valid one
    for match_str in matches:
        processed_match_str = match_str.lower()
        # Handle extra braces if present (based on your original logic)
        try:
            inner_json_obj = json.loads(processed_match_str)
            # Prioritize 'prob' key, then 'answer' for numerical value
            tmp_str_value = inner_json_obj.get("prob")
            # if tmp_str_value is None:
            #     # If 'prob' not found, check 'answer' (your PATTERN_DUAL suggests 'ANSWER' might be a key)
            #     tmp_str_value = inner_json_obj.get("answer")

            if tmp_str_value is not None:
                prob_str_values.append(str(tmp_str_value))
                #print(prob_str_value)# Store the last found valid probability string
        
        except json.JSONDecodeError:
            continue # If parsing fails, move to the next match

    if not prob_str_values:
        return None
    return  prob_str_values


def get_extracted_float_answer(generated_response_str: str) -> float | None:
    """
    Extracts a float probability from a generated response, running Python code if present.
    Returns the float value or None if extraction or conversion fails.
    """

    extracted_strs= extract_prob(generated_response_str)
    outputs=[]
    if not extracted_strs:
        return outputs
    for extracted_str in extracted_strs:
        try:
            if extracted_str is not None:
                outputs.append(float(extracted_str))
        except ValueError:
            continue # Return None if conversion to float fails
    return outputs

def check_single_response(generated_response_str: str, gt_answer_json_str: str) -> int:
    """
    Checks if a single generated code response produces the correct output.
    Returns 1 if correct, 0 otherwise.
    """
    gt_answer = extract_prob(gt_answer_json_str)
    #python_answer_float = get_extracted_float_answer(generated_response_str)
    float_gt_answer = float(gt_answer[-1])
    answer_floats=get_extracted_float_answer(generated_response_str)

    for python_answer_float in answer_floats:
        if python_answer_float is not None and float_gt_answer is not None and abs(float_gt_answer - python_answer_float) < 0.0005:
            return 1

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
    model_name = args.model_name.replace("/","_")
    
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