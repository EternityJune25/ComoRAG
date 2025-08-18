import os
import json
import glob
from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
import numpy as np
import argparse

def get_logger(name):
    
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

def normalize_answer(s):
    
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

class QAExactMatch:
    metric_name: str = "qa_exact_match"

    def __init__(self):
        self.logger = get_logger(__name__)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate exact match (EM) scores

        Args:
            gold_answers: List of standard answers, each element is a list of answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate multiple standard answers

        Returns:
            Tuple containing: average EM score dictionary, list of EM scores for each sample
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same"

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results

class QAF1Score:
    
    metric_name: str = "qa_f1_score"

    def __init__(self):
        self.logger = get_logger(__name__)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculate F1 scores

        Args:
            gold_answers: List of standard answers, each element is a list of answers
            predicted_answers: List of predicted answers
            aggregation_fn: Function to aggregate multiple standard answers

        Returns:
            Tuple containing: average F1 score dictionary, list of F1 scores for each sample
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same"

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens) if predicted_tokens else 0.0
            recall = 1.0 * num_same / len(gold_tokens) if gold_tokens else 0.0
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results

def find_and_merge_results(root_path: str) -> List[Dict]:
    """
    Find all json files containing 'results' and merge them, returning the merged result list

    Args:
        root_path: Root directory path

    Returns:
        Merged result list
    """
    logger = get_logger(__name__)
    all_results = []

    # Find all json files containing 'results' (recursive)
    pattern = os.path.join(root_path, "**/*results*.json")
    json_files = glob.glob(pattern, recursive=True)

    logger.info(f"Found {len(json_files)} results-related json files")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                    logger.info(f"Successfully loaded {json_file}, containing {len(data)} records")
                else:
                    logger.warning(f"File {json_file} is not in list format, skipping")
        except Exception as e:
            logger.error(f"Error loading file {json_file}: {e}")

    logger.info(f"Total merged records: {len(all_results)}")
    return all_results

def extract_final_answer(output_text: str) -> str:
    if not output_text:
        return ""
    
    # Find ### Final Answer marker
    final_answer_marker = "### Final Answer"
    # Find the position of the last Final Answer marker
    marker_pos = output_text.rfind(final_answer_marker)
    
    if marker_pos == -1:
        # If marker not found, return the entire text (compatibility handling)
        return output_text.strip()
    
    # Extract content after the marker
    answer_start = marker_pos + len(final_answer_marker)
    final_answer = output_text[answer_start:].strip()
    
    # Remove possible leading newlines
    if final_answer.startswith('\n'):
        final_answer = final_answer[1:].strip()

    return final_answer

def extract_answers_from_results(results: List[Dict]) -> Tuple[List[List[str]], List[str]]:
    logger = get_logger(__name__)
    gold_answers = []
    predicted_answers = []
    
    for item in results:
        # Extract standard answers
        if 'golden_answers' in item:
            gold_answers.append(item['golden_answers'])
        elif 'gold_answers' in item:
            gold_answers.append(item['gold_answers'])
        else:
            logger.warning(f"Standard answer field not found, index: {item.get('idx', 'unknown')}")
            gold_answers.append([])
        
        # Extract predicted answers - specially handle Final Answer in output field
        predicted_answer = ""
        if 'output' in item:
            # Extract content after ### Final Answer from output
            predicted_answer = extract_final_answer(item['output'])
        elif 'prediction' in item:
            predicted_answer = item['prediction']
        elif 'answer' in item:
            predicted_answer = item['answer']
        else:
            logger.warning(f"Predicted answer field not found, index: {item.get('idx', 'unknown')}")
        
        predicted_answers.append(predicted_answer)
    
    return gold_answers, predicted_answers

def save_detailed_results(results_path: str, pooled_results: Dict, example_results: List[Dict], original_data: List[Dict], extracted_answers: List[str]):
    summary_path = os.path.join(results_path, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(pooled_results, f, ensure_ascii=False, indent=2)
    
    detailed_results = []
    for i, (orig, eval_result, extracted_answer) in enumerate(zip(original_data, example_results, extracted_answers)):
        detailed_item = orig.copy()
        detailed_item['extracted_answer'] = extracted_answer
        detailed_item.update(eval_result)
        detailed_results.append(detailed_item)
    
    detailed_path = os.path.join(results_path, "detailed_evaluation_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    answer_extraction_results = []
    for i, (orig, extracted_answer) in enumerate(zip(original_data, extracted_answers)):
        answer_extraction_results.append({
            "idx": orig.get("idx", i),
            "question": orig.get("question", ""),
            "original_output": orig.get("output",orig.get("answer","")),
            "extracted_answer": extracted_answer,
            "golden_answers": orig.get("golden_answers", orig.get("gold_answers", []))
        })
    
    extraction_path = os.path.join(results_path, "answer_extraction_results.json")
    with open(extraction_path, 'w', encoding='utf-8') as f:
        json.dump(answer_extraction_results, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="QA Evaluation Tool - Automatically merge results and calculate EM and F1 metrics")
    parser.add_argument("root_path", help="Root directory path containing results.json files")
    parser.add_argument("--output", "-o", help="Output directory path for results", default=None)
    
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    
    if not os.path.exists(args.root_path):
        logger.error(f"Path does not exist: {args.root_path}")
        return
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.root_path, "evaluation_results")
    
    os.makedirs(output_path, exist_ok=True)
    
    logger.info("Starting QA evaluation process...")
    
    logger.info("Step 1: Merging all results.json files")
    merged_results = find_and_merge_results(args.root_path)
    
    if not merged_results:
        logger.error("No valid results.json files found or files are empty")
        return
    
    merged_path = os.path.join(output_path, "merged_results.json")
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Merged results saved to: {merged_path}")
    
    logger.info("Step 2: Extracting standard answers and predicted answers")
    gold_answers, predicted_answers = extract_answers_from_results(merged_results)
    
    if len(gold_answers) != len(predicted_answers):
        logger.error(f"Number of standard answers ({len(gold_answers)}) does not match number of predicted answers ({len(predicted_answers)})")
        return
    
    logger.info(f"Successfully extracted {len(gold_answers)} answer pairs")
    
    logger.info("Step 3: Calculating evaluation metrics")
    
    em_metric = QAExactMatch()
    em_pooled, em_examples = em_metric.calculate_metric_scores(gold_answers, predicted_answers)
    
    f1_metric = QAF1Score()
    f1_pooled, f1_examples = f1_metric.calculate_metric_scores(gold_answers, predicted_answers)
    
    pooled_results = {**em_pooled, **f1_pooled}
    example_results = []
    for em_ex, f1_ex in zip(em_examples, f1_examples):
        example_results.append({**em_ex, **f1_ex})
    
    logger.info("Step 4: Saving evaluation results")
    save_detailed_results(output_path, pooled_results, example_results, merged_results, predicted_answers)
    
    logger.info("="*50)
    logger.info("Evaluation Results Summary:")
    logger.info(f"Total samples: {len(gold_answers)}")
    logger.info(f"Exact Match (EM): {pooled_results['ExactMatch']:.4f}")
    logger.info(f"F1 Score: {pooled_results['F1']:.4f}")
    logger.info("="*50)
    logger.info(f"Detailed results saved to directory: {output_path}")
    logger.info("Files include:")
    logger.info("- merged_results.json: Merged original data")
    logger.info("- evaluation_summary.json: Evaluation metrics summary")
    logger.info("- detailed_evaluation_results.json: Detailed results with evaluation scores for each sample")
    logger.info("- answer_extraction_results.json: Answer extraction results (for verifying extraction correctness)")

if __name__ == "__main__":
    main()
