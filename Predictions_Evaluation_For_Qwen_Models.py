import pandas as pd
import evaluate
from datasets import load_dataset

# Load metrics once globally
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def evaluate_summaries_from_csv(pred_csv_path):
    """
    Evaluate summaries from CSV files containing predicted and reference summaries.

    Args:
        pred_csv_path (str): Path to CSV file with predicted summaries.
        ref_csv_path (str): Path to CSV file with reference summaries.
        summary_column (str): Name of the column containing summaries in both files.

    Returns:
        dict: Dictionary with ROUGE, BLEU, and METEOR scores.
    """
    # Read summaries from CSV
    preds_df = load_dataset('csv', data_files={'test':pred_csv_path})
    pred_docs = preds_df["test"]["summary"]

    ds_test = load_dataset('csv', data_files={'test':'text_only_test_data.csv'})
    # Test documents
    test_docs = ds_test["test"]["summary"]
    print(preds_df["test"].features)
    print(ds_test["test"].features)
    # Compute metrics
    rouge_results = rouge.compute(predictions=pred_docs, references=test_docs)
    bleu_results = bleu.compute(
        predictions=[pred for pred in pred_docs],
        references=[[ref.split()] for ref in test_docs]
    )
    meteor_results = meteor.compute(predictions=pred_docs, references=test_docs)

    results = {
        "rouge": rouge_results,
        "bleu": bleu_results,
        "meteor": meteor_results
    }
    return results

print("Qwen Base Results")
results = evaluate_summaries_from_csv("Results_Base/Qwen/prediction_summaries.csv")
print(results)

print("Qwen Lora Results")
results = evaluate_summaries_from_csv("Results_Lora/Qwen/prediction_summaries2.csv")
print(results)

print("Qwen LoraFA Results")
results = evaluate_summaries_from_csv("Results_LoraFA/Qwen/prediction_summaries2.csv")
print("Ëmpty")

print("Qwen AdaLora Results")
results = evaluate_summaries_from_csv("Results_AdaLora/Qwen/prediction_summaries2.csv")
print(results)
