from transformers import pipeline
import csv
from datetime import datetime
from huggingface_hub import login, logout
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm  # for progress bar
login(token="YourToken")

def get_model_tokenizer(model_path ):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def get_model_tokenizer_local(model_path ):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16, local_files_only=True)
    return model, tokenizer

# Load all models

#base_pipe = pipeline("text-generation", model=base_model, tokenizer =base_tokenizer,  device_map="auto", dtype=torch.bfloat16)
ds_test = load_dataset('csv', data_files={'test':'text_only_test_data.csv'})

test = ds_test["test"]

# Optionally add prefix on the validation set for instruction tuning
def add_prefix(example):
    prefix = "Summarize the following document: "
    example["document"] = prefix + example["document"]
    return example

test = test.map(add_prefix)
print(test.column_names)
# Test documents
test_docs = test["document"]
print(len(test_docs))
def process_all_documents(test_docs, generator, output_filename="summaries.csv"):
    """
    Process all test documents and save results to CSV
    """
    all_results = []
    
    # Process each document with progress bar
    for i, doc in enumerate(tqdm(test_docs, desc="Processing documents")):
        try:
            # Generate summary
            results = generator(
                doc, 
                max_new_tokens=3000, 
                return_full_text=False,
                # Add anti-repetition parameters
                repetition_penalty=1.2,
                do_sample=True,
                temperature=0.3
            )
            
            # Extract the generated text
            summary = results[0]['generated_text']
            
            # Store results
            result_data = {
                'doc_id': i,
                'original_text': doc,
                'summary': summary,
                'original_length': len(doc),
                'summary_length': len(summary)
            }
            
            all_results.append(result_data)
            
            # Print progress every 10 documents
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_docs)} documents")
                print(summary)
                
        except Exception as e:
            print(f"Error processing document {i}: {e}")
            # Still save the error
            all_results.append({
                'doc_id': i,
                'original_text': doc,
                'summary': f"ERROR: {str(e)}",
                'original_length': len(doc),
                'summary_length': 0
            })
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"✓ Successfully processed {len(all_results)} documents")
    print(f"✓ Results saved to {output_filename}")
    
    return df

print("Base Command")
generator_Command_base = pipeline("text-generation", model="CohereLabs/c4ai-command-r7b-12-2024")
process_all_documents(test_docs, generator_Command_base, "Results_Base/Command/prediction_summaries.csv")

print("Lora Command")
generator_Command_lora = pipeline("text-generation", model="Results_Lora/Command/my_trained_Command_model")
process_all_documents(test_docs, generator_Command_lora, "Results_Lora/Command/prediction_summaries2.csv")

print("LoraFA Command")
generator_Command_lora_fa = pipeline("text-generation", model="Results_LoraFA/Command/my_trained__command_model")
process_all_documents(test_docs, generator_Command_lora_fa, "Results_LoraFA/Command/prediction_summaries.csv")

print("AdaLora Command")
generator_Command_ada_lora = pipeline("text-generation", model="Results_AdaLora/Command/my_trained_command_model")
process_all_documents(test_docs, generator_Command_ada_lora, "Results_AdaLora/Command/prediction_summaries2.csv")
