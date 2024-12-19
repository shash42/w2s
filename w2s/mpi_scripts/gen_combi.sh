#!/bin/bash

# Define lists
datasets=("anli-r2" "sciq" "cola" "ethics-utilitarianism" "sst2" "twitter-sentiment" "boolq")
weak_models=("Qwen/Qwen2.5-0.5B" "google/gemma-2-2b" "microsoft/phi-2" "meta-llama/Llama-3.2-1B" "HuggingFaceTB/SmolLM-1.7B")
strong_models=("meta-llama/Llama-3.1-8B" "google/gemma-2-9b" "Qwen/Qwen2.5-7B")

# Output file
output_file="combinations.txt"

# Remove existing file if it exists
rm -f $output_file

# Generate all combinations
for dataset in "${datasets[@]}"; do
    for weak_model in "${weak_models[@]}"; do
        for strong_model in "${strong_models[@]}"; do
            echo "$dataset $weak_model $strong_model" >> $output_file
        done
    done
done

echo "Generated $output_file with all combinations."