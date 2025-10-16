#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
HPARAMS_DIR="hparams/R-ROME"
QUESTIONS_BASE_DIR="data/questions/hallucination_final"
EDIT_METHOD="R-ROME"
PYTHON_SCRIPT="edit_sequential_seperate.py" # Relative to the code/ directory
DEVICE_EDIT=0
DEVICE_EVAL=0

# --- Main Logic ---
echo "Starting automated testing for R-ROME..."

# Change to the code directory to resolve relative paths correctly
cd code

# Loop through each model's hparams file
for hparam_file in "$HPARAMS_DIR"/*.yaml; do
    model_name_base=$(basename "$hparam_file" .yaml)
    
    # Map hparam model name to question directory name
    question_dir=""
    case "$model_name_base" in
        "gemma-2b")
            question_dir="gemma_1.1_2b_it"
            ;;
        "gpt-j-6B")
            question_dir="gpt_j_6b"
            ;;
        "llama2-7b")
            question_dir="llama_2_7b_chat_hf"
            ;;
        "llama3-8b")
            question_dir="meta_llama_3_8b_instruct"
            ;;
        "mistral-7b")
            question_dir="mistral_7b_instruct_v0.3"
            ;;
        *)
            echo "Warning: No question directory mapping found for $model_name_base. Skipping."
            continue
            ;;
    esac

    # Adjust the path to be relative to the root, as we are now in the code/ dir
    MODEL_QUESTIONS_DIR="../$QUESTIONS_BASE_DIR/$question_dir"
    
    if [ ! -d "$MODEL_QUESTIONS_DIR" ]; then
        echo "Warning: Question directory not found for $model_name_base at $MODEL_QUESTIONS_DIR. Skipping."
        continue
    fi

    echo "--------------------------------------------------"
    echo "Processing Model: $model_name_base"
    echo "--------------------------------------------------"

    # Loop through each topic file for the current model
    for topic_file in "$MODEL_QUESTIONS_DIR"/*.csv; do
        topic_name=$(basename "$topic_file" .csv)
        echo "  - Topic: $topic_name"

        # --- Step 1: Generate Mode ---
        echo "    > Running generation..."
        python3 "$PYTHON_SCRIPT" \
            --model_name="$model_name_base" \
            --edit_method="$EDIT_METHOD" \
            --topics "$topic_file" \
            --device_edit="$DEVICE_EDIT" \
            --generate_only \
            --overwrite_result || {
                echo "    > ERROR: Generation failed for $model_name_base on topic $topic_name. Skipping evaluation."
                continue
            }

        # --- Step 2: Evaluate Mode ---
        echo "    > Running evaluation..."
        python3 "$PYTHON_SCRIPT" \
            --model_name="$model_name_base" \
            --edit_method="$EDIT_METHOD" \
            --topics "$topic_file" \
            --device_eval="$DEVICE_EVAL" \
            --evaluate_only \
            --overwrite_result || {
                echo "    > ERROR: Evaluation failed for $model_name_base on topic $topic_name."
            }
        
        echo "    > Done."
    done
done

echo "--------------------------------------------------"
echo "All tests completed."
echo "--------------------------------------------------"