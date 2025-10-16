#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
QUESTIONS_BASE_DIR="data/questions/hallucination_final"
PYTHON_SCRIPT="edit_sequential_seperate.py"
DEVICE_EDIT=0
DEVICE_EVAL=0

# --- Helper function to process models and topics ---
run_tests() {
    local edit_method=$1
    local hparams_dir=$2
    local mode=$3 # "generate_only" or "evaluate_only"

    echo "=================================================="
    echo "Running $mode for Method: $edit_method"
    echo "=================================================="

    for hparam_file in "$hparams_dir"/*.yaml; do
        local model_name_base=$(basename "$hparam_file" .yaml)
        
        local question_dir=""
        case "$model_name_base" in
            "gemma-2b") question_dir="gemma_1.1_2b_it" ;;
            "gpt-j-6B") question_dir="gpt_j_6b" ;;
            "llama2-7b") question_dir="llama_2_7b_chat_hf" ;;
            "llama3-8b") question_dir="meta_llama_3_8b_instruct" ;;
            "mistral-7b") question_dir="mistral_7b_instruct_v0.3" ;;
            *)
                echo "Warning: No question directory mapping found for $model_name_base. Skipping."
                continue
                ;;
        esac

        local model_questions_dir="../$QUESTIONS_BASE_DIR/$question_dir"
        
        if [ ! -d "$model_questions_dir" ]; then
            echo "Warning: Question directory not found for $model_name_base at $model_questions_dir. Skipping."
            continue
        fi

        echo "--------------------------------------------------"
        echo "Processing Model: $model_name_base"
        echo "--------------------------------------------------"

        for topic_file in "$model_questions_dir"/*.csv; do
            local topic_name=$(basename "$topic_file" .csv)
            echo "  - Topic: $topic_name"

            local device_flag
            local error_message
            if [ "$mode" == "generate_only" ]; then
                device_flag="--device_edit=$DEVICE_EDIT"
                error_message="Generation failed"
            else
                device_flag="--device_eval=$DEVICE_EVAL"
                error_message="Evaluation failed"
            fi

            echo "    > Running $mode..."
            python3 "$PYTHON_SCRIPT" \
                --model_name="$model_name_base" \
                --edit_method="$edit_method" \
                --topics "$topic_file" \
                $device_flag \
                --$mode \
                --overwrite_result || {
                    echo "    > ERROR: $error_message for $model_name_base on topic $topic_name."
                }
            
            echo "    > Done."
        done
    done
}

# --- Main Execution Logic ---
echo "Changing to the code directory..."
cd code

# --- Phase 1: Generation ---
run_tests "R-ROME" "hparams/R-ROME" "generate_only"
run_tests "KN" "hparams/KN" "generate_only"

# --- Phase 2: Evaluation ---
run_tests "R-ROME" "hparams/R-ROME" "evaluate_only"
run_tests "KN" "hparams/KN" "evaluate_only"

echo "=================================================="
echo "All combined tests completed."
echo "=================================================="