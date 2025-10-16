import os
import gc
import json
import time
import torch
import argparse
import pandas as pd
from hallucination_editor_seperate import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, GraceHyperParams, KNHyperParams, R_ROMEHyperParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama3-8b')
    parser.add_argument('--hparams_dir', default='./hparams', type=str)
    parser.add_argument('--results_dir', default='../results/separate/hallu_edit_sequential', type=str)
    parser.add_argument('--edit_method', required=True, help='Edit method to use')
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    parser.add_argument('--device_eval', default=0, help='device of the local evaluation model')
    parser.add_argument('--topics', nargs='+', required=True, help='List of topic CSV files to process sequentially.')
    parser.add_argument('--eval_every_n_steps', default=1, type=int, help='Evaluate every N steps during sequential editing.')
    parser.add_argument('--output_model_path', default=None, type=str, help='Path to save the final edited model.')
    parser.add_argument('--resume_from_model', default=None, type=str, help='Path to a previously edited model to resume from.')
    parser.add_argument('--overwrite_result', default=False, action='store_true', help='Overwrite the existing result file')
    parser.add_argument('--model_eval', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='model id of the local evaluation model')
    parser.add_argument('--generate_only', action='store_true', help='Only run generation and save results, skip evaluation.')
    parser.add_argument('--evaluate_only', action='store_true', help='Only run evaluation on existing generation results.')
    args = parser.parse_args()
    start_time = time.time()

    if args.generate_only and args.evaluate_only:
        raise ValueError("Cannot use --generate_only and --evaluate_only at the same time.")

    editing_method = args.edit_method
    if editing_method in ['FT-M', 'FT-L']:
        editing_hparams = FTHyperParams
    elif editing_method == 'ICL':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif editing_method == 'R-ROME':
        editing_hparams = R_ROMEHyperParams
    else:
        raise NotImplementedError

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}/{editing_method}/{args.model_name}')
    model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()
    
    results_dir = f'{args.results_dir}/{model_id_format}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    topics_name_str = "_".join([os.path.basename(t).split('.')[0] for t in args.topics])
    generation_file_name = f'{topics_name_str}_{editing_method}_generations.json'
    results_file = f'{topics_name_str}_{editing_method}.json'

    if args.evaluate_only:
        print(f"Evaluation only mode for {topics_name_str} with {editing_method}...")
        generation_file_path = f'{results_dir}/{generation_file_name}'
        if not os.path.exists(generation_file_path):
            raise FileNotFoundError(f"Generation file not found: {generation_file_path}. Please run with --generate_only first.")
        
        with open(generation_file_path, 'r') as f:
            generation_results = json.load(f)
        
        editor = BaseEditor(hparams, load_model=False)
        metrics = editor.evaluate_from_generations(
            generation_results,
            eval_model_id=args.model_eval,
            device_eval=f'cuda:{args.device_eval}',
            summary_metrics=True
        )
        json.dump(metrics, open(f'{results_dir}/{results_file}', 'w'), indent=4)
        exit()

    print(f'\nModel: {model_id_format}, Editing sequentially with {editing_method}...\n')
    
    if os.path.exists(f'{results_dir}/{results_file}') and not args.overwrite_result:
        print(f'Result {results_file} already exists, skipping.\n')
        exit()

    # --- Step 1: Initialize all data containers ---
    all_prompts = []
    all_targets = []
    all_subjects = []
    all_rephrase_prompts = []
    all_locality_inputs = {'locality': {'prompt': [], 'ground_truth': []}}
    all_portability_inputs = {} # Will be populated dynamically if columns exist
    all_yes_questions = {'yes': {'prompt': [], 'ground_truth': []}}
    all_no_questions = {'no': {'prompt': [], 'ground_truth': []}}
    all_reversed_relation_questions = {'reversed': {'prompt': [], 'ground_truth': []}}
    all_multiple_choice_questions = {'multiple_choice': {'prompt': [], 'ground_truth': []}}
    all_questions_2hop = {'2hop': {'prompt': [], 'ground_truth': []}}
    all_questions_3hop = {'3hop': {'prompt': [], 'ground_truth': []}}
    all_questions_4hop = {'4hop': {'prompt': [], 'ground_truth': []}}
    all_questions_5hop = {'5hop': {'prompt': [], 'ground_truth': []}}
    all_questions_6hop = {'6hop': {'prompt': [], 'ground_truth': []}}

    # --- Step 2: Load all data from CSV ---
    for topic_file in args.topics:
        df = pd.read_csv(topic_file)
        all_prompts.extend(df['question'].tolist())
        all_targets.extend(df['object'].tolist())
        all_subjects.extend(df['subject'].tolist())
        all_rephrase_prompts.extend(df['paraphrased_question'].tolist())
        all_locality_inputs['locality']['prompt'].extend(df['locality_question'].tolist())
        all_locality_inputs['locality']['ground_truth'].extend(df['subject'].tolist())

        # Load robustness data
        all_yes_questions['yes']['prompt'].extend(df['yes_question'].tolist())
        all_yes_questions['yes']['ground_truth'].extend(['Yes'] * len(df))
        all_no_questions['no']['prompt'].extend(df['no_question'].tolist())
        all_no_questions['no']['ground_truth'].extend(['No'] * len(df))

        # Load reversed relation data
        if 'reversed_relation_question' in df.columns:
            all_reversed_relation_questions['reversed']['prompt'].extend(df['reversed_relation_question'].tolist())
            all_reversed_relation_questions['reversed']['ground_truth'].extend(df['subject'].tolist())

        # Load multiple choice data
        if 'multiple_choice_with_letters' in df.columns and 'multiple_choice_labels' in df.columns:
            all_multiple_choice_questions['multiple_choice']['prompt'].extend(df['multiple_choice_with_letters'].tolist())
            all_multiple_choice_questions['multiple_choice']['ground_truth'].extend(df['multiple_choice_labels'].tolist())

        # Load multi-hop questions data
        hop_questions = {
            'questions_2hop': all_questions_2hop,
            'questions_3hop': all_questions_3hop,
            'questions_4hop': all_questions_4hop,
            'questions_5hop': all_questions_5hop,
            'questions_6hop': all_questions_6hop,
        }
        for i in range(2, 7):
            hop_key = f'questions_{i}hop'
            prompt_col = f'question_{i}hop'
            answer_col = f'answer_{i}hop'
            if prompt_col in df.columns and answer_col in df.columns:
                hop_questions[hop_key][f'{i}hop']['prompt'].extend(df[prompt_col].tolist())
                hop_questions[hop_key][f'{i}hop']['ground_truth'].extend(df[answer_col].tolist())

    hparams.device = args.device_edit
    editor = BaseEditor.from_hparams(hparams)

    if args.resume_from_model:
        print(f"Resuming from model at {args.resume_from_model}")
        editor.load_model(args.resume_from_model)

    # --- Step 3: Pass all loaded data to the editor ---
    edit_kwargs = {
        'prompts': all_prompts,
        'target_new': all_targets,
        'subject': all_subjects,
        'rephrase_prompts': all_rephrase_prompts,
        'locality_inputs': all_locality_inputs,
        'yes_questions': all_yes_questions,
        'no_questions': all_no_questions,
        'reversed_relation_questions': all_reversed_relation_questions,
        'multiple_choice_questions': all_multiple_choice_questions,
        'questions_2hop': all_questions_2hop,
        'questions_3hop': all_questions_3hop,
        'questions_4hop': all_questions_4hop,
        'questions_5hop': all_questions_5hop,
        'questions_6hop': all_questions_6hop,
        'sequential_edit': True,
        'eval_every_n_steps': args.eval_every_n_steps,
        'summary_metrics': True,
        'eval_model_id': args.model_eval,
        'device_eval': f'cuda:{args.device_eval}',
        'generate_only': args.generate_only
    }

    results, edited_model, _ = editor.edit(**edit_kwargs)
    
    if args.generate_only:
        print(f"Saving generation results to {results_dir}/{generation_file_name}")
        json.dump(results, open(f'{results_dir}/{generation_file_name}', 'w'), indent=4)
    else:
        print(f"Saving evaluation metrics to {results_dir}/{results_file}")
        json.dump(results, open(f'{results_dir}/{results_file}', 'w'), indent=4)

    if args.output_model_path:
        print(f"Saving final model to {args.output_model_path}")
        editor.save_model(args.output_model_path)
    
    print(f'\nSequential editing finished for Model: {model_id_format} with {editing_method}')
    del edited_model
    del editor
    gc.collect()
    torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 60 
    print(f'\nOverall running time for edit_sequential_seperate.py: {total_time:.2f} minutes')