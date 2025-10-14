import os
import gc
import json
import time
import torch
import argparse
import pandas as pd
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, GraceHyperParams,KNHyperParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama3-8b')
    parser.add_argument('--hparams_dir', default='./hparams', type=str)
    parser.add_argument('--results_dir', default='../results/hallu_edit_sequential', type=str)
    parser.add_argument('--edit_method', required=True, help='Edit method to use')
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    parser.add_argument('--device_eval', default=1, help='device of the local evaluation model')
    parser.add_argument('--topics', nargs='+', required=True, help='List of topic CSV files to process sequentially.')
    parser.add_argument('--eval_every_n_steps', default=1, type=int, help='Evaluate every N steps during sequential editing.')
    parser.add_argument('--output_model_path', default=None, type=str, help='Path to save the final edited model.')
    parser.add_argument('--resume_from_model', default=None, type=str, help='Path to a previously edited model to resume from.')
    parser.add_argument('--overwrite_result', default=False, action='store_true', help='Overwrite the existing result file')
    parser.add_argument('--model_eval', default='Qwen/Qwen2.5-0.5B-Instruct', help='model id of the local evaluation model')
    args = parser.parse_args()
    start_time = time.time()

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
    else:
        raise NotImplementedError

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}/{editing_method}/{args.model_name}')
    model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()

    print(f'\nModel: {model_id_format}, Editing sequentially with {editing_method}...\n')
    
    # Combine topics for result file name
    topics_name_str = "_".join([os.path.basename(t).split('.')[0] for t in args.topics])
    results_file = f'{args.results_dir}/{model_id_format}/{topics_name_str}_{editing_method}.json'

    if os.path.exists(results_file):
        print(f'Result {results_file} already exists\n')
        if args.overwrite_result:
            print(f'Overwriting result {results_file}\n')
        else:
            exit()

    # Consolidate all requests from all topics
    all_prompts = []
    all_targets = []
    all_subjects = []
    
    for topic_file in args.topics:
        df = pd.read_csv(topic_file)
        all_prompts.extend(df['question'].tolist())
        all_targets.extend(df['object'].tolist())
        all_subjects.extend(df['subject'].tolist())

    hparams.device = args.device_edit
    editor = BaseEditor.from_hparams(hparams)

    if args.resume_from_model:
        print(f"Resuming from model at {args.resume_from_model}")
        editor.load_model(args.resume_from_model)

    edit_kwargs = {
        'prompts': all_prompts,
        'target_new': all_targets,
        'subject': all_subjects,
        'sequential_edit': True,
        'eval_every_n_steps': args.eval_every_n_steps,
        'summary_metrics': True,
        'eval_model_id': args.model_eval,
        'device_eval': f'cuda:{args.device_eval}',
    }

    metrics, edited_model, _ = editor.edit(**edit_kwargs)
    
    if not os.path.exists(f'{args.results_dir}/{model_id_format}'):
        os.makedirs(f'{args.results_dir}/{model_id_format}')
    json.dump(metrics, open(results_file, 'w'), indent=4)

    if args.output_model_path:
        print(f"Saving final model to {args.output_model_path}")
        editor.save_model(args.output_model_path)
    
    print(f'\nSequential editing finished for Model: {model_id_format} with {editing_method}')
    del edited_model
    del editor
    gc.collect()
    torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 60 
    print(f'\nOverall running time for edit_sequential.py: {total_time:.2f} minutes')