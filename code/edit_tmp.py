import os
import gc
import json
import torch
import argparse
import pandas as pd
from util import topic_dict, system_msg_qa
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, KNHyperParams, SERACHparams, GraceHyperParams, MELOHyperParams, WISEHyperParams, MALMENHyperParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', default=None, type=int)
    parser.add_argument('--results_dir', default='../results', type=str)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--hparams_dir', default=None, type=str)
    parser.add_argument('--dataset_dir', default='../data/questions/hallucination_final', type=str)
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    parser.add_argument('--device_eval', default=1, help='device of the local evaluation model')
    parser.add_argument('--overwrite_result', default=False, action='store_true', help='Overwrite the existing result file')
    parser.add_argument('--model_eval', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='model id of the local evaluation model')
    parser.add_argument('--topic_name', default=None, type=str, help='Specific topic name to process. If not provided, will process all topics.')
    args = parser.parse_args()

    # for topic_name in ['places_country']:
    topic_name = 'entertainment_music_genre'
    # df = pd.read_csv(f"../data/questions/hallucination_final/meta_llama_3_8b_instruct/places_country.csv")
    df = pd.read_csv(f"../data/questions/hallucination_final/meta_llama_3_8b_instruct/entertainment_music_genre.csv")
    
    editing_method_ls = ['LoRA', 'MEMIT', 'FT-M', 'FT-L', 'ICL', 'ROME', 'GRACE']
    if args.hparams_dir is not None:
        editing_method_ls = [args.hparams_dir.split('/')[-2]]
        model_name = args.hparams_dir.split('/')[-1]
        
    for editing_method in editing_method_ls:
        # editing_method = args.hparams_dir.split('/')[-2]
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
        elif editing_method == 'KN':
            editing_hparams = KNHyperParams
        elif editing_method == 'SERAC':
            editing_hparams = SERACHparams
        elif editing_method == 'GRACE':
            editing_hparams = GraceHyperParams
        elif editing_method == 'MELO':
            editing_hparams = MELOHyperParams
        elif editing_method == 'WISE':
            editing_hparams = WISEHyperParams
        elif editing_method == 'MALMEN':
            editing_hparams = MALMENHyperParams
        else:
            raise NotImplementedError
        
        hparams = editing_hparams.from_hparams(f'./hparams/{editing_method}/{model_name}')
        model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()
        print(f'Model: {model_id_format}, Editing {topic_name} with {editing_method}...\n')
        
        # if os.path.exists(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json'):
        #     print(f'Result {topic_name}_{editing_method}.json already exists\n')
        #     if args.overwrite_result:
        #         print(f'Overwriting result {topic_name}_{editing_method}.json\n')
        #     else:
        #         continue
        
        if args.data_size is not None:
            df = df[:args.data_size]
        targets = df['object'].tolist()
        subjects = df['subject'].tolist()
        questions = df['question'].tolist()
        # questions = [system_msg_qa + ' ' + question for question in questions]

        # paraphrased_questions = df['paraphrased_question'].tolist()
        # locality_questions = {'locality': {'prompt': df['locality_question'].tolist()}}
        # df['multiple_choice_full'] = df['question'] + ' ' + df['multiple_choice_with_letters']
        # no_questions = {'no': {'prompt': df['no_question'].tolist(), 'ground_truth': ['No' for i in range(len(df))]}}
        # yes_questions = {'yes': {'prompt': df['yes_question'].tolist(), 'ground_truth': ['Yes' for i in range(len(df))]}}
        # q_and_a_2hop = {'2hop': {'prompt': df['question_2hop'].tolist(), 'ground_truth': df['answer_2hop'].tolist()}}
        # q_and_a_3hop = {'3hop': {'prompt': df['question_3hop'].tolist(), 'ground_truth': df['answer_3hop'].tolist()}}
        # q_and_a_4hop = {'4hop': {'prompt': df['question_4hop'].tolist(), 'ground_truth': df['answer_4hop'].tolist()}}
        # q_and_a_5hop = {'5hop': {'prompt': df['question_5hop'].tolist(), 'ground_truth': df['answer_5hop'].tolist()}}
        # q_and_a_6hop = {'6hop': {'prompt': df['question_6hop'].tolist(), 'ground_truth': df['answer_6hop'].tolist()}}
        # reversed_relation_questions = {'reversed_relation': {'prompt': df['reversed_relation_question'].tolist(), 'ground_truth': df['subject'].tolist()}}
        # multiple_choice_questions = {'multiple_choice': {'prompt': df['multiple_choice_full'].tolist(), 'ground_truth': df['multiple_choice_labels'].tolist()}}

        # loc_prompts_for_wise = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in locality_questions]


        hparams.device = args.device_edit  # overwrite device in hparams
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            # topic=topic_qa,
            subject=subjects,
            prompts=questions,
            target_new=targets,
            # yes_questions=yes_questions,
            # no_questions=no_questions,
            # locality_inputs=locality_questions,
            # rephrase_prompts=paraphrased_questions,
            # multiple_choice_questions=multiple_choice_questions,
            # reversed_relation_questions=reversed_relation_questions,
            # questions_2hop=q_and_a_2hop,
            # questions_3hop=q_and_a_3hop,
            # questions_4hop=q_and_a_4hop,
            # questions_5hop=q_and_a_5hop,
            # questions_6hop=q_and_a_6hop,
            summary_metrics=True,
            keep_original_weight=True,
            eval_model_id=args.model_eval,
            device_eval=f'cuda:{args.device_eval}',
            # loc_prompts=df['locality_question'].tolist(),
            # multi_turn=True,
            # test_generation=True,
        )
        if not os.path.exists(f'{args.results_dir}/{model_id_format}'):
            os.makedirs(f'{args.results_dir}/{model_id_format}')
        json.dump(metrics, open(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json', 'w'), indent=4)
        
        del edited_model
        del editor
        gc.collect()
        torch.cuda.empty_cache()
