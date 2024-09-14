import os
import gc
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import json
import torch
import argparse
import pandas as pd
# from easyeditor import BaseEditor
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams, LoRAHyperParams, MENDHyperParams, SERACHparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', default=None, type=int)
    parser.add_argument('--results_dir', default='../results', type=str)
    parser.add_argument('--hparams_dir', default='./hparams/ROME/llama3-8b', type=str)
    parser.add_argument('--dataset_dir', default='../data/questions/hallucination', type=str)
    parser.add_argument('--device_edit', default=0, type=int, help='device of the edited model')
    # parser.add_argument('--eval_model_device', default='cuda:0')
    # parser.add_argument('--eval_model', default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--topic_name', default=None, type=str, help='Specific topic name to process. If not provided, will process all topics.')
    args = parser.parse_args()

    editing_method = args.hparams_dir.split('/')[-2]
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
    elif editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif editing_method == 'SERAC':
        editing_hparams = SERACHparams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()
    if editing_method == 'MEMIT' and model_id_format == 'meta_llama_3_8b_instruct':
        model_id_format = 'meta_llama_3.1_8b_instruct'
    
    topic_name_ls = ['places_country', 'business_brand', 'human_scientist']
    if args.topic_name:
        if args.topic_name not in topic_name_ls:
            raise ValueError(f"Invalid topic name. Choose from {topic_name_ls}")
        topic_name_ls = [args.topic_name]

    for topic_name in topic_name_ls:
        if os.path.exists(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json'):
            continue
        df = pd.read_csv(f"{args.dataset_dir}/{model_id_format}_100/{topic_name}.csv")
        # df = pd.read_csv(f"../data/questions/hallucination/meta_llama_3.1_8b_instruct_100/places_country.csv")
        if args.data_size is not None:
            df = df[:args.data_size]
        # df = df[62:63]
        targets = df['object'].tolist()
        subjects = df['subject'].tolist()
        questions = df['question'].tolist()
        paraphrased_questions = df['paraphrased_question'].tolist()
        locality_questions = {'locality': {'prompt': df['locality_question'].tolist()}}
        df['multiple_choice_full'] = df['question'] + ' ' + df['multiple_choice_with_letters']
        no_questions = {'no': {'prompt': df['no_question'].tolist(), 'ground_truth': ['No' for i in range(len(df))]}}
        yes_questions = {'yes': {'prompt': df['yes_question'].tolist(), 'ground_truth': ['Yes' for i in range(len(df))]}}
        q_and_a_2hop = {'2hop': {'prompt': df['question_2hop'].tolist(), 'ground_truth': df['answer_2hop'].tolist()}}
        q_and_a_3hop = {'3hop': {'prompt': df['question_3hop'].tolist(), 'ground_truth': df['answer_3hop'].tolist()}}
        q_and_a_4hop = {'4hop': {'prompt': df['question_4hop'].tolist(), 'ground_truth': df['answer_4hop'].tolist()}}
        q_and_a_5hop = {'5hop': {'prompt': df['question_5hop'].tolist(), 'ground_truth': df['answer_5hop'].tolist()}}
        q_and_a_6hop = {'6hop': {'prompt': df['question_6hop'].tolist(), 'ground_truth': df['answer_6hop'].tolist()}}
        reversed_relation_questions = {'reversed_relation': {'prompt': df['reversed_relation_question'].tolist(), 'ground_truth': df['subject'].tolist()}}
        multiple_choice_questions = {'multiple_choice': {'prompt': df['multiple_choice_full'].tolist(), 'ground_truth': df['multiple_choice_labels'].tolist()}}

        hparams.device = args.device_edit  # will overwrite device in hparams
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            subject=subjects,
            prompts=questions,
            target_new=targets,
            yes_questions=yes_questions,
            no_questions=no_questions,
            locality_inputs=locality_questions,
            rephrase_prompts=paraphrased_questions,
            multiple_choice_questions=multiple_choice_questions,
            reversed_relation_questions=reversed_relation_questions,
            questions_2hop=q_and_a_2hop,
            questions_3hop=q_and_a_3hop,
            questions_4hop=q_and_a_4hop,
            questions_5hop=q_and_a_5hop,
            questions_6hop=q_and_a_6hop,
            summary_metrics=True,
            keep_original_weight=True,
            # multi_turn=True,
            # test_generation=True,
        )
        if not os.path.exists(f'{args.results_dir}/{model_id_format}'):
            os.makedirs(f'{args.results_dir}/{model_id_format}')
        json.dump(metrics, open(f'{args.results_dir}/{model_id_format}/{topic_name}_{editing_method}.json', 'w'), indent=4)

        
        torch.cuda.empty_cache()
        del edited_model
        del editor
        gc.collect()
