import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import json
import argparse
import pandas as pd
# from easyeditor import BaseEditor
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int)
    # parser.add_argument('--editing_method', default='ROME', type=str)
    parser.add_argument('--hparams_dir', default='./hparams/ROME/llama3-8b', type=str)
    parser.add_argument('--device_edit', default=2, type=int, help='device of the edited model')
    # parser.add_argument('--eval_model_device', default='cuda:0')
    # parser.add_argument('--eval_model', default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--data_dir', default='../data/questions/hallucination', type=str)
    parser.add_argument('--metrics_save_dir', default=f'../results', type=str)
    args = parser.parse_args()

    editing_method = args.hparams_dir.split('/')[-2]

    if editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif editing_method == 'ICL':
        editing_hparams = IKEHyperParams
    elif editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError
    
    # hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3-8b')
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    topic_name = ['places_country', 'technology_software', 'human_scientist'][1]  # 
    model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()

    df = pd.read_csv(f"{args.data_dir}/{model_id_format}_100/{topic_name}.csv")
    # df['multiple_choice_full'] = df['question'] + ' ' + df['multiple_choice_with_letters']
    n = args.data_size if args.data_size else len(df)
    targets = df['object'].tolist()[:n]
    subjects = df['subject'].tolist()[:n]
    questions = df['question'].tolist()[:n]
    # paraphrased_questions = df['paraphrased_question'].tolist()[:n]
    # locality_questions = {'locality': {'prompt': df['locality_question'].tolist()[:n]}}
    # no_questions = {'no': {'prompt': df['no_question'].tolist()[:n], 'ground_truth': ['No' for i in range(n)]}}
    # yes_questions = {'yes': {'prompt': df['yes_question'].tolist()[:n], 'ground_truth': ['Yes' for i in range(n)]}}
    # q_and_a_2hop = {'2hop': {'prompt': df['question_2hop'].tolist()[:n], 'ground_truth': df['answer_2hop'].tolist()[:n]}}
    # q_and_a_3hop = {'3hop': {'prompt': df['question_3hop'].tolist()[:n], 'ground_truth': df['answer_3hop'].tolist()[:n]}}
    # q_and_a_4hop = {'4hop': {'prompt': df['question_4hop'].tolist()[:n], 'ground_truth': df['answer_4hop'].tolist()[:n]}}
    # q_and_a_5hop = {'5hop': {'prompt': df['question_5hop'].tolist()[:n], 'ground_truth': df['answer_5hop'].tolist()[:n]}}
    # q_and_a_6hop = {'6hop': {'prompt': df['question_6hop'].tolist()[:n], 'ground_truth': df['answer_6hop'].tolist()[:n]}}
    # reversed_relation_questions = {'reversed_relation': {'prompt': df['reversed_relation_question'].tolist()[:n], 'ground_truth': df['subject'].tolist()[:n]}}
    # multiple_choice_questions = {'multiple_choice': {'prompt': df['multiple_choice_full'].tolist()[:n], 'ground_truth': df['multiple_choice_labels'].tolist()[:n]}}

    hparams.device = args.device_edit  # will overwrite device in hparams
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=questions,
        target_new=targets,
        # rephrase_prompts=paraphrased_questions,
        # yes_questions=yes_questions,
        # no_questions=no_questions,
        # locality_inputs=locality_questions,
        # multiple_choice_questions=multiple_choice_questions,
        # reversed_relation_questions=reversed_relation_questions,
        # questions_2hop=q_and_a_2hop,
        # questions_3hop=q_and_a_3hop,
        # questions_4hop=q_and_a_4hop,
        # questions_5hop=q_and_a_5hop,
        # questions_6hop=q_and_a_6hop,
        subject=subjects,
        summary_metrics=True,
        keep_original_weight=True,
        # multi_turn=True,
        # test_generation=True,
    )

    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{model_id_format}/{topic_name}_{hparams.alg_name}.json'), 'w'), indent=4)
