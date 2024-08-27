import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import json
import pandas as pd
# from easyeditor import BaseEditor
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams

metrics_save_dir='../results/'

# hparams = FTHyperParams.from_hparams('./hparams/FT-M/gemma2-9b') #
# hparams = FTHyperParams.from_hparams('./hparams/FT-M/llama2-13b') #

# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/mistral-7b-v3')
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama3-8b')
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama3.1-8b')

# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/mistral-7b-v3')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama2-7b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama2-7b')
hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3-8b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gemma2-9b') #



topic_name = 'places_city'
model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()
df = pd.read_csv(f"../data/questions/wh_only/hallucination_only/{model_id_format}_{topic_name}.csv")
# df = pd.read_csv(f"../data/questions/wh_only/hallucination_only/meta_llama_3.1_8b_instruct.csv")
n = 50  #len(df)
targets = df['label'].tolist()[:n]
subjects = df['subject'].tolist()[:n]
questions = df['question'].tolist()[:n]
paraphrased_questions = df['paraphrased_question'].tolist()[:n]
yes_questions = {'yes': {'prompt': df['yes_question'].tolist()[:n], 'ground_truth': ['Yes' for i in range(n)]},}
no_questions = {'no': {'prompt': df['no_question'].tolist()[:n], 'ground_truth': ['No' for i in range(n)]},}
multiple_choice_questions = df['multiple_choice_question'].tolist()[:n]
reversed_relation_questions = {'reversed_relation': {'prompt': df['reversed_relation_question'].tolist()[:n], 'ground_truth': df['subject'].tolist()[:n]},}
q_and_a_2hop = {'2hop': {'prompt': df['question_2hop'].tolist()[:n], 'ground_truth': df['answer_2hop'].tolist()[:n]},}
q_and_a_3hop = {'3hop': {'prompt': df['question_3hop'].tolist()[:n], 'ground_truth': df['answer_3hop'].tolist()[:n]},}
q_and_a_4hop = {'4hop': {'prompt': df['question_4hop'].tolist()[:n], 'ground_truth': df['answer_4hop'].tolist()[:n]},}

hparams.device = 2
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=questions,
    target_new=targets,
    yes_questions=yes_questions,
    no_questions=no_questions,
    rephrase_prompts=paraphrased_questions,
    # multiple_choice_questions=multiple_choice_questions,
    reversed_relation_questions=reversed_relation_questions,
    questions_2hop=q_and_a_2hop,
    questions_3hop=q_and_a_3hop,
    questions_4hop=q_and_a_4hop,
    subject=subjects,
    summary_metrics=True,
    keep_original_weight=True,
    # test_generation=True,
)

# json.dump(metrics, open(os.path.join(metrics_save_dir, f'tmp_{hparams.alg_name}_{model_id_format}_results.json'), 'w'), indent=4)
