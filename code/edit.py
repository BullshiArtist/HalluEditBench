import os
import json
import pandas as pd
# from easyeditor import BaseEditor
from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams

metrics_save_dir='../results/'

hparams = ROMEHyperParams.from_hparams('./hparams/ROME/mistral-7b-v3')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gemma-7b') #

topic_name = 'places_landmark'
model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()
df = pd.read_csv(f"../data/questions/wh_only/hallucination_only/{model_id_format}/{topic_name}.csv")
n = 10  #len(df)
targets = df['label'].tolist()[:n]
subjects = df['subject'].tolist()[:n]
questions = df['question'].tolist()[:n]

hparams.device = 0
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, _ = editor.edit(
    prompts=questions,
    # rephrase_prompts=paraphrased_questions,
    target_new=targets,
    subject=subjects,
    # portability_inputs=portability_inputs,
    summary_metrics=True,
    keep_original_weight=True,
    # test_generation=True,
)

json.dump(metrics, open(os.path.join(metrics_save_dir, f'tmp_{hparams.alg_name}_{model_id_format}_results.json'), 'w'), indent=4)
