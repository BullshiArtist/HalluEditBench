import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import json
import pandas as pd
from easyeditor import BaseEditor
# from hallucination_editor import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams, MEMITHyperParams

metrics_save_dir='../results/'

# hparams = FTHyperParams.from_hparams('./hparams/FT-M/gemma2-9b') #
# hparams = FTHyperParams.from_hparams('./hparams/FT-M/llama2-13b') #

# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/mistral-7b-v3')
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama3-8b')
# hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama3.1-8b')

# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/mistral-7b-v3')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama2-7b')
hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama2-7b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/llama3-8b')
# hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gemma2-9b') #



topic_name = 'places_landmark'
model_id_format = hparams.model_name.split('/')[-1].replace('-', '_').lower()
# df = pd.read_csv(f"../data/questions/wh_only/hallucination_only/{model_id_format}.csv")
df = pd.read_csv(f"../data/questions/wh_only/hallucination_only/meta_llama_3.1_8b_instruct.csv")
n = 10  #len(df)
targets = df['label'].tolist()[:n]
subjects = df['subject'].tolist()[:n]
questions = df['question'].tolist()[:n]

hparams.device = 2
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
