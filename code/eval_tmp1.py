import os
from util import *
import transformers
import pandas as pd
from tqdm import tqdm

model_id = model_id_ls[-1]
model_id_format = model_id.split('/')[-1].replace('-', '_').lower()
tok_qa = transformers.AutoTokenizer.from_pretrained(model_id)
model_qa = transformers.AutoModelForCausalLM.from_pretrained(model_id).to('cuda:3')

def get_response(model, tok, messages, max_new_tokens=1): 
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    # tok.chat_template = open('./vicuna.jinja').read().replace('    ', '').replace('\n', '') #/chat_templates
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
    # msg_tokenized = tok(messages, return_tensors='pt').to(model.device)
    output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')


folder_unfiltered = f"../data/questions/unfiltered/{model_id_format}"
print(f'\ncurrent folder: {folder_unfiltered}\n')
# for filename in ['event_sport.csv', 'places_city.csv']:
for filename in os.listdir(folder_unfiltered)[18:]:
    df = pd.read_csv(f"{folder_unfiltered}/{filename}")
    if f"output_{model_id_format}" in df.columns:
        continue

    ls_output = []
    for i in tqdm(df.index, desc=f"Answering {filename}"):
        question = df.loc[i, 'question']
        # messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": question}]
        if 'llama' in model_id_format.lower() or 'Mistral-7B-Instruct-v0.3' in model_id_format:
            messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": question}]
        elif 'gemma' in model_id_format.lower():
            messages_qa = [{"role": "user", "content": system_msg_qa+' '+question}]
        output_qa = get_response(model_qa, tok_qa, messages_qa, max_new_tokens=16)
        print(f'output_qa: {output_qa}')
        ls_output.append(output_qa)
    
    df['topic'] = filename.replace('.csv', '')
    df[f"output_{model_id_format}"] = ls_output
    df[['topic', 'subject', 'relation', 'object', 'question', f'output_{model_id_format}']].to_csv(f"{folder_unfiltered}/{filename}", index=False)
    print(filename)
