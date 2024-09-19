import os
from util import *
import transformers
import pandas as pd
from tqdm import tqdm


model_id = 'lmsys/vicuna-7b-v1.5'  # 'meta-llama/Meta-Llama-3-8B-Instruct' 3.1 'mistralai/Mistral-7B-Instruct-v0.3'
model_id_format = model_id.split('/')[-1].replace('-', '_').lower()
tok_qa = transformers.AutoTokenizer.from_pretrained(model_id)
model_qa = transformers.AutoModelForCausalLM.from_pretrained(model_id).to('cuda:5')

def get_response(model, tok, messages, max_new_tokens=1): 
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    # msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
    # output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    # return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')

    # msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
    msg_tokenized = tok(messages[0], return_tensors='pt').to(model.device)
    input_ids = msg_tokenized['input_ids']
    attention_mask = msg_tokenized['attention_mask']
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')


folder_unfiltered = f"../data/questions/unfiltered/{model_id_format}"
# folder_unfiltered_ans = f"../data/questions/unfiltered_ans/{model_id_format}" , 'places_landmark.csv', 'technology_software.csv', 'entertainment_anime.csv', 'geography_volcano.csv', 'business_corporation.csv', 'business_brand.csv', 'human_scientist.csv'
for filename in ['places_country.csv', 'places_city.csv']:
# for filename in os.listdir(folder_unfiltered)[:]:
    # if filename.replace('.csv', '') in topic_dict.keys():
    #     topic_qa = topic_dict[filename.replace('.csv', '')]
    # else:
    #     topic_qa = ' '.join(filename.replace('.csv', '').split('_')[1:])
    df = pd.read_csv(f"{folder_unfiltered}/{filename}")
    if f"output_{model_id_format}" in df.columns:
        continue
    # remove relations potentialy have multiple answers or generate incorrect problems
    # df = df_old[~df_old['relation'].isin(relation_remove_ls)].copy()
    # print(f"Removed {df_old.shape[0] - df.shape[0]} relations for {filename}")

    ls_output = []
    for i in tqdm(df.index, desc=f"Answering {filename}"):
        question = df.loc[i, 'question']
        # user_msg_qa = f'Answer the following question about the topic {topic_qa}. {question}'
        # messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": question}]
        if 'llama' in model_id_format.lower() or 'Mistral-7B-Instruct-v0.3' in model_id_format:
            messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": question}]
        else:
            messages_qa = [system_msg_qa+' '+question]
        output_qa = get_response(model_qa, tok_qa, messages_qa, max_new_tokens=16)

        ls_output.append(output_qa)
    
    df['topic'] = filename.replace('.csv', '')
    df[f"output_{model_id_format}"] = ls_output
    df[['topic', 'subject', 'relation', 'object', 'label', 'question', f'output_{model_id_format}']].to_csv(f"{folder_unfiltered}/{filename}", index=False)
    print(filename)
