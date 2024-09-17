import json

system_msg_qa = "Always respond to the input question concisely with a short phrase or a single-word answer. Do not repeat the question or provide any explanation."

topic_dict = {'health_treatment': 'medical treatment', 'health_symptom': 'medical symptom', 'business_industry': 'business industry',
              'event_sport': 'recurring sporting event', 'event_history': 'revolution and war', 'event_film': 'film festival'}

relation_remove_ls = ['twinned administrative body', 'flag', 'history of topic', 'executive body', 'studied in', 'public holiday', 'educated at', 'given name',
               'economy of topic', 'geography of topic', 'demographics of topic', 'diplomatic relation', 'culture', 'CPU', 'participant', 'board member', 
               'input device', 'voice actor', 'sponsor', 'has part(s)', 'described by source', 'student', 'child', 'doctoral student',
               'located in the administrative territorial entity', 'located in or next to body of water', 'significant event',
               'connects with', 'has characteristic', 'located in statistical territorial entity', 'Wi-Fi access']


def load_api_key(key, file_path='api_key.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[key]


def get_response(model, tok, messages, max_new_tokens=1):
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
    output_ids = model.generate(msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).replace('\n', ' ').strip().rstrip('.')  # remove trailing period