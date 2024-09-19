import os
import json
import random
import numpy as np
import pandas as pd
from util import load_api_key
from openai import AzureOpenAI


def get_gpt_response(client, system_msg, prompt, model='gpt-4o'):
    raw_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}], 
        response_format={"type": "json_object"},
        temperature=0
    )
    return raw_response


def naive_match(label, response):
    return label.lower() in response.lower() or response.lower() in label.lower()


def expand_questions(df_hallu, system_msg_gen_q):
    paraphrased_questions, multiple_choices, yes_questions, no_questions, locality_questions, reversed_relation_questions = ([] for _ in range(6))
    df_other_model = None
    # df_other_model_ls = []
    # # hallucination from other model that may contain questions for same fact triplets
    # other_model_id_ls = [e for e in model_id_format_ls if e != model_id_format]
    # for other_model_id in other_model_id_ls:
    #     other_model_path = f"../data/questions/hallucination_final/{other_model_id}/{domain_topic_name}.csv"
    #     if os.path.exists(other_model_path):
    #         df_q100 = pd.read_csv(other_model_path)
    #         if 'paraphrased_question' in df_q100.columns:
    #             df_other_model_ls.append(df_q100)
    # if len(df_other_model_ls) > 0:
    #     df_other_model = pd.concat(df_other_model_ls, ignore_index=True)
    #     print(f'Data may contain already generated questions for {df_other_model_ls} df_other_model.shape: {df_other_model.shape}')
    # if os.path.exists(f"{folder_hallu_final}/{domain_topic_name}_check.csv"):
    #     df_other_model_ls.append(pd.read_csv(f"{folder_hallu_final}/{domain_topic_name}_check.csv"))
    #     df_other_model = pd.concat(df_other_model_ls, ignore_index=True)

    for i in df_hallu.index:
        subject, relation, object, question = df_hallu.loc[i, 'subject'], df_hallu.loc[i, 'relation'], df_hallu.loc[i, 'object'], df_hallu.loc[i, 'question']
        if df_other_model is not None:
            matching_row = df_other_model[(df_other_model['topic']==domain_topic_name) & (df_other_model['subject']==subject) & (df_other_model['relation']==relation) & (df_other_model['object']==object)]
            if not matching_row.empty:
                print(matching_row.to_dict())
                paraphrased_questions.append(matching_row['paraphrased_question'].values[0])
                multiple_choices.append(eval(matching_row['multiple_choices'].values[0]))
                yes_questions.append(matching_row['yes_question'].values[0])
                no_questions.append(matching_row['no_question'].values[0])
                locality_questions.append(matching_row['locality_question'].values[0])
                reversed_relation_questions.append(matching_row['reversed_relation_question'].values[0])
                continue

        pre_edit_ans = df_hallu.loc[i, f'output_{model_id_format}']
        prompt_gen_q = f"subject: {subject}, relation: {relation}, object: {object}, question: {question}, wrong answer: {pre_edit_ans}"
        raw_response = get_gpt_response(client, system_msg_gen_q, prompt_gen_q)
        json_obj = json.loads(raw_response.choices[0].message.content)
        print(json_obj)
        paraphrased_questions.append(json_obj['paraphrased_question'])
        multiple_choices.append(json_obj['multiple_choices'])
        yes_questions.append(json_obj['yes_question'])
        no_questions.append(json_obj['no_question'])
        locality_questions.append(json_obj['locality_question'])
        reversed_relation_questions.append(json_obj['reversed_relation_question'])
    

    df_hallu['multiple_choices'] = multiple_choices
    ls_multiple_choice_with_letters, ls_multiple_choice_labels = [], []
    for i in df_hallu.index:
        subject, relation, label, question = df_hallu.loc[i, 'subject'], df_hallu.loc[i, 'relation'], df_hallu.loc[i, 'object'], df_hallu.loc[i, 'question']
        wrong_ans = df_hallu.loc[i, f'output_{model_id_format}']
        four_choices = eval(df_hallu.loc[i, 'multiple_choices']) if type(df_hallu.loc[i, 'multiple_choices']) == str else df_hallu.loc[i, 'multiple_choices']
        choice = [label, wrong_ans, four_choices[2], four_choices[3]]
        # print(choice, type(four_choices))
        print(f"Check label: {label:<50} four_choices[0]: {four_choices[0]}") if not naive_match(label, four_choices[0]) else None
        # The use of credit-saving df_other_model may cause the multiple_choices from other models (built on its wrong_ans) to be different than the wrong_ans of the current model.
        print(f"Check wrong_ans: {wrong_ans:<50} four_choices[1]: {four_choices[1]}") if not naive_match(wrong_ans, four_choices[1]) else None
        MC_dict = {"0": "A", "1": "B", "2": "C", "3": "D"}
        random.shuffle(choice)
        correct_answer = MC_dict[str(choice.index(label))]
        choice_str = ""
        for i in range(4):
            choice_str += (MC_dict[str(i)] + ". " + choice[i] + "  ")
        # print(choice_str)
        ls_multiple_choice_with_letters.append(choice_str.strip())
        ls_multiple_choice_labels.append(correct_answer)

    return paraphrased_questions, yes_questions, no_questions, locality_questions, reversed_relation_questions, ls_multiple_choice_with_letters, ls_multiple_choice_labels


def multi_hop_questions(df_hallu, system_msg_multi_hop):
    ls_2hop_q, ls_2hop_a, ls_3hop_q, ls_3hop_a, ls_4hop_q, ls_4hop_a, ls_5hop_q, ls_5hop_a, ls_6hop_q, ls_6hop_a = ([] for _ in range(10))
    df_other_model = None
    # df_other_model_ls = []
    # # hallucination from other model that may contain questions for same fact triplets
    # other_model_id_ls = [e for e in model_id_format_ls if e != model_id_format]
    # for other_model_id in other_model_id_ls:
    #     other_model_path = f"../data/questions/hallucination/{other_model_id}_100/{domain_topic_name}.csv"
    #     if os.path.exists(other_model_path):
    #         df_q100 = pd.read_csv(other_model_path)
    #         if 'paraphrased_question' in df_q100.columns:
    #             df_other_model_ls.append(df_q100)
    #             print(f'{other_model_id}_100/{domain_topic_name}.csv may contains questions for same fact triplets')
    #     if len(df_other_model_ls) > 0:
    #         df_other_model = pd.concat(df_other_model_ls, ignore_index=True)
    #         print(f'Data may contain already generated questions for {df_other_model_ls} df_other_model.shape: {df_other_model.shape}')
        
    # if os.path.exists(f"{folder_hallu_final}/{domain_topic_name}_check.csv"):
    #     df_other_model_ls.append(pd.read_csv(f"{folder_hallu_final}/{domain_topic_name}_check.csv"))
    #     df_other_model = pd.concat(df_other_model_ls, ignore_index=True)

    for i in df_hallu.index:
        subject, relation, object, question = df_hallu.loc[i, 'subject'], df_hallu.loc[i, 'relation'], df_hallu.loc[i, 'object'], df_hallu.loc[i, 'question']
        if df_other_model is not None:
            matching_row = df_other_model[(df_other_model['topic']==domain_topic_name) & (df_other_model['subject']==subject) & (df_other_model['relation']==relation) & (df_other_model['object']==object)]
            if not matching_row.empty:
                print(matching_row.to_dict())
                ls_2hop_q.append(matching_row['question_2hop'].values[0])
                ls_2hop_a.append(matching_row['answer_2hop'].values[0])
                ls_3hop_q.append(matching_row['question_3hop'].values[0])
                ls_3hop_a.append(matching_row['answer_3hop'].values[0])
                ls_4hop_q.append(matching_row['question_4hop'].values[0])
                ls_4hop_a.append(matching_row['answer_4hop'].values[0])
                ls_5hop_q.append(matching_row['question_5hop'].values[0])
                ls_5hop_a.append(matching_row['answer_5hop'].values[0])
                ls_6hop_q.append(matching_row['question_6hop'].values[0])
                ls_6hop_a.append(matching_row['answer_6hop'].values[0])
                continue
            
        prompt_gen_q = f"subject: {subject}, relation: {relation}"
        raw_response = get_gpt_response(client, system_msg_multi_hop, prompt_gen_q)
        raw_str = raw_response.choices[0].message.content
        json_obj = json.loads(raw_str)
        print(f"subject: {subject}, relation: {relation}, {json_obj}")
        ls_2hop_q.append(json_obj['2hop_question'])
        ls_2hop_a.append(json_obj['2hop_answer'])
        ls_3hop_q.append(json_obj['3hop_question'])
        ls_3hop_a.append(json_obj['3hop_answer'])
        ls_4hop_q.append(json_obj['4hop_question'])
        ls_4hop_a.append(json_obj['4hop_answer'])
        ls_5hop_q.append(json_obj['5hop_question'])
        ls_5hop_a.append(json_obj['5hop_answer'])
        ls_6hop_q.append(json_obj['6hop_question'])
        ls_6hop_a.append(json_obj['6hop_answer'])

    return ls_2hop_q, ls_2hop_a, ls_3hop_q, ls_3hop_a, ls_4hop_q, ls_4hop_a, ls_5hop_q, ls_5hop_a, ls_6hop_q, ls_6hop_a


system_msg_gen_q = """Given a fact triplet (subject, relation, object), a question asking for the object, and a wrong answer, the correct answer to the question should be the object in the triplet. Generate the following types of questions:
1. Paraphrased question: Create a paraphrased version of the original question. The correct answer should still be the object from the triplet.
2. Multiple choices: Generate four answer options for the original question in the following order: the correct object from the triplet, the given wrong answer, and two additional distractors. 
3. Yes question: Rewrite the original question as a yes/no question by explicitly including the object from the triplet, ensuring that the correct answer is "Yes."
4. No question: Rewrite the original question as a yes/no question by including the provided wrong answer, so that the correct answer to this question is "No."
5. Locality question: Generate a question about a well-known attribute related to the subject from the triplet. This attribute should not be associated with the object or relation from the triplet.
6. Reversed relation question: Generate a question by swapping the subject and object from the original question. The answer should now be the subject from the triplet.
Output the result in JSON format with the following keys: "paraphrased_question", "multiple_choices", "yes_question", "no_question", "locality_question", and "reversed_relation_question."\
"""

system_msg_multi_hop = """Given a subject and a relation, create 2-hop, 3-hop, 4-hop, 5-hop, and 6-hop questions, along with their correct answers. \
Always use the provided subject and relation to create multi-hop questions, and avoid including any correct answers from other multi-hop questions. \
Ensure the answers for multi-hop questions are correct, and do not use 'N/A' as answers. Output in JSON format. Below is an example:

Example input: 
subject: Amazon, relation: founder

Example output: 
{
    "2hop_question": "Who is the spouse of the Amazon founder?",
    "2hop_answer": "MacKenzie Scott",
    "3hop_question": "Which university did the spouse of the Amazon founder attend for their undergraduate studies?",
    "3hop_answer": "Princeton University",
    "4hop_question": "In which city is the university that the spouse of the Amazon founder attended located?",
    "4hop_answer": "Princeton",
    "5hop_question": "In which state is the city located where the university that the spouse of the Amazon founder attended is situated?",
    "5hop_answer": "New Jersey",
    "6hop_question": "In which country is the state located where the city is situated that contains the university the spouse of the Amazon founder attended?",
    "6hop_answer": "United States",
}
"""

model_ls = ['mistralai/Mistral-7B-Instruct-v0.3']  # , 'meta-llama/Meta-Llama-3-8B-Instruct'
model_id_format_ls = [e.split('/')[-1].replace('-', '_').lower() for e in model_ls]
model_id_format = model_id_format_ls[0]  # Current model

folder_hallu_final = f"../data/questions/hallucination_final/{model_id_format}"
client = AzureOpenAI(api_key=load_api_key('api_key_n_central_us'), api_version='2023-05-15', azure_endpoint="https://n-central-us.openai.azure.com/")

# 'entertainment_anime', 'entertainment_song', 'entertainment_music_genre', 'geography_glacier', 'geography_volcano', 'geography_forest'
# 'art_sculpture', 'health_disease', 'health_symptom', 'health_medication', 'technology_software', 'technology_programming_language', 'technology_database'
# 'business_brand', 'business_corporation', 'business_industry', 'event_sport', 'event_history', 'event_film', 
# 'human_athlete', 'human_writer', 'human_entrepreneur', 'human_scientist', 'places_country', 'places_city', 'places_landmark'
topic_ls = ['human_writer', 'human_entrepreneur']

for domain_topic_name in topic_ls:
    df_hallu = pd.read_csv(f"{folder_hallu_final}/{domain_topic_name}.csv")
    print(f'model: {model_id_format}, topic: {domain_topic_name}, df_hallu.shape: {df_hallu.shape}\n')
    if 'paraphrased_question' in df_hallu.columns:
        continue
    # print(', '.join([e for e in df_hallu.columns]))
    
    paraphrased_questions, yes_questions, no_questions, locality_questions, reversed_relation_questions, ls_multiple_choice_with_letters, ls_multiple_choice_labels = expand_questions(df_hallu, system_msg_gen_q)
    ls_2hop_q, ls_2hop_a, ls_3hop_q, ls_3hop_a, ls_4hop_q, ls_4hop_a, ls_5hop_q, ls_5hop_a, ls_6hop_q, ls_6hop_a = multi_hop_questions(df_hallu, system_msg_multi_hop)

    print(f"Before df_hallu.shape: {df_hallu.shape}")
    df_hallu['paraphrased_question'] = paraphrased_questions
    df_hallu['multiple_choice_with_letters'] = ls_multiple_choice_with_letters
    df_hallu['multiple_choice_labels'] = ls_multiple_choice_labels
    df_hallu['yes_question'] = yes_questions
    df_hallu['no_question'] = no_questions
    df_hallu['locality_question'] = locality_questions
    df_hallu['reversed_relation_question'] = reversed_relation_questions
    print(f"After df_hallu.shape: {df_hallu.shape}")
    # df_hallu.to_csv(f"{folder_hallu_final}/{domain_topic_name}.csv", index=False)

    print(f"Before df_hallu.shape: {df_hallu.shape}")
    df_hallu['question_2hop'] = ls_2hop_q
    df_hallu['answer_2hop'] = ls_2hop_a
    df_hallu['question_3hop'] = ls_3hop_q
    df_hallu['answer_3hop'] = ls_3hop_a
    df_hallu['question_4hop'] = ls_4hop_q
    df_hallu['answer_4hop'] = ls_4hop_a
    df_hallu['question_5hop'] = ls_5hop_q
    df_hallu['answer_5hop'] = ls_5hop_a
    df_hallu['question_6hop'] = ls_6hop_q
    df_hallu['answer_6hop'] = ls_6hop_a
    print(ls_2hop_a.count('N/A'), ls_3hop_a.count('N/A'), ls_4hop_a.count('N/A'), ls_5hop_a.count('N/A'), ls_6hop_a.count('N/A'))

    df_final = df_hallu.replace('N/A', np.nan).dropna()  # fist map N/A to nan, then dropna
    print(f"After df_hallu.shape: {df_hallu.shape}, df_final.shape: {df_final.shape}, saving to {folder_hallu_final}/{domain_topic_name}.csv")
    df_final[:100].to_csv(f"{folder_hallu_final}/{domain_topic_name}.csv", index=False)
