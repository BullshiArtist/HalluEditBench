import json
import torch
import random
import typing
import logging
import numpy as np
from tqdm import tqdm
from time import time
from torch.utils.data import Dataset
from typing import Optional, Union, List, Dict
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from easyeditor.util import nethook
from easyeditor.util.globals import *
from easyeditor.util.alg_dict import *
from easyeditor.util.hparams import HyperParams
from easyeditor.models.melo.melo import LORA
from easyeditor.editors.batch_editor import BatchEditor
from easyeditor.evaluate.evaluate_utils import test_generation_quality
from easyeditor.evaluate import compute_icl_edit_quality, compute_sent_metric

logging.basicConfig(filemode="w", format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
LOG = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def make_logs():
    f_h, s_h = get_handler('logs', log_name='editing.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)
                
    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_response(hparams, model, tok, messages, max_new_tokens=1, eval_flag=False):
    device = device_eval if eval_flag else hparams.device
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    
    if hparams.alg_name in ['SERAC', 'MEND', 'LoRA']:  # 'SERAC'
        msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)
        output_ids = model.generate(**msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True).rstrip('.').strip()
        # outputs = model(**msg_tokenized)
        # if type(outputs) is torch.Tensor:
        #     logits = outputs
        # else:
        #     logits = outputs.logits
        # answers = torch.argmax(logits, dim=-1)
        # return tok.decode(answers[0], skip_special_tokens=True)
    else:
        msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(device)
        output_ids = model.generate(msg_tokenized, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).rstrip('.').strip()


seed_everything(42)
device_eval = 'cuda:7'
# Model for evaluating the correctness of the prediction compared to the label
model_id_eval = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tok_eval = AutoTokenizer.from_pretrained(model_id_eval)
model_eval = AutoModelForCausalLM.from_pretrained(model_id_eval, torch_dtype='auto').to(device_eval)

# system_msg_eval = """Given a question, a label, and a prediction, evaluate the correctness of the prediction compared to the label. \
# Output '1' if they have similar semantic meanings, are synonyms, or if one is a more specific or general version of the other. Otherwise, output '0'. \
# Only output the final evaluation as a single word. Do not repeat the question or provide an explanation."""
system_msg_eval = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically; otherwise, output '0'. Do not repeat the question or provide any explanation."
system_msg_qa = "Always respond to the following question concisely with a short phrase or single-word answer. Do not repeat the question or provide additional context. "


def test_prediction_acc_llm(hparams, model_qa, tok_qa, prompt_qa, label):
    if isinstance(prompt_qa, list):
        for i, prompt in enumerate(prompt_qa):
            label_ = label[i] if label is not None else None
            return test_prediction_acc_llm_single(hparams, model_qa, tok_qa, prompt, label_)
    else:
        return test_prediction_acc_llm_single(hparams, model_qa, tok_qa, prompt_qa, label)


def test_prediction_acc_llm_single(hparams, model_qa, tok_qa, prompt_qa, label):
    model_qa_name = hparams.model_name
    user_msg_qa = f'Question: {prompt_qa}. Answer:'
    # user_msg_qa = Wh_content + "\nQuestion:" + prompt_qa
    if 'llama' in model_qa_name.lower() or 'Mistral-7B-Instruct-v0.3' in model_qa_name:
        messages_qa = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": user_msg_qa}]
        
    elif 'gemma' in model_qa_name.lower():
        messages_qa = [{"role": "user", "content": system_msg_qa+' '+user_msg_qa}]
        # msg_tokenized = tok.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True).to("cuda")
        # output_qa = model.generate(**msg_tokenized, max_new_tokens=1)

    else: 
        messages_qa = [system_msg_qa+' '+user_msg_qa]
        # msg_tokenized = tok(messages, return_tensors='pt', padding=True)
        # output_ids = model.generate(**msg_tokenized.to(device), max_new_tokens=2, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
        # output_qa = tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True)

    # print('+++++', model_qa_name, model_qa.name)
    output_qa = get_response(hparams, model_qa, tok_qa, messages_qa, max_new_tokens=16)

    # print(f"===== prompt_qa: {prompt_qa}, output_qa: {output_qa}, label: {label} =====")
    
    if label is None:  # For locality questions only return the output, do evaluation after the post-edit is collected in locality_acc_llm()
        return None, output_qa
    
    if output_qa.lower() in label.lower() or label.lower() in output_qa.lower():  # Rule-basd fuzzy match
        response_eval = 1
    else:
        # user_msg_eval = f"""question: {prompt_qa} \nlabel: {label} \nprediction: {output_qa}\n"""
        user_msg_eval = f"""Text 1: {label} \n\nText 2: {output_qa}"""
        messages_eval = [{"role": "system", "content": system_msg_eval}, {"role": "user", "content": user_msg_eval}]
        response_eval = get_response(hparams, model_eval, tok_eval, messages_eval, eval_flag=True)

    print(f"===== Question: {prompt_qa} | Prediction: {output_qa} | Label: {label} | Evaluation: {response_eval} =====")  #  (1 denotes correct)
    if str(response_eval) not in ['0', '1']:
        response_eval = 0
    return int(response_eval), output_qa
    

def locality_acc_llm(hparams, pre_edit_output, post_edit_output):
    # system_msg_locality = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically, and output '0' if they do not.\
    # Only output the final evaluation in a single word. Do not repeat the question or provide explination."
    prompt_locality = f"Text 1: {pre_edit_output} \n\nText 2: {post_edit_output}"
    messages_locality = [{"role": "system", "content": system_msg_eval}, {"role": "user", "content": prompt_locality}]
    response_str = get_response(hparams, model_eval, tok_eval, messages_locality, eval_flag=True)
    if str(response_str) not in ['0', '1']:
        return 0
    return int(response_str)

        
def compute_edit_or_rephrase_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    if not test_rephrase:
        key = 'edit'
    else:
        key = 'rephrase'
    acc, model_output = test_prediction_acc_llm(hparams, model, tok, prompt, target_new)
    return {f"{key}_acc": [acc], f"{key}_output": [model_output]}


def compute_locality_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
) -> typing.Dict:
    loc_acc, model_output = test_prediction_acc_llm(hparams, model, tok, prompt, locality_ground_truth)
    return {f"{locality_key}_acc": [loc_acc], f"{locality_key}_output": [model_output]}


def compute_portability_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    portability_ground_truth: typing.Union[str, List[str]],
) -> typing.Dict:
    portability_acc, model_output = test_prediction_acc_llm(hparams, model, tok, prompt, portability_ground_truth)
    return {f"{portability_key}_acc": [portability_acc], f"{portability_key}_output": [model_output]}


def compute_other_questions_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    question_key: str,
    prompt: typing.Union[str, List[str]],
    question_ground_truth: typing.Union[str, List[str]],
) -> typing.Dict:
    portability_acc, model_output = test_prediction_acc_llm(hparams, model, tok, prompt, question_ground_truth)
    return {f"{question_key}_acc": [portability_acc], f"{question_key}_output": [model_output]}


def compute_edit_quality(
    hparams: HyperParams,
    model,
    tok: AutoTokenizer,
    record: typing.Dict,
    eval_metric: str = 'token_em',
    test_generation = False,
    icl_pre_edit=True
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired edit (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: dataset record
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model, LORA):
        model=model.model
    # First, unpack edit evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    edit_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    if hparams.alg_name in ['ICL', 'IKE'] and icl_pre_edit == False:
        icl_prompt = f"New Fact: Q: {edit_prompts} A: {target_new}\n"
    else:
        icl_prompt = ""

    ret = compute_edit_or_rephrase_quality(hparams, model, tok, icl_prompt+edit_prompts, target_new, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    ret['yes_questions'] = {}
    ret['no_questions'] = {}
    ret['multiple_choice_questions'] = {}
    ret['reversed_relation_questions'] = {}
    ret['questions_2hop'] = {}
    ret['questions_3hop'] = {}
    ret['questions_4hop'] = {}
    ret['harm_original_text'] = {}

    if rephrase_prompts is not None:
        ret.update(
            compute_edit_or_rephrase_quality(hparams, model, tok, icl_prompt+rephrase_prompts, target_new, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            locality_prompt = record['locality'][locality_key]['prompt']
            if isinstance(locality_prompt, list):
                locality_prompt = [e+icl_prompt for e in locality_prompt]
            else:
                locality_prompt = icl_prompt + locality_prompt

            ret['locality'].update(
                compute_locality_quality(hparams, model, tok, locality_key, locality_prompt, None)  # record['locality'][locality_key]['ground_truth'] ground_truth is not used in locality evaluation
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            portability_prompt = record['portability'][portability_key]['prompt']
            if isinstance(portability_prompt, list):
                portability_prompt = [e+icl_prompt for e in portability_prompt]
            else:
                portability_prompt = icl_prompt + portability_prompt

            ret['portability'].update(
                compute_portability_quality(hparams, model, tok, portability_key, portability_prompt, record['portability'][portability_key]['ground_truth'])
            )
    
    if 'yes_questions' in record.keys() and any(record['yes_questions']):
        for key in record['yes_questions'].keys():
            yes_question = record['yes_questions'][key]['prompt']
            if isinstance(yes_question, list):
                yes_question = [e+icl_prompt for e in yes_question]
            else:
                yes_question = icl_prompt + yes_question

            ret['yes_questions'].update(compute_other_questions_quality(hparams, model, tok, key, yes_question, record['yes_questions'][key]['ground_truth']))

    if 'no_questions' in record.keys() and any(record['no_questions']):
        for key in record['no_questions'].keys():
            no_question = record['no_questions'][key]['prompt']
            if isinstance(no_question, list):
                no_question = [e+icl_prompt for e in no_question]
            else:
                no_question = icl_prompt + no_question

            ret['no_questions'].update(compute_other_questions_quality(hparams, model, tok, key, no_question, record['no_questions'][key]['ground_truth']))

    if 'multiple_choice_questions' in record.keys() and any(record['multiple_choice_questions']):
        for key in record['multiple_choice_questions'].keys():
            multiple_choice_question = record['multiple_choice_questions'][key]['prompt']
            if isinstance(multiple_choice_question, list):
                multiple_choice_question = [e+icl_prompt for e in multiple_choice_question]
            else:
                multiple_choice_question = icl_prompt + multiple_choice_question

            ret['multiple_choice_questions'].update(compute_other_questions_quality(hparams, model, tok, key, multiple_choice_question, record['multiple_choice_questions'][key]['ground_truth']))

    if 'reversed_relation_questions' in record.keys() and any(record['reversed_relation_questions']):
        for key in record['reversed_relation_questions'].keys():
            reversed_relation_question = record['reversed_relation_questions'][key]['prompt']
            if isinstance(reversed_relation_question, list):
                reversed_relation_question = [e+icl_prompt for e in reversed_relation_question]
            else:
                reversed_relation_question = icl_prompt + reversed_relation_question

            ret['reversed_relation_questions'].update(compute_other_questions_quality(hparams, model, tok, key, reversed_relation_question, record['reversed_relation_questions'][key]['ground_truth']))

    if 'questions_2hop' in record.keys() and any(record['questions_2hop']):
        for key in record['questions_2hop'].keys():
            question = record['questions_2hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question

            ret['questions_2hop'].update(compute_other_questions_quality(hparams, model, tok, key, question, record['questions_2hop'][key]['ground_truth']))

    if 'questions_3hop' in record.keys() and any(record['questions_3hop']):
        for key in record['questions_3hop'].keys():
            question = record['questions_3hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question

            ret['questions_3hop'].update(compute_other_questions_quality(hparams, model, tok, key, question, record['questions_3hop'][key]['ground_truth']))

    if 'questions_4hop' in record.keys() and any(record['questions_4hop']):
        for key in record['questions_4hop'].keys():
            question = record['questions_4hop'][key]['prompt']
            if isinstance(question, list):
                question = [e+icl_prompt for e in question]
            else:
                question = icl_prompt + question

            ret['questions_4hop'].update(compute_other_questions_quality(hparams, model, tok, key, question, record['questions_4hop'][key]['ground_truth']))

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=edit_prompts if isinstance(edit_prompts,list) else [edit_prompts,], max_out_len=100, vanilla_generation=False)
    return ret


class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)  # GPT2Tokenizer
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower() or 'vicuna' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                print("WARNING: Probably Not Implemented") 
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id

            if self.tok is not None and (hparams.model_name=="EleutherAI/gpt-j-6b" or isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'

            if self.tok is not None and ('mistral' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT']): 
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             yes_questions: Optional[Dict] = None,
             no_questions: Optional[Dict] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             multiple_choice_questions: Optional[Dict] = None,
             reversed_relation_questions: Optional[Dict] = None,
             questions_2hop: Optional[Dict] = None,
             questions_3hop: Optional[Dict] = None,
             questions_4hop: Optional[Dict] = None,
             harm_original_text: Optional[Union[str, List[str]]] = None,
             keep_original_weight=False,
             verbose=True,
             summary_metrics=False, 
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `locality_inputs`: dict
            for locality
        """

        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, yes_questions, no_questions, 
                                              locality_inputs, portability_inputs, multiple_choice_questions, reversed_relation_questions,
                                              questions_2hop, questions_3hop, questions_4hop, harm_original_text, **kwargs)
        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_edit": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name in ['IKE', 'ICL']:
                    # assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {
                        # "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                        #                                 request, self.hparams.device, pre_edit=True)
                        "pre": compute_edit_quality(self.hparams, self.model, self.tok, request, test_generation=test_generation, icl_pre_edit=True)
                    }
                else:
                    metrics = {
                        "pre": compute_edit_quality(self.hparams, self.model, self.tok, request, test_generation=test_generation)
                    }
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                ### Store the pre_edit metric to refrain computing repeatedly
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        for i, request in enumerate(requests):
            start = time()

            if self.alg_name in ['IKE', 'ICL']:
                edited_model, weights_copy = self.model, {}
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    # train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            start = time()
            if self.alg_name in ['IKE', 'ICL']:
                all_metrics[i].update({
                    'case_id': i,
                    "requested_edit": request,
                    "time": exec_time,
                    "post": compute_edit_quality(self.hparams, edited_model, self.tok, request, test_generation=test_generation, icl_pre_edit=False),
                })
            else:
                all_metrics[i].update({
                    'case_id': i,
                    "requested_edit": request,
                    "time": exec_time,
                    "post": compute_edit_quality(self.hparams, edited_model, self.tok, request, test_generation=test_generation),
                })
            if "metric_kwargs" in kwargs:
                all_metrics[i].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok, metric_kwargs=kwargs["metric_kwargs"][i], device=self.hparams.device))
            if self.alg_name == 'KN' or (self.alg_name == 'GRACE' and keep_original_weight):
                with torch.no_grad():
                    weights_copy() # unpatch_fn
            elif self.alg_name == 'LoRA' and keep_original_weight:
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            elif self.alg_name == 'LoRA' and not keep_original_weight:
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            if 'locality' in all_metrics[i]['post'].keys():
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for pre_edit_output, post_edit_output in zip(all_metrics[i]['pre']['locality'][f'{locality_key}_output'], all_metrics[i]['post']['locality'][f'{locality_key}_output']):
                        locality_result.append(locality_acc_llm(self.hparams, pre_edit_output, post_edit_output))
                    all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[i]['pre']['locality'].pop(f'{locality_key}_acc')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        if isinstance(edited_model, LORA):
            edited_model=edited_model.model
        #for melo
        
        if summary_metrics and len(all_metrics)!=0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics,]
            # logs_dir = './logs'  
            # if not os.path.exists(logs_dir):  
            #     os.makedirs(logs_dir)  
            # output_file = os.path.join(logs_dir, 'results.json')
            # with open(output_file, 'w') as f:  
            #     json.dump(all_metrics, f, ensure_ascii=False, indent=4)
            
            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["edit_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability", "yes_questions", "no_questions", "multiple_choice_questions", "reversed_relation_questions", "questions_2hop", "questions_3hop", "questions_4hop"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        # for lkey in all_metrics[0][eval][key].keys():
                        #     if lkey.endswith("acc"):
                        #         mean_metrics[eval][key][lkey] = np.mean([metric[eval][key][lkey] for metric in all_metrics])
                        for lkey in get_all_acc_keys(all_metrics):
                            metrics = [metric[eval][key][lkey] for metric in all_metrics if lkey in metric[eval][key].keys()]
                            if len(metrics) > 0:
                                mean_metrics[eval][key][lkey] = np.mean(metrics)
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
            
            print("Metrics Summary: ", mean_metrics)


        # del model_eval
        return all_metrics, edited_model, weights_copy


    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
        
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          yes_questions: Optional[Dict] = None,
                          no_questions: Optional[Dict] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          multiple_choice_questions: Optional[Dict] = None,
                          reversed_relation_questions: Optional[Dict] = None,
                          questions_2hop: Optional[Dict] = None,
                          questions_3hop: Optional[Dict] = None,
                          questions_4hop: Optional[Dict] = None,
                          harm_original_text: Union[str, List[str]] = None,
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
            'portability': {},
            'locality': {},
            'yes_questions': {},
            'no_questions': {},
            'multiple_choice_questions': {},
            'reversed_relation_questions': {},
            'questions_2hop': {},
            'questions_3hop': {},
            'questions_4hop': {},
            'harm_original_text': {}
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )

        if harm_original_text is not None:
            if isinstance(harm_original_text, str):
                harm_original_text = [harm_original_text,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'harm_original_text': harm_original_text[i],
                    }
                )

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                #     locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                # assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                # == len(requests), print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    if locality_inputs[locality_key]['prompt'][i] is not None:
                        request['locality'].update(
                            {
                                locality_key: {
                                    f'prompt': locality_inputs[locality_key]['prompt'][i],
                                    # f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                                }
                            }
                        )

        if portability_inputs is not None:
            for portability_key in portability_inputs.keys():
                if isinstance(portability_inputs[portability_key]['prompt'], str):
                    portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                    portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
                assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one portability input.....')

                for i, request in enumerate(requests):
                    if portability_inputs[portability_key]['prompt'][i] is not None:
                        request['portability'].update(
                            {
                                portability_key: {
                                    'prompt': portability_inputs[portability_key]['prompt'][i],
                                    'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                                }
                            }
                        )

        if yes_questions is not None:
            for key in yes_questions.keys():
                if isinstance(yes_questions[key]['prompt'], str):
                    yes_questions[key]['prompt'] = [yes_questions[key]['prompt'],]
                    yes_questions[key]['ground_truth'] = [yes_questions[key]['ground_truth'], ]
                assert len(yes_questions[key]['prompt']) == len(yes_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if yes_questions[key]['prompt'][i] is not None:
                        request['yes_questions'].update({key: {'prompt': yes_questions[key]['prompt'][i], 'ground_truth': yes_questions[key]['ground_truth'][i]}})

        if no_questions is not None:
            for key in no_questions.keys():
                if isinstance(no_questions[key]['prompt'], str):
                    no_questions[key]['prompt'] = [no_questions[key]['prompt'],]
                    no_questions[key]['ground_truth'] = [no_questions[key]['ground_truth'], ]
                assert len(no_questions[key]['prompt']) == len(no_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if no_questions[key]['prompt'][i] is not None:
                        request['no_questions'].update({key: {'prompt': no_questions[key]['prompt'][i],  'ground_truth': no_questions[key]['ground_truth'][i]}})

        if multiple_choice_questions is not None:
            for key in multiple_choice_questions.keys():
                if isinstance(multiple_choice_questions[key]['prompt'], str):
                    multiple_choice_questions[key]['prompt'] = [multiple_choice_questions[key]['prompt'],]
                    multiple_choice_questions[key]['ground_truth'] = [multiple_choice_questions[key]['ground_truth'], ]
                assert len(multiple_choice_questions[key]['prompt']) == len(multiple_choice_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if multiple_choice_questions[key]['prompt'][i] is not None:
                        request['multiple_choice_questions'].update({key: {'prompt': multiple_choice_questions[key]['prompt'][i], 'ground_truth': multiple_choice_questions[key]['ground_truth'][i]}})

        if reversed_relation_questions is not None:
            for key in reversed_relation_questions.keys():
                if isinstance(reversed_relation_questions[key]['prompt'], str):
                    reversed_relation_questions[key]['prompt'] = [reversed_relation_questions[key]['prompt'],]
                    reversed_relation_questions[key]['ground_truth'] = [reversed_relation_questions[key]['ground_truth'], ]
                assert len(reversed_relation_questions[key]['prompt']) == len(reversed_relation_questions[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if reversed_relation_questions[key]['prompt'][i] is not None:
                        request['reversed_relation_questions'].update({key: {'prompt': reversed_relation_questions[key]['prompt'][i], 'ground_truth': reversed_relation_questions[key]['ground_truth'][i]}})

        if questions_2hop is not None:
            for key in questions_2hop.keys():
                if isinstance(questions_2hop[key]['prompt'], str):
                    questions_2hop[key]['prompt'] = [questions_2hop[key]['prompt'],]
                    questions_2hop[key]['ground_truth'] = [questions_2hop[key]['ground_truth'], ]
                assert len(questions_2hop[key]['prompt']) == len(questions_2hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_2hop[key]['prompt'][i] is not None:
                        request['questions_2hop'].update({key: {'prompt': questions_2hop[key]['prompt'][i], 'ground_truth': questions_2hop[key]['ground_truth'][i]}})

        if questions_3hop is not None:
            for key in questions_3hop.keys():
                if isinstance(questions_3hop[key]['prompt'], str):
                    questions_3hop[key]['prompt'] = [questions_3hop[key]['prompt'],]
                    questions_3hop[key]['ground_truth'] = [questions_3hop[key]['ground_truth'], ]
                assert len(questions_3hop[key]['prompt']) == len(questions_3hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_3hop[key]['prompt'][i] is not None:
                        request['questions_3hop'].update({key: {'prompt': questions_3hop[key]['prompt'][i], 'ground_truth': questions_3hop[key]['ground_truth'][i]}})

        if questions_4hop is not None:
            for key in questions_4hop.keys():
                if isinstance(questions_4hop[key]['prompt'], str):
                    questions_4hop[key]['prompt'] = [questions_4hop[key]['prompt'],]
                    questions_4hop[key]['ground_truth'] = [questions_4hop[key]['ground_truth'], ]
                assert len(questions_4hop[key]['prompt']) == len(questions_4hop[key]['ground_truth']) == len(requests), print('One Edit instance needs one input question.....')

                for i, request in enumerate(requests):
                    if questions_4hop[key]['prompt'][i] is not None:
                        request['questions_4hop'].update({key: {'prompt': questions_4hop[key]['prompt'][i], 'ground_truth': questions_4hop[key]['ground_truth'][i]}})
        return requests


    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        keep_original_weight=False,
        epoch: int=5,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        assert BatchEditor.is_batchable_method(self.alg_name), print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls specify the batch_size....')

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=keep_original_weight,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy

