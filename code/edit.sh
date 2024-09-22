# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=4 --device_eval=5 --topic_name=places_country --data_size=20 --data_size=2 --results_dir=../tmp

# python3 edit.py --hparams_dir=./hparams/ICL/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/ROME/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/FT-M/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/FT-L/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/LoRA/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/GRACE/llama3-8b --device_edit=4 --device_eval=5  --topic_name=business_brand


# python3 edit_tmp.py --hparams_dir=./hparams/GRACE/vicuna-7b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/ICL/vicuna-7b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/FT-M/vicuna-7b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/ROME/vicuna-7b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/FT-L/vicuna-7b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/LoRA/vicuna-7b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp

python3 edit_all_topic.py --model_name=llama3-8b --device_edit=3 --device_eval=7 --edit_method=GRACE --results_dir=../tmp/all_grace
python3 edit_all_topic.py --model_name=mistral-7b --device_edit=2 --device_eval=5 --edit_method=GRACE --results_dir=../tmp/all_grace



python3 edit_tmp.py --hparams_dir=./hparams/LoRA/vicuna-7b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp

python3 edit_tmp.py --model_name=alpaca-7b --device_edit=1 --device_eval=7 --data_size=2 --results_dir=../tmp

python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/vicuna-7b --device_edit=2 --device_eval=4 --data_size=5 --results_dir=../tmp
python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/gemma2-9b --device_edit=2 --device_eval=0 --data_size=7 --results_dir=../tmp

python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=1 --device_eval=7 --data_size=20 --results_dir=../tmp

python3 edit_tmp.py --hparams_dir=./hparams/GRACE/llama3-8b --device_edit=1 --device_eval=5 --results_dir=../tmp --data_size=20

 --data_size=20

python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/gemma2-9b --device_edit=2 --device_eval=7 --data_size=2 --results_dir=../tmp

# python3 edit_tmp.py --hparams_dir=./hparams/GRACE/gemma2-9b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/ICL/gemma2-9b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/FT-M/gemma2-9b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/ROME/gemma2-9b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/FT-L/gemma2-9b --device_edit=1 --device_eval=7 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/LoRA/gemma2-9b --device_edit=0 --device_eval=7 --data_size=5 --results_dir=../tmp


# python3 edit.py --hparams_dir=./hparams/ICL/mistral-7b --topic_name=event_film --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/ROME/mistral-7b --topic_name=event_film --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/FT-M/mistral-7b --topic_name=event_film --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/FT-L/mistral-7b --topic_name=event_film --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/MEMIT/mistral-7b --topic_name=event_film --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/GRACE/mistral-7b --topic_name=event_film --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/LoRA/mistral-7b --topic_name=event_film --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/GRACE/mistral-7b --topic_name=business_corporation --device_edit=6 --device_eval=7


# python3 edit.py --hparams_dir=./hparams/SERAC/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --overwrite_result --data_size=20


# python3 edit_tmp.py --hparams_dir=./hparams/WISE/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2
# python3 edit_tmp.py --hparams_dir=./hparams/MALMEN/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2

# OutOfMemoryError
# python3 edit.py --hparams_dir=./hparams/MELO/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2
# python3 edit.py --hparams_dir=./hparams/KN/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2
# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/llama2-13b --device_edit=0 --device_eval=7 --data_size=2 --results_dir=../tmp


# Tonight run   

# sleep for 20 minutes
# sleep 1200



# multi-turn
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=human_writer --results_dir=../tmp
