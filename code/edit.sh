# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=4 --device_eval=5 --topic_name=places_country --data_size=20 --data_size=2 --results_dir=../tmp

# python3 edit.py --hparams_dir=./hparams/ICL/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/ROME/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/FT-M/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/FT-L/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/LoRA/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/GRACE/llama3-8b --device_edit=4 --device_eval=5  --topic_name=business_brand


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


# Tonight run   

# sleep for 20 minutes
sleep 1200



# multi-turn
python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=human_writer --results_dir=../tmp
