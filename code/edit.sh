# python3 edit.py --hparams_dir=./hparams/MEMIT/llama2-7b --device_edit=0 --topic_name=technology_software --data_size=2 --results_dir=../tmp

python3 edit2.py --hparams_dir=./hparams/ICL/mistral-7b --device_edit=2 --device_eval=3
python3 edit2.py --hparams_dir=./hparams/ROME/mistral-7b --device_edit=2 --device_eval=3
python3 edit2.py --hparams_dir=./hparams/FT-M/mistral-7b --device_edit=2 --device_eval=3
python3 edit2.py --hparams_dir=./hparams/FT-L/mistral-7b --device_edit=2 --device_eval=3
python3 edit2.py --hparams_dir=./hparams/LoRA/mistral-7b --device_edit=2 --device_eval=3
python3 edit2.py --hparams_dir=./hparams/MEMIT/mistral-7b --device_edit=2 --device_eval=3
python3 edit_tmp.py --hparams_dir=./hparams/GRACE/mistral-7b --device_edit=0 --device_eval=1 --results_dir=../tmp


# python3 edit.py --hparams_dir=./hparams/ICL/mistral-7b --topic_name=human_scientist --device_edit=1 --device_eval=3
# python3 edit.py --hparams_dir=./hparams/ROME/mistral-7b --topic_name=human_scientist --device_edit=1 --device_eval=3
# python3 edit.py --hparams_dir=./hparams/FT-M/mistral-7b --topic_name=human_scientist --device_edit=1 --device_eval=3
# python3 edit.py --hparams_dir=./hparams/FT-L/mistral-7b --topic_name=human_scientist --device_edit=1 --device_eval=3
# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/mistral-7b --topic_name=technology_software --device_edit=0 --device_eval=1
# python3 edit.py --hparams_dir=./hparams/GRACE/mistral-7b --topic_name=places_country --device_edit=0 --device_eval=2 --overwrite_result --data_size=20
# python3 edit.py --hparams_dir=./hparams/GRACE/mistral-7b --device_edit=0 --device_eval=2


# python3 edit.py --hparams_dir=./hparams/SERAC/llama3-8b --topic_name=places_country --device_edit=1 --device_eval=3 --overwrite_result --data_size=20


# python3 edit_tmp.py --hparams_dir=./hparams/WISE/llama3-8b --topic_name=places_country --device_edit=1 --device_eval=3 --data_size=2
# python3 edit_tmp.py --hparams_dir=./hparams/MALMEN/llama3-8b --topic_name=places_country --device_edit=1 --device_eval=3 --data_size=2

# OutOfMemoryError
# python3 edit.py --hparams_dir=./hparams/MELO/llama3-8b --topic_name=places_country --device_edit=1 --device_eval=3 --data_size=2
# python3 edit.py --hparams_dir=./hparams/KN/llama3-8b --topic_name=places_country --device_edit=1 --device_eval=3 --data_size=2


# Tonight run
# python3 edit.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=0 --device_eval=2 --topic_name=technology_software --data_size=20