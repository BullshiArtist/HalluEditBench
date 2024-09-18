# Block 1: places_country
(
python3 edit.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=0 --device_eval=3 --topic_name=places_country
python3 edit.py --hparams_dir=./hparams/ICL/llama3-8b --device_edit=0 --device_eval=3 --topic_name=places_country
python3 edit.py --hparams_dir=./hparams/ROME/llama3-8b --device_edit=0 --device_eval=3 --topic_name=places_country
python3 edit.py --hparams_dir=./hparams/FT-M/llama3-8b --device_edit=0 --device_eval=3 --topic_name=places_country
python3 edit.py --hparams_dir=./hparams/FT-L/llama3-8b --device_edit=0 --device_eval=3 --topic_name=places_country
python3 edit.py --hparams_dir=./hparams/GRACE/llama3-8b --device_edit=0 --device_eval=3 --topic_name=places_country
python3 edit.py --hparams_dir=./hparams/LoRA/llama3-8b --device_edit=0 --device_eval=3 --topic_name=places_country
) &

# Block 2: places_city
(
python3 edit.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=1 --device_eval=5 --topic_name=places_city
python3 edit.py --hparams_dir=./hparams/ICL/llama3-8b --device_edit=1 --device_eval=5 --topic_name=places_city
python3 edit.py --hparams_dir=./hparams/ROME/llama3-8b --device_edit=1 --device_eval=5 --topic_name=places_city
python3 edit.py --hparams_dir=./hparams/FT-M/llama3-8b --device_edit=1 --device_eval=5 --topic_name=places_city
python3 edit.py --hparams_dir=./hparams/FT-L/llama3-8b --device_edit=1 --device_eval=5 --topic_name=places_city
python3 edit.py --hparams_dir=./hparams/GRACE/llama3-8b --device_edit=1 --device_eval=5 --topic_name=places_city
python3 edit.py --hparams_dir=./hparams/LoRA/llama3-8b --device_edit=1 --device_eval=5 --topic_name=places_city
) &

wait