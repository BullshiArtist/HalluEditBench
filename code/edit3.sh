python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=human_writer --model_name=mistral-7b &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_country --model_name=mistral-7b &
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=places_country &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=human_writer &
wait