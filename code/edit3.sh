(
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=6 --topic_name=places_country --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=art_sculpture --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_landmark --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=places_city --model_name=mistral-7b --editing_method=GRACE --overwrite_result
wait
)

(
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=6 --topic_name=places_country --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=art_sculpture --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_landmark --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=places_city --model_name=llama3-8b --editing_method=GRACE --overwrite_result
wait
)

(
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=6 --topic_name=places_country --model_name=llama2-7b --editing_method=GRACE --overwrite_result &
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=art_sculpture --model_name=llama2-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_landmark --model_name=llama2-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=places_city --model_name=llama2-7b --editing_method=GRACE --overwrite_result
# wait
)

