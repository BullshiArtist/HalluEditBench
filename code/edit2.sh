(
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=event_film --model_name=llama3-8b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=event_film --model_name=llama3-8b --multi_turn=sure &
python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=event_film --model_name=mistral-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=event_film --model_name=mistral-7b --multi_turn=sure &
wait
)

(
python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=event_sport --model_name=mistral-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=event_sport --model_name=mistral-7b --multi_turn=sure &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=event_history --model_name=mistral-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=7 --topic_name=event_history --model_name=mistral-7b --multi_turn=sure &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=event_sport --model_name=llama3-8b --multi_turn=yes &
wait
)

(
python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=event_film --model_name=llama2-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=event_film --model_name=llama2-7b --multn=surei_tur &
python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=event_sport --model_name=llama2-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=event_sport --model_name=llama2-7b --multi_turn=sure &
python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=event_history --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=event_history --model_name=llama2-7b --multi_turn=sure &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=event_history --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=event_history --model_name=llama3-8b --multi_turn=sure &

wait
)

# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=7 --topic_name=event_sport --model_name=llama3-8b --multi_turn=sure