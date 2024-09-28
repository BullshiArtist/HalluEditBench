(
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=6 --topic_name=entertainment_song --model_name=llama3-8b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=entertainment_anime --model_name=llama3-8b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=6 --topic_name=entertainment_music_genre --model_name=llama3-8b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=7 --topic_name=event_history --model_name=llama3-8b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=entertainment_anime --model_name=mistral-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=6 --device_eval=7 --topic_name=entertainment_song --model_name=mistral-7b --multi_turn=yes &
wait
)

(
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=entertainment_music_genre --model_name=mistral-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=event_history --model_name=llama3-8b --multi_turn=yes
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=entertainment_anime --model_name=llama2-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=entertainment_song --model_name=llama2-7b --multi_turn=yes &
python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=entertainment_music_genre --model_name=llama2-7b --multi_turn=yes &
wait
)



