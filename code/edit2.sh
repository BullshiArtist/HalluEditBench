# (
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=event_film --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=event_sport --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=event_history --model_name=llama3-8b --multi_turn=yes &
# wait
# )

# (
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=event_film --model_name=mistral-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=event_sport --model_name=mistral-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=event_history --model_name=mistral-7b --multi_turn=yes &
# wait
# )

# (
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=event_film --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=event_sport --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=event_history --model_name=llama2-7b --multi_turn=yes &
# wait
# )

# (
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=places_landmark --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_landmark --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=places_landmark --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=places_landmark --model_name=mistral-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=places_landmark --model_name=mistral-7b --multi_turn=yes &
# wait
# )

# # (

# # wait
# # )

# (
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=places_landmark --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_landmark --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=places_landmark --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=places_landmark --model_name=mistral-7b --multi_turn=yes &

# wait
# )


# (
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=business_brand --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=business_corporation --model_name=llama3-8b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=business_industry --model_name=llama3-8b --multi_turn=yes &
# wait
# )

# (
# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=business_brand --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=business_corporation --model_name=llama2-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=business_industry --model_name=llama2-7b --multi_turn=yes &
# wait
# )

# (
# python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=7 --topic_name=business_brand --model_name=mistral-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=business_corporation --model_name=mistral-7b --multi_turn=yes &
# python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=business_industry --model_name=mistral-7b --multi_turn=yes &
# wait
# )


# (
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=entertainment_anime --model_name=llama2-7b --multi_turn=yes --editing_method=ROME --overwrite_result&
# python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=technology_programming_language --model_name=llama2-7b --multi_turn=yes --editing_method=ROME --overwrite_result
# wait
# )

# GRACE fixed: human, art, places

topics=(
    # 'business_brand' 'business_corporation' 'business_industry'
    # 'entertainment_anime' 'entertainment_song' 'entertainment_music_genre'
    'geography_glacier' 'geography_volcano' 'geography_forest'
    'health_disease' 'health_symptom' 'health_medication'
    'technology_software' 'technology_programming_language' 'technology_database'
    'event_sport' 'event_history' 'event_film'
    # 'human_athlete' 'human_writer' 'human_entrepreneur' 'human_scientist'
    # 'art_sculpture' 'places_country' 'places_city' 'places_landmark'
)

(
python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=business_brand --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=business_corporation --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=business_industry --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=entertainment_anime --model_name=mistral-7b --editing_method=GRACE --overwrite_result
python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=entertainment_song --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=entertainment_music_genre --model_name=mistral-7b --editing_method=GRACE --overwrite_result &
wait
)

(
python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=business_brand --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=business_corporation --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=business_industry --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=entertainment_anime --model_name=llama3-8b --editing_method=GRACE --overwrite_result
python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=entertainment_song --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=entertainment_music_genre --model_name=llama3-8b --editing_method=GRACE --overwrite_result &
wait
)

(
python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=6 --topic_name=business_brand --model_name=llama2-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=business_corporation --model_name=llama2-7b --editing_method=GRACE --overwrite_result
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=business_industry --model_name=llama2-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=entertainment_anime --model_name=llama2-7b --editing_method=GRACE --overwrite_result &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=entertainment_song --model_name=llama2-7b --editing_method=GRACE --overwrite_result &
python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=entertainment_music_genre --model_name=llama2-7b --editing_method=GRACE --overwrite_result
wait
)

