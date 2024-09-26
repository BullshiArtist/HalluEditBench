# python3 edit_all_method.py --model_name=llama3-8b --device_edit=6 --device_eval=7 --topic_name=places_landmark

# python3 edit_all_method.py --model_name=mistral-7b --device_edit=0 --device_eval=5 --topic_name=art_sculpture

# python3 edit_all_method.py --model_name=llama2-7b --device_edit=1 --device_eval=6 --topic_name=event_sport

# python3 edit_all_method.py --model_name=gemma-2b --device_edit=1 --device_eval=6 --topic_name=event_sport

topics=(
     'places_city' 'places_landmark' 'art_sculpture'
    # 'entertainment_anime' 'entertainment_song' 'entertainment_music_genre'
    # 'geography_glacier' 'geography_volcano' 'geography_forest'
    # 'art_sculpture' 'health_disease' 'health_symptom' 'health_medication'
    # 'technology_software' 'technology_programming_language' 'technology_database'
    # 'business_brand' 'business_corporation' 'business_industry'
    # 'event_sport' 'event_history' 'event_film'
    # 'human_athlete' 'human_writer' 'human_entrepreneur' 'human_scientist'
    # 'places_country' 'places_city' 'places_landmark'
)

# start_time=$(date +%s)

# for topic in "${topics[@]}"; do
#     python3 edit_all_method.py --model_name=llama2-7b --device_edit=0 --device_eval=7 --topic_name="$topic"
# done

# end_time=$(date +%s)
# runtime=$((end_time - start_time))
# runtime_minutes=$(echo "scale=2; $runtime / 60" | bc)
# echo "Runtime for geography_forest: $runtime_minutes minutes"


# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_country --model_name=mistral-7b --multi_turn_type=yes &
# python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=1 --topic_name=places_country --model_name=llama3-8b --multi_turn_type=yes &
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=6 --topic_name=places_country --model_name=llama2-7b --multi_turn_type=yes
# wait

python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=art_sculpture --model_name=mistral-7b --multi_turn_type=sure &
python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=art_sculpture --model_name=llama3-8b --multi_turn_type=sure &
python3 edit_all_method_multi_turn.py --device_edit=3 --device_eval=7 --topic_name=art_sculpture --model_name=llama2-7b --multi_turn_type=sure &
wait


# python3 edit_all_method_multi_turn.py --device_edit=0 --device_eval=7 --topic_name=places_country --model_name=llama3-8b --data_size=20 --multi_turn_type=yesModGrace 
