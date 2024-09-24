# # python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=human_writer --model_name=mistral-7b &
# # python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_country --model_name=mistral-7b &
# # python3 edit_all_method_multi_turn.py --device_edit=5 --device_eval=7 --topic_name=places_country &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_country --model_name=mistral-7b --multi_turn_type=yes &
# python3 edit_all_method_multi_turn.py --device_edit=1 --device_eval=6 --topic_name=places_country --model_name=llama3-8b --multi_turn_type=yes &
# python3 edit_all_method_multi_turn.py --device_edit=2 --device_eval=6 --topic_name=places_country --model_name=llama2-7b --multi_turn_type=yes
# wait

topics=(
    # 'art_sculpture' 'business_brand' 'business_corporation' 'business_industry'
    # 'entertainment_anime' 'entertainment_song' 'entertainment_music_genre'
    # 'geography_glacier' 'geography_volcano' 'geography_forest'
    # 'health_disease' 'health_symptom' 'health_medication'
    # 'technology_software' 'technology_programming_language' 'technology_database'
    # 'event_sport' 'event_history' 'event_film'
    'human_athlete' 'human_writer' 'human_entrepreneur' 'human_scientist'
    'places_country' 'places_city' 'places_landmark'
)

start_time=$(date +%s)

for topic in "${topics[@]}"; do
    # python3 edit.py --hparams_dir=./hparams/GRACE/mistral-7b --topic_name="$topic" --device_edit=1 --device_eval=6 --results_dir=../tmp/all_grace
    # python3 edit_all_method.py --model_name=llama2-7b --device_edit=0 --device_eval=6 --topic_name="$topic"
    python3 edit_all_method.py --model_name=gemma-2b --device_edit=2 --device_eval=6 --topic_name="$topic"
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_minutes=$(echo "scale=2; $runtime / 60" | bc)
echo "Runtime in total: $runtime_minutes minutes"