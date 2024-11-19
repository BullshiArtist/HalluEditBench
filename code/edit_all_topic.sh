topics=(
    'art_sculpture' 'business_brand' 'business_industry' 'business_corporation' 
    'entertainment_anime' 'entertainment_song' 'entertainment_music_genre'
    'geography_glacier' 'geography_volcano' 'geography_forest'
    'health_disease' 'health_symptom' 'health_medication'
    'technology_software' 'technology_programming_language' 'technology_database'
    'event_sport' 'event_history' 'event_film'
    'human_athlete' 'human_writer' 'human_entrepreneur' 'human_scientist'
    'places_country' 'places_city' 'places_landmark'
)

start_time=$(date +%s)

# mistral-7b
(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=places_country --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=places_city --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=places_landmark --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=human_athlete --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=human_writer --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=human_entrepreneur --model_name=mistral-7b
wait
)

(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=human_scientist --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=technology_database --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=geography_volcano --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=geography_forest --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=technology_software --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=technology_programming_language --model_name=mistral-7b
wait
)

(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=business_corporation --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=entertainment_anime --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=entertainment_song --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=entertainment_music_genre --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=business_brand --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=business_industry --model_name=mistral-7b
wait
)

(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=health_disease --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=health_symptom --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=health_medication --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=event_sport --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=event_history --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=event_film --model_name=mistral-7b
wait
)


# llama2-7b
(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=business_corporation --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=entertainment_anime --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=entertainment_song --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=entertainment_music_genre --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=business_brand --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=business_industry --model_name=llama2-7b
wait
)

(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=places_country --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=places_city --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=places_landmark --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=human_athlete --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=human_writer --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=human_entrepreneur --model_name=llama2-7b
wait
)

(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=human_scientist --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=technology_database --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=geography_volcano --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=geography_forest --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=technology_software --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=technology_programming_language --model_name=llama2-7b
wait
)

(
python3 edit_all_method.py --device_edit=0 --device_eval=6 --topic_name=health_disease --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=health_symptom --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=health_medication --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=event_sport --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=event_history --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=event_film --model_name=llama2-7b
wait
)


(
python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=art_sculpture --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=3 --device_eval=7 --topic_name=art_sculpture --model_name=mistral-7b &
python3 edit_all_method.py --device_edit=4 --device_eval=7 --topic_name=geography_glacier --model_name=llama2-7b &
python3 edit_all_method.py --device_edit=5 --device_eval=7 --topic_name=geography_glacier --model_name=mistral-7b
wait
)


# # If you have multiple GPUs, you can run experiments for multiple LLMs in parallel. Specify `--results_dir` 
# # to save the results to a specific directory, otherwise the default directory is where we save the results that we report in the paper.
# for topic in "${topics[@]}"; do
#     python3 edit_all_method.py --model_name=llama2-7b --device_edit=0 --device_eval=3 --topic_name="$topic" --results_dir=../tmp &
#     python3 edit_all_method.py --model_name=llama3-8b --device_edit=1 --device_eval=3 --topic_name="$topic" --results_dir=../tmp &
#     python3 edit_all_method.py --model_name=mistral-7b --device_edit=2 --device_eval=3 --topic_name="$topic" --results_dir=../tmp &
#     wait
# done

# # Otherwise, you can run experiments for one LLM at a time.
# # for topic in "${topics[@]}"; do
# #     python3 edit_all_method.py --model_name=llama2-7b --device_edit=0 --device_eval=1 --topic_name="$topic"
# #     # python3 edit_all_method.py --model_name=llama3-8b --device_edit=0 --device_eval=1 --topic_name="$topic"
# #     # python3 edit_all_method.py --model_name=mistral-7b --device_edit=0 --device_eval=1 --topic_name="$topic"
# # done

end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_minutes=$(echo "scale=2; $runtime / 60" | bc)
echo "Runtime in total: $runtime_minutes minutes"