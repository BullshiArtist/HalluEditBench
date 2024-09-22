# # llama3-8b
# python3 edit_all_method.py --device_edit=6 --device_eval=7 --topic_name=places_landmark
# python3 edit_all_method.py --device_edit=7 --device_eval=5 --topic_name=entertainment_anime
# python3 edit_all_method.py --device_edit=4 --device_eval=6 --topic_name=entertainment_song
# python3 edit_all_method.py --device_edit=4 --device_eval=6 --topic_name=business_corporation
# python3 edit_all_method.py --device_edit=1 --device_eval=5 --topic_name=geography_volcano
# python3 edit_all_method.py --device_edit=3 --device_eval=5 --topic_name=technology_software
# python3 edit_all_method.py --device_edit=6 --device_eval=7 --topic_name=human_athlete
# python3 edit_all_method.py --device_edit=6 --device_eval=7 --topic_name=human_writer
# python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=event_film
# python3 edit_all_method.py --device_edit=4 --device_eval=6 --topic_name=event_history
# python3 edit_all_method.py --device_edit=7 --device_eval=6 --topic_name=event_sport
# python3 edit_all_method.py --device_edit=3 --device_eval=5 --topic_name=geography_glacier
# python3 edit_all_method.py --device_edit=6 --device_eval=7 --topic_name=human_entrepreneur
# python3 edit_all_method.py --device_edit=1 --device_eval=5 --topic_name=health_disease
# python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=business_industry
# python3 edit_all_method.py --device_edit=0 --device_eval=5 --topic_name=entertainment_music_genre
# python3 edit_all_method.py --device_edit=7 --device_eval=5 --topic_name=geography_forest
# python3 edit_all_method.py --device_edit=1 --device_eval=6 --topic_name=technology_programming_language
# python3 edit_all_method.py --device_edit=2 --device_eval=6 --topic_name=technology_database
# python3 edit_all_method.py --device_edit=3 --device_eval=6 --topic_name=art_sculpture
# python3 edit_all_method.py --device_edit=0 --device_eval=5 --topic_name=health_symptom
# python3 edit_all_method.py --device_edit=4 --device_eval=5 --topic_name=health_medication


# # mistral
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=0 --device_eval=5 --topic_name=art_sculpture
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=2 --device_eval=5 --topic_name=business_industry
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=4 --device_eval=5 --topic_name=event_history
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=7 --device_eval=6 --topic_name=entertainment_music_genre
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=1 --device_eval=6 --topic_name=event_sport
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=1 --device_eval=6 --topic_name=health_symptom
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=4 --device_eval=5 --topic_name=health_medication
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=3 --device_eval=7 --topic_name=health_disease
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=1 --device_eval=6 --topic_name=geography_forest
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=3 --device_eval=6 --topic_name=geography_glacier
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=2 --device_eval=6 --topic_name=technology_programming_language
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=2 --device_eval=6 --topic_name=human_writer
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=0 --device_eval=7 --topic_name=human_entrepreneur
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=3 --device_eval=5 --topic_name=human_athlete
# python3 edit_all_method.py --model_name=mistral-7b --device_edit=3 --device_eval=5 --topic_name=technology_database



python3 edit_all_method.py --model_name=vicuna-7b --device_edit=1 --device_eval=5 --topic_name=event_sport

python3 edit_all_method.py --model_name=llama2-7b --device_edit=1 --device_eval=6 --topic_name=event_sport
python3 edit_all_method.py --model_name=llama2-7b --device_edit=2 --device_eval=7 --topic_name=health_symptom
python3 edit_all_method.py --model_name=llama2-7b --device_edit=4 --device_eval=6 --topic_name=geography_forest
python3 edit_all_method.py --model_name=llama2-7b --device_edit=3 --device_eval=7 --topic_name=business_corporation
python3 edit_all_method.py --model_name=llama2-7b --device_edit=1 --device_eval=6 --topic_name=places_city
python3 edit_all_method.py --model_name=llama2-7b --device_edit=2 --device_eval=7 --topic_name=entertainment_music_genre


start_time=$(date +%s)
python3 edit_all_method.py --model_name=mistral-7b --device_edit=1 --device_eval=6 --topic_name=geography_forest
end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_minutes=$(echo "scale=2; $runtime / 60" | bc)
echo "Runtime for geography_forest: $runtime_minutes minutes"

# # wait
#  