# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=4 --device_eval=5 --topic_name=places_country --data_size=20 --data_size=2 --results_dir=../tmp

# python3 edit.py --hparams_dir=./hparams/ICL/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/ROME/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/FT-M/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/FT-L/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/LoRA/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand
# python3 edit.py --hparams_dir=./hparams/MEMIT/llama3-8b --device_edit=4 --device_eval=5 --topic_name=business_brand --data_size=20 --results_dir=../tmp
# python3 edit.py --hparams_dir=./hparams/GRACE/llama3-8b --device_edit=4 --device_eval=5  --topic_name=business_brand


# python3 edit_all_topic.py --model_name=llama3-8b --device_edit=3 --device_eval=7 --edit_method=GRACE --results_dir=../tmp/all_grace
# python3 edit_all_topic.py --model_name=mistral-7b --device_edit=0 --device_eval=7 --edit_method=GRACE --results_dir=../tmp/all_grace



# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/vicuna-7b --device_edit=2 --device_eval=4 --data_size=5 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/gemma-2b --device_edit=0 --device_eval=7 --data_size=1 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/gemma2-9b --device_edit=0 --device_eval=7 --data_size=1 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/ROME/gemma2-9b --device_edit=0 --device_eval=7 --data_size=1 --results_dir=../tmp  # normal
# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/gpt-j-6b --device_edit=2 --device_eval=7 --data_size=1 --results_dir=../tmp

# python3 edit_tmp.py --hparams_dir=./hparams/GRACE/llama3-8b --device_edit=1 --device_eval=5 --results_dir=../tmp --data_size=20


# python3 edit.py --hparams_dir=./hparams/SERAC/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --overwrite_result --data_size=20
# python3 edit_tmp.py --hparams_dir=./hparams/WISE/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2
# python3 edit_tmp.py --hparams_dir=./hparams/MALMEN/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2


# # Can run
# python3 edit_tmp.py --model_name=gemma-2b --device_edit=1 --device_eval=7 --data_size=1 --results_dir=../tmp
# python3 edit_tmp.py --model_name=alpaca-7b --device_edit=1 --device_eval=7 --data_size=2 --results_dir=../tmp


# # OutOfMemoryError
# python3 edit.py --hparams_dir=./hparams/MELO/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2
# python3 edit.py --hparams_dir=./hparams/KN/llama3-8b --topic_name=places_country --device_edit=0 --device_eval=1 --data_size=2
# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/llama2-13b --device_edit=0 --device_eval=7 --data_size=2 --results_dir=../tmp
# python3 edit_tmp.py --hparams_dir=./hparams/MEMIT/gemma-7b --device_edit=2 --device_eval=6 --data_size=1 --results_dir=../tmp


# Tonight run

# sleep for 20 minutes
# sleep 1200


# multi-turn
# python3 edit_all_method_multi_turn.py --device_edit=4 --device_eval=7 --topic_name=human_writer --results_dir=../tmp


topics=(
    'art_sculpture' 'business_brand' 'business_corporation' 'business_industry'
    'entertainment_anime' 'entertainment_song' 'entertainment_music_genre'
    'geography_glacier' 'geography_volcano' 'geography_forest'
    'health_disease' 'health_symptom' 'health_medication'
    'technology_software' 'technology_programming_language' 'technology_database'
    'event_sport' 'event_history' 'event_film'
    'human_athlete' 'human_writer' 'human_entrepreneur' 'human_scientist'
    'places_country' 'places_city' 'places_landmark'
)

start_time=$(date +%s)

for topic in "${topics[@]}"; do
    # python3 edit.py --hparams_dir=./hparams/GRACE/mistral-7b --topic_name="$topic" --device_edit=1 --device_eval=6 --results_dir=../tmp/all_grace
    python3 edit.py --hparams_dir=./hparams/MEMIT/gemma-2b --topic_name="$topic" --device_edit=0 --device_eval=7
    # python3 edit_all_method.py --model_name=llama2-7b --device_edit=0 --device_eval=6 --topic_name="$topic"
    # python3 edit_all_method.py --model_name=gemma-2b --device_edit=2 --device_eval=6 --topic_name="$topic"
done

end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_minutes=$(echo "scale=2; $runtime / 60" | bc)
echo "Runtime in total: $runtime_minutes minutes"