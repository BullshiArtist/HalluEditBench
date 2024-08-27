// ZsRE
python run_knowedit_llama2_new_eval.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'

python run_knowedit_llama2_new_eval.py --editing_method=FT-L --hparams_dir=./hparams/FT-L/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'

python run_knowedit_llama2_new_eval.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre' --train_data_path=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json

python run_knowedit_llama2_new_eval.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'
    
python run_knowedit_llama2_new_eval.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'

python run_knowedit_llama2_new_eval.py --editing_method=MEND --hparams_dir=./hparams/MEND/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre' --ds_size=2 --metrics_save_dir=../tmp
    
python run_knowedit_llama2_new_eval.py --editing_method=SERAC --hparams_dir=./hparams/SERAC/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'

python run_knowedit_llama2_new_eval.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json --datatype='zsre'


// WikiRecent
python run_knowedit_llama2_new_eval.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'

python run_knowedit_llama2_new_eval.py --editing_method=FT-L --hparams_dir=./hparams/FT-L/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'

python run_knowedit_llama2_new_eval.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent' --train_data_path=../data/KnowEdit/benchmark/wiki_recent/recent_test.json

python run_knowedit_llama2_new_eval.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'
    
python run_knowedit_llama2_new_eval.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'

python run_knowedit_llama2_new_eval.py --editing_method=MEND --hparams_dir=./hparams/MEND/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent' --ds_size=2 --metrics_save_dir=../tmp
    
python run_knowedit_llama2_new_eval.py --editing_method=SERAC --hparams_dir=./hparams/SERAC/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent' --ds_size=2

python run_knowedit_llama2_new_eval.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_recent/recent_test.json --datatype='recent'


// Wikibio
python run_knowedit_llama2_new_eval.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2_new_eval.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2_new_eval.py --editing_method=FT-L --hparams_dir=./hparams/FT-L/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2_new_eval.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2_new_eval.py --editing_method=MEND --hparams_dir=./hparams/MEND/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'
    
python run_knowedit_llama2_new_eval.py --editing_method=SERAC --hparams_dir=./hparams/SERAC/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'

python run_knowedit_llama2_new_eval.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio' --train_data_path=../data/KnowEdit/benchmark/ZsRE/ZsRE-test-all.json

python run_knowedit_llama2_new_eval.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/WikiBio/wikibio-test-all.json --datatype='wikibio'


// Counter fact
python run_knowedit_llama2_new_eval.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2_new_eval.py --editing_method=MEMIT --hparams_dir=./hparams/MEMIT/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2_new_eval.py --editing_method=FT --hparams_dir=./hparams/FT-M/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2_new_eval.py --editing_method=MEND --hparams_dir=./hparams/MEND/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2_new_eval.py --editing_method=SERAC --hparams_dir=./hparams/SERAC/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'

python run_knowedit_llama2_new_eval.py --editing_method=IKE --hparams_dir=./hparams/IKE/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact' --train_data_path=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json

python run_knowedit_llama2_new_eval.py --editing_method=LoRA --hparams_dir=./hparams/LoRA/llama2-7b \
    --data_dir=../data/KnowEdit/benchmark/wiki_counterfact/test_cf.json --datatype='counterfact'




start_time=$(date +%s)
python edit.py
end_time=$(date +%s)
duration1=$((end_time - start_time))

# start_time=$(date +%s)
# python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --ds_size=100 --metrics_save_dir=../tmp_local_eval
# end_time=$(date +%s)
# duration2=$((end_time - start_time))

# start_time=$(date +%s)
# python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --ds_size=100 --metrics_save_dir=../tmp_local_eval
# end_time=$(date +%s)
# duration3=$((end_time - start_time))

echo "Runtime: $((duration1 / 60)) minutes and $((duration1 % 60)) seconds"
# echo "Runtime: $((duration2 / 60)) minutes and $((duration2 % 60)) seconds"
# echo "Runtime: $((duration3 / 60)) minutes and $((duration3 % 60)) seconds"


# ROME mistral parallel Runtime: 1 minutes and 53 seconds
# ROME mistral No parallel Runtime: 1 minutes and 47 seconds

