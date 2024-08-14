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

# gemma ROME parallel Runtime: 129 minutes and 37 seconds
# Traceback (most recent call last):
#   File "/data1/baixiang/workspace/edit/hallucination/code/edit.py", line 25, in <module>
#     metrics, edited_model, _ = editor.edit(
#   File "/data1/baixiang/workspace/edit/hallucination/code/easyeditor/editors/editor.py", line 164, in edit
#     return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, **kwargs)
#   File "/data1/baixiang/workspace/edit/hallucination/code/easyeditor/editors/editor.py", line 339, in edit_requests
#     edited_model, weights_copy, icl_examples = edit_func(request)
#   File "/data1/baixiang/workspace/edit/hallucination/code/easyeditor/editors/editor.py", line 291, in edit_func
#     edited_model, weights_copy = self.apply_algo(
#   File "/data1/baixiang/workspace/edit/hallucination/code/easyeditor/models/rome/rome_main.py", line 41, in apply_rome_to_model
#     deltas = execute_rome(model, tok, request, hparams)
#   File "/data1/baixiang/workspace/edit/hallucination/code/easyeditor/models/rome/rome_main.py", line 101, in execute_rome
#     left_vector: torch.Tensor = compute_u(
#   File "/data1/baixiang/workspace/edit/hallucination/code/easyeditor/models/rome/compute_u.py", line 114, in compute_u
#     u = get_inv_cov(
# RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
