# Can Knowledge Editing Really Correct Hallucinations?

- **Respository Oveview**: This repository contains the code, results and dataset for the paper **["Can Knowledge Editing Really Correct Hallucinations?"](https://github.com/link-omitted-during-review/hallucination)**
- **TLDR**: Existing evaluations of knowledge editing often neglect the factual accuracy of LLMs before editing. To address this, HalluEdit offers a comprehensive and accurate benchmark for assessing knowledge editing and other hallucination mitigation methods.
<!-- - **Authors** :  -->


## Overview
Large Language Models (LLMs) suffer from hallucinations, which refer to the inclusion of non-factual information in generated content. In response, knowledge editing has emerged as a promising paradigm designed to correct erroneous factual knowledge encoded in LLMs, offering the advantage of avoiding retraining from scratch. However, current knowledge editing techniques are typically evaluated based solely on the performance of post-edit LLMs on question-answering datasets. A critical flaw in these datasets is that they do not ensure LLMs actually hallucinate on the evaluation questions before editing, leading to an unreliable assessment of the effectiveness of different knowledge editing techniques in addressing hallucinations. Therefore, a fundamental question remains inadequately validated: *Can knowledge editing truly correct hallucinations in LLMs?*

To address this, we propose **HalluEdit**, a holistic benchmark designed to evaluate knowledge editing methods in correcting real-world hallucinations. First, we rigorously construct a comprehensive hallucination dataset across 3 LLMs, 9 domains, and 26 topics. Then, we assess the performance of knowledge editing methods across five dimensions: *Efficacy*, *Generalization*, *Portability*, *Locality*, and *Robustness*. Through **HalluEdit**, we provide new insights into the strengths and weaknesses of different knowledge editing methods in correcting hallucinations, offering inspiration for future improvements and facilitating progress in the field of knowledge editing.


<img src="https://github.com//blob/master/data/intro.png" width=85%>


# Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Data Preparation](#data-preparation)
    2. [Running Experiments](#running-experiments)
<!-- 5. [Contributing](#contributing) -->
5. [Acknowledgements](#acknowledgements)


## Repository Structure
- `data/`: Contains the dataset.
- `code/`: Includes scripts and code to reproduce the results in the paper.
- `results/`: Results of the experiments.


## Installation
To set up the environment for running the code, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/link-omitted-during-review/hallu-edit.git
    cd hallu-edit
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n HalluEdit python=3.9
    conda activate HalluEdit
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Data Preparation

1. Datasets are stored in the `data/` directory. There are three folders:

```bash
data/
    ├── questions
    │   ├── hallucination_final
    │   │   ├── llama_2_7b_chat_hf
    │   │   ├── meta_llama_3_8b_instruct
    │   │   └── mistral_7b_instruct_v0.3
    ├── topic
    └── triplet
```

### Running Experiments

To get started (e.g. using ROME to edit llama3-8b on the places_landmark data), run:

```bash
python3 edit_all_method.py \
    --model_name=llama3-8b \
    --edit_method=ROME \
    --topic_name=places_landmark \
    --device_edit=0 \
    --device_eval=1 \
    --model_eval=meta-llama/Meta-Llama-3-8B-Instruct
```

<!-- If you use an API model (such as GPT-4) as the evaluator, you need to set your `YOUR_API_KEY` in Line 60 of `code/editor_new_eval.py`. One example is as follows: -->

We a local LLM (e.g., Llama3-8b) as the evaluator (to evalueate if model reponses match the labels). We recommend running experiments with at least one GPU with 48 GB of memory (we use NVIDIA RTX A6000 GPUs) or two GPUs with 24 GB of vRAM: one for loading the edited models (both the pre-edit and post-edit models) and one for loading the local evaluation model. Modify the device number and the evaluation model through `--model_eval` and `--device_eval` as shown in the example above:

For full experiments:
1. To run the knowledge editing experiment for all the 26 topics:
    ```bash
    ./code/edit_all_topic.sh
    ```

<!-- 2. To run the bias injection experiment:
    ```bash
    ./code/edit.sh
    ```


<!-- An OpenAI API key is required for GPT-4 evaluation. Save it in the "api_key.json" file. -->

We evaluate instruction-tuned models including `Llama-2-7B-chat`, `Llama-3-8B-Instruct`, and `Mistral-7B-v0.3`. All parameters are in the `code/hparams/<method_name>/<model_name>`. 

Results are stored at `llama_2_7b_chat_hf`, `meta_llama_3_8b_instruct`, `mistral_7b_instruct_v0` under the `results` folder.

To summarize the results, use the jupyter notebook `code/result_summary.ipynb` and `code/previous_benchmarks.ipynb`
<!-- 
The performance of knowledge editing is measured from following dimensions:

- `Efficacy`: whether the edited models could recall the exact editing fact under editing prompts
- `Generalization`: whether the edited models could recall the editing fact under paraphrase prompts
- `Locality`: whether the output of the edited models for inputs out of editing scope remains unchanged after editing
- `Additivity`: the degree of perturbation to neighboring knowledge when appending. -->


<!-- ## Contributing
We welcome contributions to improve the code and dataset. Please open an issue or submit a pull request if you have any suggestions or improvements. -->


## License
This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). 


## Ethics Statement



## Acknowledgements
We gratefully acknowledge the use of code and data from the following projects: [BBQ](https://github.com/nyu-mll/BBQ), [BoolQ](https://github.com/google-research-datasets/boolean-questions), [GSM8K](https://github.com/openai/grade-school-math), [EasyEdit](https://github.com/zjunlp/EasyEdit),[Natural Questions](https://github.com/google-research-datasets/natural-questions), [NLI](https://nlp.stanford.edu/projects/snli/), [ROME](https://github.com/kmeng01/rome)
<!-- [IKE]() -->

<!-- ## Citation
If you find our paper or code useful, we will greatly appreacite it if you could consider citing our paper:
```

``` -->

<!-- Please note that we do not have ownership of the data and therefore cannot provide a license or control its use. However, we kindly request that the data only be used for research purposes. -->
