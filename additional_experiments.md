# Can Knowledge Editing Really Correct Hallucinations?

**Oveview**: This file contains additional results for GPT-J-6B and Gemma-1-2B for the paper "Can Knowledge Editing Really Correct Hallucinations?"


## Statistics
<img src="results/additional_figs/stat_gemma_gptj.png" width=75%>\
Statistics of {\halluedit} Across Topics and Domains


## Efficacy
<img src="results/additional_figs/plot_efficacy_gptj_6b_domain.png" width=75%>\
Efficacy Scores for GPT-J-6B across 9 domains. The ``overall'' refers to the Efficacy Score (%) on the whole HalluEdit embracing 9 domains for different methods. The Efficacy Score on each domain is also reported. Efficacy scores (%) are measured by the accuracy on Efficacy Evaluation Question-answer Pairs, where the pre-edit scores of each LLM are ensured 0.

<img src="results/additional_figs/plot_efficacy_gemma_1_2b_domain.png" width=75%>\
Efficacy Scores for Gemma-1-2B across 9 domains.


## Generalization
<img src="results/additional_figs/plot_generalization_overall.png" width=75%>\
**Generalization Scores of Knowledge Editing Methods**. Generalization Scores (%) are measured by accuracy on five types of Generalization Evaluation Questions including Rephrased Questions ("rephrase"), Yes-or-No Questions with Yes or No as answers ("yes" or "no"), Multi-Choice Questions ("mc"), Reversed Questions ("reversed"). The "average" refers to averaged scores over five question types.


## Portability
<img src="results/additional_figs/plot_portability_gpt_j_6b.png" width=75%>\
Portability Scores of Knowledge Editing Methods for GPT-J-6B. Portability Scores (%) are measured by the accuracy on Portability Evaluation Questions, which are Efficacy Evaluation Questions with N hops (N = 1-6). The Portability Evaluation Questions are the same as Efficacy Evaluation Questions when N is 1. 
<!-- The Portability Scores on two domains "human" and "places" are reported in the figure. The results for more domains are given in Appendix: Portability Scores of Knowledge Editing Methods on All the Domains. The "overall" refers to the Portability Score (%) on the whole HalluEdit embracing 9 domains. -->
<img src="results/additional_figs/plot_portability_gemma_1_2b.png" width=75%>\
Portability Scores of Knowledge Editing Methods for Gemma-1-2B. 


## Locality
<img src="results/additional_figs/plot_locality_gptj_6b.png" width=75%>\
Locality Scores of Knowledge Editing Methods for GPT-J-6B. Locality Scores (%) are measured by the unchanging rate on Locality Evaluation Questions after applying knowledge editing methods on LLMs. A higher Locality Score indicates that there is a higher percentage of LLMs' answers to the unrelated questions keeping the same and a less side effect on general knowledge in LLMs. The "overall" refers to the Locality Score (%) on the whole HalluEdit embracing 9 domains for different methods. The Locality Score on each domain is also reported in the figure.

<img src="results/additional_figs/plot_locality_gemma_1_2b.png" width=75%>\
Locality Scores of Knowledge Editing Methods for Gemma-1-2B. 


<!-- ## Robustness -->
<!-- <img src="results/additional_figs/plot_robustness_gptj.png" width=75%>\
**Robustness Scores of Knowledge Editing Methods**. Robustness Scores are calculated by the accuracy on Robustness Evaluation Questions with M turns (M = 1-10). We regard Efficacy Scores as the Robustness Scores when M is 0. The Robustness Scores on two domains "human" and "places" are reported in the figure. The results for more domains are given in Appendix: Robustness Scores of Knowledge Editing Methods on All the Domains. The "overall" refers to the Robustness Score (%) on the whole HalluEdit embracing 9 domains. -->




