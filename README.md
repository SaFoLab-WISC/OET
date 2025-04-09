# OET: Optimization-based prompt injection Evaluation Toolkit
[Jinsheng Pan](), [Xiaogeng Liu](https://sheltonliu-n.github.io), and [Chaowei Xiao](https://xiaocw11.github.io).

## Abstract 
Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation, enabling their widespread adoption across various domains. However, their susceptibility to prompt injection attacks poses significant security risks, as adversarial inputs can manipulate model behavior and override intended instructions. Despite numerous defense strategies, a standardized framework to rigorously evaluate their effectiveness, especially under adaptive adversarial scenarios—is lacking. To address this gap, we introduce OET, an optimization-based evaluation toolkit that systematically benchmarks prompt injection attacks and defenses across diverse datasets using an adaptive testing framework. Our toolkit features a modular workflow that facilitates adversarial string generation, dynamic attack execution, and comprehensive result analysis, offering a unified platform for assessing adversarial robustness. Crucially, the adaptive testing framework leverages optimization methods with both white-box and black-box access to generate worst-case adversarial examples, thereby enabling strict red-teaming evaluations. Extensive experiments underscore the limitations of current defense mechanisms, with some models remaining susceptible even after implementing security enhancements.

## Table of Contents
- [ Update ](#Update)
- [ Installation ](#Installation)
- [ Evaluation ](#Evaluation)
    - [ Data Transformation ](#Data-Transformation)
    - [ Open-sourced Model ](#Open-sourced-Model)
    - [ Close-sourced Model ](#Close-sourced-Model)
- [ License ](#license)
- [ Citation ](#citation)
- [ Acknowledgement ](#acknowledgement)

## Update
| Date       | Event    |
|------------|----------|
| **2025/4/09** | We have released our code.  |

## Installation
```shell
    git clone https://github.com/Victor-lol/OET.git

    conda create -n oet python=3.10
    conda activate oet
    
    cd ./OET
    python3 setup.py install 
    pip install -r requirements.txt
    pip install -e. 

```

## Evaluation

### Data Transformation
an example is shown in ```example/ex_data.py```, users can transform data in their desired format. Deafult format: json, csv

### Open-sourced Model
1. Construct configuration, modify ```configs/optimizer_config.yaml ```
2. Train Adv Strig, create ```EvalOptimizerModel``` from  ```eval.open_pipeline``` as the pipeline and then call **train** function or create your own training function
3. Attack and check result, call **complete** function to run attack, and then run **check_refusal_completions** to calculate ASR. User can write their own attack and metric function using ```EvalOptimizerModel``` object

a usage example is shown in ```example/example.py ``` and ```example/train.sh ``` and ```example/infer.sh ```
an example of creating customized training and attack function is shown in ```example/eval_struq.py ```
For chat Template options, please refers to [FastChat](https://github.com/lm-sys/FastChat)

### Close-sourced Model
Directly Attack and check result, create ```EvalAPIModel``` from  ```eval.close_pipeline``` as the pipeline and then call **complete** function or create your own attack function, and call **check_refusal** to calculate ASR. 

a usage example is shown in ```example/example_close.py ``` and ```example/infer_close.sh ```

    
## Citation
If this work is helpful, please kindly cite as:

```bibtex

```

## Acknowledgement
This repo benefits from [HarmBench](https://github.com/centerforaisafety/HarmBench), [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN/tree/main), and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.

