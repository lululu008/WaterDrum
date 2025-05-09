# Evaluating Unlearning in Generative Models via Watermarking

WaterDrum is the first data-centric unlearning metric, which applies robust text watermarking to overcome these limitations in utility-centric unlearning metrics. This is the official implementation of the paper "WaterDrum: Watermarking for Data-centric Unlearning Metric"


## Links

- [**arXiv Paper**](https://arxiv.org/abs/2505.05064): Detailed discussion on the desiderata for unlearning metrics, introduction of our WaterDrum metric, and the empirical experiments on WaterDrum-TOFU and WaterDrum-Ax datasets.
- [**WaterDrum-TOFU**](https://huggingface.co/datasets/Glow-AI/WaterDrum-TOFU): The TOFU corpus of WaterDrum, which comprises unwatermarked and watermarked question-answering datasets.
- [**WaterDrum-Ax**](https://huggingface.co/datasets/Glow-AI/WaterDrum-Ax): The arxiv corpus of WaterDrum, which comprises unwatermarked and watermarked ArXiv paper abstracts.


## Initialize submodules and install requirements
```
git submodule init
git submodule update
pip install -r requirements.txt
```


## How to run

We provide example scripts on TOFU and arXiv dataset in [`scripts/<dataset_name>`](./scripts).
To evaluate the watermarking strength on different subsets after applying an unlearning algorithm, the original training/finetuning dataset must be watermarked, e.g. using the Waterfall text watermarking framework.

We publish the watermarked and original (unwatermarked) datasets on huggingface: [[WaterDrum-TOFU](https://huggingface.co/datasets/Glow-AI/WaterDrum-TOFU)] | [[WaterDrum-Ax](https://huggingface.co/datasets/Glow-AI/WaterDrum-Ax)].

To run the experiments on the watermarked datasets, example scripts can be found for:

**TOFU:**
1. Finetuning the pretrained model on the watermarked TOFU dataset: `scripts/tofu/train.sh`
2. Running an unlearning algorithm: `scripts/tofu/unlearn.sh`
3. Running evaluation on an unlearned model, either using metrics from [the original TOFU paper](https://arxiv.org/abs/2401.06121) or using our watermarking strength:
    - TOFU's metrics: `scripts/tofu/run_eval_tofu.sh`
    - Watermarking strength: `scripts/tofu/run_wtm_eval.sh`

**Arxiv：**
1. Finetuning the pretrained model on the watermarked Arxiv dataset: `scripts/arxiv/train.sh`
2. Running an unlearning algorithm: `scripts/arxiv/unlearn.sh`
3. Running evaluation on an unlearned model, using baseline metrics and our watermarking strength:
   -  Baseline metrics: `scripts/arxiv/run_eval_arxiv.sh`
   -  Watermarking strength: `scripts/arxiv/run_wtm_eval.sh`
  
## Ablations

To unlearn and evaluate different percentages of forget set, change `--data_config_path` in the training, unlearning, and evaluation scripts accordingly. Example configs can be found for:
   -  TOFU: unlearn different percentages of data: `config/tofu/data_forget01.yaml`, `config/tofu/data_forget05.yaml`, and `config/tofu/data_forget10.yaml`
   -  Arxiv unlearn different number of classes: `config/arxiv/data_forget01.yaml`, `config/arxiv/data_forget03.yaml`, and `config/arxiv/data_forget05.yaml`

To run the experiments on unwatermarked datasets, load the unwatermarked datasets by changing `--data_config_path` in the scripts accordingly. Example configs can be found for:
   -  TOFU unwatermarked: `config/tofu/data_unwatermarked_forget01.yaml`
   -  Arxiv unwatermarked: `config/arxiv/data_unwatermarked_forget01.yaml`

### Calibration

We aim to analyse the calibration of the proposed metric by varying the percentage of the forget set. An ideal metric should monotonically increase with larger forget sets.
To run calibration experiments, change `--data_config_path` in the scripts to config different calibration percentages accordingly. Example configs can be found for:

   -  TOFU: `config/tofu/calibration/data_calibration_1pct.yaml`
   -  Arxiv: `config/arxiv/calibration/data_calibration_1pct.yaml`

### Duplicate

We also assess the robustness of the proposed metrics to similar data. We consider exact duplicates and semantic duplicates. To run the experiments, load the duplicate datasets and semantic duplicate datasets accordingly by changing `--data_config_path` in the scripts. Example configs can be found for:

   -  Exact duplicate: `config/arxiv/data_forget01_dup01.yaml`
   -  Semantic duplicate: `config/arxiv/data_forget01_semdup01.yaml`

Similarly, to run calibration experiments with duplicates, change `--data_config_path` in the scripts to config different calibration percentages accordingly. Example configs can be found for:

   -  Exact duplicate: `config/arxiv/calibration_duplicate/data_calibration_1pct.yaml`
   -  Semantic duplicate: `config/arxiv/calibration_semantic-duplicate/data_calibration_1pct.yaml`
