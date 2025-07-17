# DiMA: Diffusion on Language Model Encodings for Protein Sequence Generation (ICML'25)

<div align="center">
  Viacheslav Meshchaninov<sup>*</sup> &emsp; <b>&middot;</b> &emsp;
  Pavel Strashnov<sup>*</sup> &emsp; <b>&middot;</b> &emsp;
  Andrey Shevtsov<sup>*</sup> &emsp; <b>&middot;</b> &emsp;
  Fedor Nikolaev
  <br>
  Nikita Ivanisenko &emsp; <b>&middot;</b> &emsp;
  Olga Kardymon &emsp; <b>&middot;</b> &emsp;
  Dmitry Vetrov
  <br>
  <span><sup>*</sup><small><em>core contributor</em></small></span>
  <br>
  <a href="https://openreview.net/pdf?id=xB9eROwBCB" target="_blank">Paper</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://meshchaninovviacheslav.github.io/DiMA/" target="_blank">Project&nbsp;Page</a>
</div>

<br><br>

<div align="center">
    <img width="600" alt="main_figure" src="assets/main_figure.png"/>
</div>

<br><br>

## Abstract

Protein sequence design has seen significant advances through discrete diffusion and autoregressive approaches, yet the potential of continuous diffusion remains underexplored. Here, we present **DiMA**, a latent diffusion framework that operates on protein language model representations. Through systematic exploration of architectural choices and diffusion components, we develop a robust methodology that generalizes across multiple protein encoders ranging from 8M to 3B parameters. We demonstrate that our framework achieves consistently high performance across sequence-only (ESM-2, ESMc), dual-decodable (CHEAP), and multimodal (SaProt) representations using the same architecture and training approach. We conduct extensive evaluation of existing methods alongside **DiMA** using multiple metrics across two protein modalities, covering quality, diversity, novelty, and distribution matching of generated proteins. **DiMA** consistently produces novel, high-quality and diverse protein sequences and achieves strong results compared to baselines such as autoregressive, discrete diffusion and flow matching language models. The model demonstrates versatile functionality, supporting conditional generation tasks including protein family-generation, motif scaffolding and infilling, and fold-specific sequence design, despite being trained solely on sequence data. This work provides a universal continuous diffusion framework for protein sequence generation, offering both architectural insights and practical applicability across various protein design scenarios.


## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/MeshchaninovViacheslav/DiMA.git
```

Create the environment for the repository:

```bash
conda env create --file environment.yaml
conda activate dima_env
```

## Unconditional Generation

1- Prepare the config (`src/configs/config.yaml`): \
    a- Change the `project.path` to your repository location. \
    b- Change the `defaults.encoder` field. There are 3 possible DiMA models, based on the encoder used: \
        * `esm2`: ESM2-3B, sequence only encoder \
        * `cheap`: CHEAP, dual-decodable (sequence and structure) encoder \
        * `saprot`: SaProt-35M, multimodel encoder (strucutre is represented with structure tokens)

2- Run the generation code.
You can find and example run in the `example.ipynb`. 

```python
import torch
from src.diffusion.dima import DiMAModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiMAModel(config_path="../configs", device=device)
model.load_pretrained()

sequences = model.generate_samples(num_texts=10)
```

## In case of new training:

### Data Preparation

DiMA was trained and evaluated on two distinct datasets, each selected for specific properties that facilitate robust model development and large-scale evaluation.

* **SwissProt**: This is a high-quality, manually curated subset of the UniProt database. Its relatively small size and high-quality annotations make it an excellent choice for initial proof-of-concept studies and detailed component analyses. For our experiments, sequences were filtered to a length between 128 and 254 amino acids.

* **AFDBv4-90**: This is a large-scale dataset derived from UniRef50, containing 2.2 million protein sequences. It is specifically curated to ensure high structural quality and diversity by including only proteins with an average predicted pLDDT score above 90 and sequence identity below 50%. This dataset is ideal for evaluating the scalability and performance of the model across diverse protein representation spaces.

#### Downloading the Data

The datasets are available on the Hugging Face Hub. First, you must specify the main configuration file, `src/configs/config.yaml`: \ 
1- Set the `name` field under the `defaults.datasets` section to one of the following:

* **`afdb`**: [https://huggingface.co/datasets/bayes-group-diffusion/AFDB-v2](https://huggingface.co/datasets/bayes-group-diffusion/AFDB-v2)
* **`swissprot`**: [https://huggingface.co/datasets/bayes-group-diffusion/swissprot](https://huggingface.co/datasets/bayes-group-diffusion/swissprot)

2- set `path` under the `project` section to the propriate path \

After configuring the dataset name, run the following script. It will automatically download the specified dataset from the Hugging Face Hub and save it to the path defined in your configuration.

```bash
python -m src.datasets.load_hub \
    --config_path="../configs" \
    --load_from_hub \
    --group_name="bayes-group-diffusion"
```

#### Prepare Length Distribution

Sequence length determination is crucial for protein generation inference. Our model, DiMA, focuses solely on semantic tokens during training via an attention mask, avoiding detrimental padding tokens. For inference, target sequence length is sampled from the training data distribution to ensure realistic protein lengths. This is followed by sampling a Gaussian vector, iterative refinement to generate 

```bash
python -m src.helpers.prepare_length_distribution \
    --config_path="../configs"
```

### Training

The training process for DiMA is structured in three main stages. This ensures that the latent space is properly prepared and the model components are optimized for generating high-fidelity protein sequences.

#### Stage 1: Pre-calculation of Normalization Statistics

Before training the diffusion model, we first calculate the mean and variance statistics of the protein language model representations across the training dataset.

This step is crucial for adapting the discrete protein data to our continuous Gaussian diffusion model. The calculated statistics are used to apply dimension-wise normalization, transforming the latent representations to have a zero mean and unit variance.
The default save path for the statistics is configured in the `config.statistics_path` field within your encoder configuration file (e.g., `src/configs/encoder/esm2.yaml`).

To calculate and save these statistics, run the following command:


```bash
python -m src.preprocessing.calculate_statistics \
    --config_path="../configs"
```

#### Stage 2: Decoder Training

As demonstrated in our work, fine-tuning the decoder for amino acid reconstruction from latent representations can significantly improve the accuracy of sequence generation during inference. This stage trains the decoder to be more resilient to minor deviations in the latents produced by the diffusion model.

This step can be skipped if the default pre-trained decoder from the encoder model (e.g., the `lm_head`) is sufficient for your application.

You can configure the decoder architecture in `src/configs/config.yaml` by setting the decoder parameter:

- `default`: Uses the decoder from the encoder's language model head (`lm_head`).
- `transformer`: Implements a more complex transformer-based decoder. Its parameters can be configured in `src/configs/decoder/transformer.yaml`.

To launch the decoder training, run the following command:

```bash
python -m src.preprocessing.train_decoder \
    --config_path="../configs"
```

#### Stage 3: Diffusion Training

The final stage is the training of the denoising diffusion model itself. DiMA is a continuous-time Gaussian diffusion model that leverages a self-conditioning technique to enhance generation quality. The model is trained to denoise latent protein representations, progressively transforming a random Gaussian vector into a valid protein representation.

You can configure the training setup in the main configuration file: `src/configs/config.yaml`. Key parameters under the training section include:
- `training_iters`: The total number of training iterations.
- `batch_size`: The total batch size distributed across all GPUs.
- `batch_size_per_gpu`: The batch size for each individual GPU. This is calculated automatically as `batch_size // nproc_per_node`.
- `eval_interval`: The evaluation frequency. At every eval_interval, the validation loss is computed, and a sample generation with metric calculation is performed.
- `save_interval`: The checkpointing frequency.
- `init_se`: The checkpoint used to initialize the score estimator.

To launch the diffusion model training on a multi-GPU node, run the following command:


```bash
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=8 --master_port=31345  train_diffusion.py
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=8 --master_port=31345  train_diffusion_cfg_antibody.py
```

For detailed training hyperparameters, please refer to Appendix C.2 of our paper.


# Citation

```
@article{meshchaninov2024diffusion,
  title={Diffusion on language model embeddings for protein sequence generation},
  author={Meshchaninov, Viacheslav and Strashnov, Pavel and Shevtsov, Andrey and Nikolaev, Fedor and Ivanisenko, Nikita and Kardymon, Olga and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:2403.03726},
  year={2024}
}
```
