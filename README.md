# AI-Powered Reaction Condition Prediction (AI_RCP)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18185850-blue.svg)](https://doi.org/10.5281/zenodo.18185850)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=for-the-badge&logo=xgboost&logoColor=white)
![Wandb](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)

This repository contains the code for a comprehensive computational pipeline designed to accelerate the optimisation of chemical reactions. By integrating generative and predictive modelling with a novel plate design framework for High-Throughput Experimentation (HTE), this project provides a robust platform for exploring vast reaction condition spaces. It is designed to be run on high-performance computing (HPC) clusters and uses Weights & Biases for experiment tracking.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Generative Model](#training-the-generative-model)
  - [Training the XGBoost Yield Prediction Model](#training-the-xgboost-yield-prediction-model)
  - [Running Sweeps for Hyperparameter Optimization](#running-sweeps-for-hyperparameter-optimization)
  - [Retraining Best Models and Computing Baselines](#retraining-best-models-and-computing-baselines)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Third-party packages](#third-party-packages)
- [Notes for macOS users](#notes-for-macos-users)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Project Overview

High-throughput experimentation (HTE) has greatly accelerated chemical research, yet the rational exploration of vast reaction condition spaces remains a significant challenge. This project introduces a computational pipeline to tackle this challenge by combining generative and predictive modeling with a unique plate design framework for HTE.

The pipeline is built around two core components:
1.  **Generative Modeling**: A Variational Autoencoder (VAE)-based generative model, specifically a sequence-to-sequence architecture conditioned on physicochemical embeddings (Seq+Emb), is trained on HTE datasets of Buchwald-Hartwig and Suzuki-Miyaura cross-couplings. This model learns to suggest viable reaction conditions, but like the XGBoost model, it struggles to extrapolate to entirely new transformations.
2.  **Yield Prediction**: An XGBoost model is trained to predict reaction yields, providing a way to score and prune the conditions suggested by the generative model. Our analysis also reveals the surprising effectiveness of a simple, structure-agnostic Frequency-Chain baseline, which often rivals more complex models.

The pipeline culminates in a novel Integer Linear Programming (ILP) framework for automated plate design. This method surpasses naive frequency-based approaches by offering a customizable strategy that optimally balances the inclusion of predicted positive conditions while minimising ineffective ones, thereby enhancing experimental efficiency.

While the individual modelling components exhibit limitations in generalisation, the modular, plug-and-play architecture of the pipeline, combined with the principled ILP design stage, presents a robust and extensible platform for accelerating reaction optimisation.

## Features

*   **VAE-Based Generative Models** 🤖: Includes implementations for VAEs, including a sequence-to-sequence architecture conditioned on physicochemical embeddings (Seq+Emb).
*   **XGBoost for Yield Prediction** 🎯: Utilizes XGBoost for accurate and robust yield prediction, allowing for effective pruning of reaction conditions.
*   **ILP-Based Plate Design** 🧪: A novel Integer Linear Programming (ILP) framework for automated plate design that optimizes for experimental efficiency.
*   **HPC Integration** 💻: Comes with scripts for easy job submission and management on HPC clusters.
*   **Weights & Biases Integration** 📊: All experiments are logged with `wandb` for easy tracking and visualization.
*   **Configurability** ⚙️: Every aspect of the project, from model architecture to training parameters, can be configured via YAML files.
*   **Molecular Feature Calculation** ⚛️: Integrates the `morfeus` library for on-the-fly calculation of various molecular descriptors.

## Repository Structure

Here is a brief overview of the key directories in this repository:

```
├── configs/                            # Configuration files for training, inference, and sweeps
├── data/                               # Raw datasets (CSV) and processed arrays / split indices (.npz, .pkl)
├── data_embeddings/                    # Cached per-reaction physicochemical embeddings (.pkl)
├── data_graphs/                        # Cached molecular graph data used by the generative model (.pkl)
├── condition_embeddings_cache/         # Cached reaction-condition embeddings consumed by the XGBoost stage (.pkl)
├── reagents_dfs/                       # Reagent dataframes exported by SURF2VAEinput.py (--export_reagent_df)
├── models/                             # Model definitions (PyTorch and XGBoost)
├── outputs_xgboost/                    # Output directory for XGBoost models and results
├── scripts/                            # Helper scripts for running sweeps and jobs on HPC
├── trained_models/                     # Output directory for model weights you train (none shipped)
├── utils/                              # Utility scripts and helper functions (incl. bootstrap.py)
├── precompute_condition_embeddings.py  # Script to precompute condition embeddings before XGBoost training
├── train_gen_model.py                  # Script for training the generative model
├── train_xgboost_yield.py              # Script for training the XGBoost yield prediction model
├── retrain_best_models.py              # Retrains the best generative-model configurations multiple times
├── retrain_random_baselines.py         # Computes random / structured / frequency-chain baseline metrics
├── experiment_2.py                     # Single-transformation zero-shot / few-shot case study
├── experiment_3.py                     # Constrained ILP vs naive plate-design comparison
├── inference.py                        # Script for running inference
├── environment.yaml                    # Portable Conda environment file (recommended)
├── environment_hpc_linux_cuda.yaml     # Fully-pinned HPC environment (Linux + CUDA 12.4 only)
├── LICENSE                             # MIT license
├── .gitignore                          # Ignore patterns (caches, wandb runs, build artefacts, ...)
└── README.md                           # This file
```

### Cached / regeneratable artefacts

The following directories are not raw inputs — they are caches produced by the pipeline and are checked in only for convenience so the model-training and inference steps can be exercised without first running the full preprocessing chain. They can be deleted at any time and will be regenerated by the corresponding script:

| Directory / file | Produced by | Consumed by |
|---|---|---|
| `data/*_processed.npz`, `data/split_indices_*.pkl` | `SURF2VAEinput.py` | `train_gen_model.py`, `retrain_best_models.py`, `retrain_random_baselines.py` |
| `data_embeddings/` | `train_gen_model.py` (first run, via `utils/create_graphs.py`) | `train_gen_model.py`, retrain scripts |
| `data_graphs/` | `train_gen_model.py` (first run) | `train_gen_model.py`, retrain scripts |
| `reagents_dfs/` | `SURF2VAEinput.py --export_reagent_df` | `precompute_condition_embeddings.py`, `train_xgboost_yield.py` |
| `condition_embeddings_cache/` | `precompute_condition_embeddings.py` | `train_xgboost_yield.py`, `inference.py` |

### Trained model weights

**No pre-trained model weights are distributed with this repository.** The models behind the paper
were trained on the full internal datasets, which cannot be released, so they are not published
here; `trained_models/` and `outputs_xgboost/models/` are output directories that the training
scripts create and populate.

Every script that needs weights therefore expects you to train them first, which the public data in
`data/` supports end to end:

```bash
# a generative model (needed by inference.py and experiment_3.py)
python retrain_best_models.py --config_file configs/case_study/bh_all_seq_emb.yaml --n_trainings 1

# optionally, the XGBoost yield model used for the pruning step in inference.py
python train_xgboost_yield.py --config_file configs/xgboost_config.yaml
```

Then point `model_path_bh` / `model_path_sm` (and, if you want the pruning step,
`xgb_model_path_bh` / `xgb_model_path_sm`) in `configs/inference_config.yaml` at the files that
produces.

A model trained here uses the public component vocabulary throughout (57 components for `bh_all`,
against 92 internally) and is self-consistent, so the pipeline runs normally. It will not reproduce
the exact numbers reported in the paper, which come from the larger internal datasets — see
[Datasets](#datasets).

## Installation

The repository ships with two Conda environment files:

| File | Use it when... |
|---|---|
| `environment.yaml` | **Recommended.** Portable, cross-platform install — works on macOS (incl. Apple Silicon), Linux x86_64, and Linux with NVIDIA GPUs. |
| `environment_hpc_linux_cuda.yaml` | You need to reproduce the *exact* environment used to run the paper's experiments on the HPC cluster. Linux + NVIDIA GPU + CUDA 12.4 only. |

### Recommended (portable) install

```bash
conda env create -f environment.yaml
conda activate AI_RCP_env
```

This installs the project's direct dependencies (PyTorch, DGL, XGBoost, RDKit, the xtb / Auto3D / tblite chemistry stack, transformers, wandb, ...) and lets Conda resolve the right binaries for your platform. On macOS the resulting PyTorch supports the Apple Silicon MPS backend; on Linux Conda will pull CUDA-enabled wheels automatically when an NVIDIA driver is detected.

> **Native dependency — xtb.** Several feature-extraction steps (e.g. `morfeus.xtb.XTB`) shell out to the `xtb` binary. It is provided by the `xtb` conda package listed in `environment.yaml`, so the recommended install is sufficient. If you choose to skip the conda environment and manage dependencies yourself, install xtb separately (e.g. `conda install -c conda-forge xtb` or via your distro's package manager) and make sure the resulting binary is on `PATH` — `inference.py` automatically prepends the active Python's `bin` directory, but only that location.

If you hit a solver issue for a particular package on your platform, please open an issue — `environment.yaml` is intentionally lightly pinned so that future resolves stay possible, but individual packages may need tweaking as upstream channels evolve.

### Exact HPC reproduction

To recreate the fully-pinned Linux + CUDA 12.4 environment used on the HPC cluster:

```bash
conda env create -f environment_hpc_linux_cuda.yaml
conda activate AI_RCP_env
```

> **Note:** this file is a `conda env export` from a `linux-64` machine with an NVIDIA GPU. It will *not* solve on macOS, on non-NVIDIA Linux, or on Linux distros whose base libraries differ substantially from the original host.

### Optional: explicit CUDA build on Linux

If you want to force the CUDA 12.4 PyTorch + DGL build without using the fully pinned HPC file, edit `environment.yaml` to:

* replace the `dglteam` channel with `dglteam/label/th24_cu124`,
* replace `pytorch` with `pytorch-cuda=12.4` (and add the `nvidia` channel),

then re-run `conda env create -f environment.yaml`.

## Datasets

The `data/` directory contains curated datasets for Buchwald-Hartwig (bh) and Suzuki-Miyaura (sm) cross-coupling reactions. The data is provided in CSV format and follows the Simple User-Friendly Reaction Format (SURF).

The following datasets are available:

*   **`bh_data_clean_all_whitelisted.csv`**: A comprehensive dataset of **10,138** Buchwald-Hartwig reactions, including both positive and negative outcomes.
*   **`bh_data_clean_positive_whitelisted.csv`**: A subset of the Buchwald-Hartwig dataset, containing **3,441** reactions with positive outcomes.
*   **`sm_data_clean_all_whitelisted.csv`**: A comprehensive dataset of **3,426** Suzuki-Miyaura reactions, including both positive and negative outcomes.
*   **`sm_data_clean_positive_whitelisted.csv`**: A subset of the Suzuki-Miyaura dataset, containing **1,878** reactions with positive outcomes.

Each dataset includes detailed information for every reaction, such as:
*   Reaction identifiers (`rxn_id`, `rxn_type`, `rxn_date`)
*   Reaction conditions (`temperature_deg_c`, `time_h`)
*   Starting materials, reagents, catalysts, and solvents, with their names, SMILES strings, and stoichiometric equivalents
*   Product information, including SMILES and yield (`product_1_area%`)

### Relationship to the Zenodo archive

The CSVs in `data/` and the CSVs in the [Zenodo archive](https://doi.org/10.5281/zenodo.18185850)
contain **the same reactions** — identical row counts, aligned row-for-row by `rxn_id` — but not
quite the same columns. The files here are the ones the pipeline in this repository consumes; the
Zenodo files are the SURF-format archival release. If you want to reproduce the results in the
paper, use the copies in `data/`.

| column | in `data/` | on Zenodo | note |
|---|---|---|---|
| `ligand_smiles` | ✗ | ✓ | structure of the phosphine/NHC ligand; not needed by the pipeline, which keys on `catalyst_name` |
| `suff_yield` | ✓ | ✗ | binary positive/negative label. **Exactly** reconstructible as `product_1_area% > 0.05` (verified on all 10,138 / 3,426 rows of the `_all` files) |
| `additives_name_merged`, `additives_smiles_merged`, `additives_fraction_merged` | ✓ | ✗ | the `Additive` condition class |
| `YieldCategory` | ✓ | ✗ | not used anywhere in the pipeline |

Only the additive columns are genuinely irrecoverable from the Zenodo files. They matter for a
minority of wells — a real (non-`NoAdditive`) additive appears in 4.6 % of Buchwald–Hartwig rows
(TBAB, sodium trifluoroacetate, potassium 2-ethylhexanoate) and 14.9 % of Suzuki–Miyaura rows
(MeOH, potassium 2-ethylhexanoate) — but running `SURF2VAEinput.py` on the Zenodo CSVs will silently
drop the `Additive` category, giving a smaller condition vocabulary (57 → 55 components for
`bh_all`, 34 → 32 for `sm_all`) and therefore numbers that will not match the paper exactly.

## Usage

### Data Preparation

The project expects reaction data to be in the Simple User-Friendly Reaction Format (SURF). For more information on the SURF format, please refer to the [SURF GitHub repository](https://github.com/alexarnimueller/surf).

Before training the models, you need to process the input data from a SURF-formatted CSV file. This is done using the `SURF2VAEinput.py` script, which prepares the data for the VAE model and exports a reagent dataframe.

```bash
python SURF2VAEinput.py --infile data/bh_data_clean_all.csv --export_reagent_df
```

This script will process `data/bh_data_clean_all.csv` and generate the necessary input files for the next steps in the pipeline.

### Training the Generative Model

To train the generative model, you can use the `train_gen_model.py` script. You will need to configure the training process by editing `configs/gen_config.yaml`.

```bash
python train_gen_model.py --config_path configs/gen_config.yaml
```

### Training the XGBoost Yield Prediction Model

Once you have a trained generative model, you first need to precompute the embeddings for the reaction conditions. This is done using the `precompute_condition_embeddings.py` script. This process is configured via `configs/precompute_config.yaml`.

```bash
python precompute_condition_embeddings.py --config_path configs/precompute_config.yaml
```

Then, you can train the XGBoost model for yield prediction. The training process is controlled by the `configs/xgboost_config.yaml` file.

```bash
python train_xgboost_yield.py --config_path configs/xgboost_config.yaml
```

### Running Sweeps for Hyperparameter Optimization

The project supports hyperparameter sweeps using Weights & Biases. You can define your sweep configuration in `configs/sweep_config.yaml` and then run the sweep using the `run_sweep.py` script.

The `scripts/` directory also contains several helper scripts for managing sweeps on an HPC cluster.

#### Submitting Parallel Jobs

To submit multiple sweep agents in parallel on an HPC cluster, you can use the `submit_sweep_wrapper.sh` script. This script will create a sweep and submit a specified number of jobs to the cluster.

```bash
scripts/submit_sweep_wrapper.sh --model my_model --reaction bh --dataset all --count 3 --agents 10
```

*   `--model`: The name of the model to use for the sweep.
*   `--reaction`: The reaction type (`bh` for Buchwald-Hartwig or `sm` for Suzuki-Miyaura).
*   `--dataset`: The dataset to use (`all` or `positive`).
*   `--count`: The number of runs for each agent.
*   `--agents`: The number of parallel agents to launch.

#### Adding Agents to an Existing Sweep

If you have an existing sweep and want to add more agents to it, you can use the `add_agents_to_sweep.sh` script:

```bash
scripts/add_agents_to_sweep.sh --sweep-id xyz789ghi --model my_model --agents 2 --count 3
```

*   `--sweep-id`: The ID of the `wandb` sweep to add agents to.
*   `--model`: The name of the model to use for the sweep.
*   `--agents`: The number of additional agents to launch.
*   `--count`: The number of runs for each new agent.

### Retraining Best Models and Computing Baselines

After a sweep has identified a best-performing generative-model configuration, `retrain_best_models.py` retrains it multiple times (different random seeds / data splits) so that variance can be estimated. It takes a base config and a number of training repetitions:

```bash
python retrain_best_models.py --config_file configs/gen_config.yaml --n_trainings 10
```

`retrain_random_baselines.py` evaluates structure-agnostic baselines (uniform random, structured random, and frequency-chain) on the same splits, which is what the paper uses as a sanity check against the learned models:

```bash
python retrain_random_baselines.py --config_file configs/gen_config.yaml --n_runs 10
```

### Single-transformation case study (zero-shot vs. few-shot)

`experiment_2.py` runs the single-target case study of the paper: one model is trained on all
transformations of a reaction class and evaluated on a single held-out target, either with none of
that target's condition instances in training (*zero-shot*) or with a random fraction of them
injected (*few-shot*). It trains from scratch and takes no pre-trained weights.

The test set is selected by SMILES rather than by a split file, so the target transformation is
given on the command line:

```bash
# zero-shot: the target is excluded from training entirely
python experiment_2.py \
    --config_file configs/case_study/sm_all_seq_emb.yaml \
    --test_smiles_list "<reaction SMILES>" \
    --test_injection_percentage 0.0

# few-shot: 20% of the target's condition instances are injected into the training set
python experiment_2.py \
    --config_file configs/case_study/sm_all_seq_emb.yaml \
    --test_smiles_list "<reaction SMILES>" \
    --test_injection_percentage 0.2
```

`scripts/single_job.sh` contains a complete invocation for the Suzuki–Miyaura target used in the
paper. The injected subset is drawn with a single uniform sample without replacement over the
target's condition instances, using the `random_seed` in the config (42 in the shipped files). It is
**not** stratified by outcome, by condition class or by measured area%, so its positive/negative
balance simply follows that of the target transformation.

`data_graphs/` ships only the Buchwald–Hartwig cache, so the first Suzuki–Miyaura run needs its
graphs built: set `load_graphs: false` (and `save_graphs: true` to keep them) in the config, as for
any other script in this repository.

Two things to be aware of when comparing against the paper:

*   The published case-study numbers were produced on the full internal datasets. Both target
    transformations are present in the public subset, but with fewer condition instances (1,700 of
    1,847 for the Buchwald–Hartwig target, 358 of 480 for the Suzuki–Miyaura one) and a different
    rare-component vocabulary, so a public re-run reproduces the protocol and the qualitative
    zero-shot/few-shot contrast, not the exact figures.
*   For the Suzuki–Miyaura branch the twelve ground-truth solvent/base pairs are hard-coded (see the
    comment at that point in the script) and are specific to the paper's target. The
    Buchwald–Hartwig branch derives its top-12 from the held-out positives at run time; use that
    approach for any other target.

### Constraint-aware plate design (ILP vs. naive)

`experiment_3.py` is the plate-design comparison behind Table 3: for each target transformation it
samples positive and negative conditions from a generative model, projects them to the
(ligand, solvent/base) plate-well level, and then builds a 96-well plate twice — once with the naive
frequency-ranking heuristic used in prior work, and once with the ILP constrained to match the
naive positive coverage, so that any gain must come from avoiding predicted-negative wells. Each
plate is reported as four disjoint categories (positive / negative / uncertain / unknown) summing
to 96.

Unlike `experiment_2.py`, this script trains nothing — it consumes an already-trained generative
model, so you supply one you have trained here:

```bash
python experiment_3.py \
    --gen_model_path trained_models/<your_model>.pt \
    --gen_config_path configs/case_study/bh_all_seq_emb.yaml \
    --reaction_smiles_list "<reaction SMILES>" ... \
    --n_conditions 500
```

Targets are given on the command line; with none supplied the script samples ten at random from the
dataset. The paper's ten Buchwald–Hartwig targets are not reproduced here, because four of them
involve products that cannot be shared outside Roche (six are drawn in the SI). Running the script
on public transformations reproduces the method and the four-category accounting, not the specific
numbers in Table 3.

The ILP itself — objective, coverage constraints, and the chemical-incompatibility constraints — is
also used by `inference.py`, which designs a plate for a single transformation at the end of an
inference run.

### Inference

To run inference, you need a trained generative model to sample reaction conditions. None is shipped with this repository (see [Trained model weights](#trained-model-weights)), so train one first. Optionally, you can also use a trained XGBoost model to score these conditions and predict their yield, which helps in pruning low-quality suggestions.

1.  **Configure Inference**: Open `configs/inference_config.yaml`.
    *   Set `model_path_bh` or `model_path_sm` (whichever matches `rtype`) to the path of the generative model you trained.
    *   To enable yield prediction and pruning, set `xgb_model_path` to the path of your trained XGBoost model and `xgb_config_path` to its corresponding configuration file.
    *   Specify the input reaction by setting `starting_material_1`, `starting_material_2`, and `product` SMILES strings.

2.  **Run the script**:
    ```bash
    python inference.py --config_file configs/inference_config.yaml
    ```

The script will output the predicted positive and negative reaction conditions. If XGBoost scoring is enabled, it will first generate a large number of conditions and then use the XGBoost model to filter them based on predicted yield. Finally, a 96-vial plate is designed, ready to be tested with HTE

## Configuration

The behavior of the scripts is controlled by YAML configuration files in the `configs/` directory. Here's a brief overview:

*   `gen_config.yaml`: Configuration for the generative model training.
*   `precompute_config.yaml`: Configuration for precomputing condition embeddings.
*   `xgboost_config.yaml`: Configuration for the XGBoost model training.
*   `inference_config.yaml`: Configuration for the inference script.
*   `sweep_config.yaml`: Configuration for hyperparameter sweeps with Weights & Biases.
*   `case_study/{bh,sm}_all_seq_emb.yaml`: The per-model hyperparameters selected for `CondVAE` on
    the two `all` datasets, used by `experiment_2.py`. `rtype` and `data_type` are derived from
    `filepath`, so pointing `filepath` at a different processed dataset is enough to switch target.

Please refer to the configuration files for detailed explanations of each parameter.

## Third-party packages

*   **`morfeus`** — molecular-feature library from the [Digital Chemistry Laboratory](https://github.com/digital-chemistry-laboratory/morfeus). The repository previously vendored it as a git submodule; it is now installed from `conda-forge` as the `morfeus-ml` package (already listed in `environment.yaml`), so no extra setup step is required.

## Notes for macOS users

* The `environment.yaml` solver is constrained to `pytorch 2.3.x` because the only `osx-arm64` build of `dgl` on `conda-forge` (currently 2.3) hard-pins that version. Newer DGL builds for Apple Silicon are not yet available; revisit this pin when `dgl >= 2.4` lands on `conda-forge`.
* Because `transformers >= 4.46` refuses to call `torch.load` on `torch < 2.6` (CVE-2025-32434) and the upstream `seyonec/ChemBERTa-zinc-base-v1` checkpoint only ships a legacy `pytorch_model.bin`, the project includes a small bootstrap helper (`utils/bootstrap.py::ensure_chemberta_safetensors`) that converts the cached weights to `model.safetensors` on first use. Inference, training, and embedding-precomputation scripts call it automatically; you do not need to run anything by hand.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## Citation

### Dataset (Zenodo archive)

The curated HTE datasets used in this work are archived on Zenodo as
*"High-Throughput Experimentation Datasets for Palladium-Catalyzed Suzuki-Miyaura and Buchwald-Hartwig Cross-Coupling Reactions in SURF Format"*.

> Tarricone, L., Schmid, S. P., Jost, V., Lutz, M., Schneider, G., Wuitschik, G., & Jorner, K. *High-Throughput Experimentation Datasets for Palladium-Catalyzed Suzuki-Miyaura and Buchwald-Hartwig Cross-Coupling Reactions in SURF Format*. Zenodo. https://doi.org/10.5281/zenodo.18185850

```bibtex
@dataset{tarricone_hte_surf_datasets,
  author    = {Tarricone, Lorenzo and Schmid, Stefan P. and Jost, Vera and Lutz, Marius and Schneider, Gisbert and Wuitschik, Georg and Jorner, Kjell},
  title     = {High-Throughput Experimentation Datasets for Palladium-Catalyzed Suzuki-Miyaura and Buchwald-Hartwig Cross-Coupling Reactions in SURF Format},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18185850},
  url       = {https://doi.org/10.5281/zenodo.18185850}
}
```

The DOI `10.5281/zenodo.18185850` is the concept (record) DOI and always resolves to the latest version of the archive.

### Publication

The methodology implemented in this repository is described in:

> Tarricone, L., Schmid, S. P., Jost, V., Lutz, M., Schneider, G., Wuitschik, G., & Jorner, K. *End-to-End Conditions Generation for High-Throughput Experimentation under Practical Constraints*. <!-- TODO: add journal / preprint venue once available -->

<!-- TODO: replace with the final publication URL / DOI once the paper is published -->
<!-- TODO: add the BibTeX entry below with the correct journal, year, volume, pages, and DOI -->

```bibtex
@article{tarricone_e2e_hte,
  author  = {Tarricone, Lorenzo and Schmid, Stefan P. and Jost, Vera and Lutz, Marius and Schneider, Gisbert and Wuitschik, Georg and Jorner, Kjell},
  title   = {End-to-End Conditions Generation for High-Throughput Experimentation under Practical Constraints},
  journal = {TODO},
  year    = {TODO},
  doi     = {TODO},
  url     = {TODO}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
