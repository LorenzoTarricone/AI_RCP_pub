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
├── trained_models/                     # Saved model weights (large .pt files; see note below)
├── utils/                              # Utility scripts and helper functions (incl. bootstrap.py)
├── precompute_condition_embeddings.py  # Script to precompute condition embeddings before XGBoost training
├── train_gen_model.py                  # Script for training the generative model
├── train_xgboost_yield.py              # Script for training the XGBoost yield prediction model
├── retrain_best_models.py              # Retrains the best generative-model configurations multiple times
├── retrain_random_baselines.py         # Computes random / structured / frequency-chain baseline metrics
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

`trained_models/` ships with several pre-trained `.pt` checkpoints (the largest are ~80 MB each, totalling several hundred MB). These are tracked directly in the repository for now so that `inference.py` works out of the box; a future revision may move them to the Zenodo archive (see [Citation](#citation)) or to Git LFS. If you only intend to retrain from scratch, you can safely ignore or delete this directory.

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

### Inference

To run inference, you need a trained generative model to sample reaction conditions. Optionally, you can also use a trained XGBoost model to score these conditions and predict their yield, which helps in pruning low-quality suggestions.

1.  **Configure Inference**: Open `configs/inference_config.yaml`.
    *   Set `model_path` to the path of your trained generative model (e.g., `trained_models/model_seq_emb_bh_all_0.pt`).
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
