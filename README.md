# AI-Powered Reaction Condition Prediction (AI_RCP)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
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
  - [Inference](#inference)
- [Configuration](#configuration)
- [Submodules](#submodules)
- [Contributing](#contributing)
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
├── data/                               # Raw and processed data
├── HPC_scripts/                        # Scripts for running jobs on an HPC cluster
├── models/                             # Model definitions (PyTorch and XGBoost)
├── morfeus/                            # Submodule for molecular feature calculation
├── outputs_xgboost/                    # Output directory for XGBoost models and results
├── scripts/                            # Helper scripts for running sweeps and jobs on HPC
├── trained_models/                     # Saved model weights
├── utils/                              # Utility scripts and helper functions
├── precompute_condition_embeddings.py  # Script to precompute condition embeddings before XGBoost training
├── train_gen_model.py                  # Script for training the generative model
├── train_xgboost_yield.py              # Script for training the XGBoost yield prediction model
├── inference.py                        # Script for running inference
├── environment.yaml                    # Conda environment file
└── README.md                           # This file
```

## Installation

To set up the environment for this project, you will need to have Conda installed. Then, you can create the environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate AI_RCP_env
```

This will install all the necessary dependencies, including PyTorch, DGL, XGBoost, and RDKit.

## Usage

### Data Preparation

Place your raw and processed data in the `data/` directory. The project expects the data to be in a specific format, so you may need to adapt your data or the data loading scripts in `utils/`.

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

## Submodules

This repository includes the following submodule:

*   **`morfeus`**: A Python package for calculating molecular features, developed by the [Digital Chemistry Laboratory](https://github.com/digital-chemistry-laboratory/morfeus). For more information, please refer to the `morfeus/README.md` file or its official documentation.

To clone the repository with the submodule, use:
```bash
git clone --recurse-submodules <repository-url>
```

If you have already cloned the repository, you can initialize the submodule with:
```bash
git submodule update --init --recursive
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
