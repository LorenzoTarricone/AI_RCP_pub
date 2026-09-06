import argparse
import numpy as np
import yaml
import os
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import json
import shutil
from collections import defaultdict

def flatten_wandb_config(config):
    """Recursively flattens a wandb config dictionary."""
    flat_config = {}
    for key, value in config.items():
        if key.startswith('_'):  # Skip internal wandb keys
            continue
        if isinstance(value, dict) and 'value' in value:
            flat_config[key] = value['value']
        else:
            flat_config[key] = value
    return flat_config

from utils.create_graphs import get_graph_data, load_graph_data
from utils.collate_functions import collate_reaction_graphs, collate_graphs_and_embeddings
from utils.dataset import GraphDataset, get_cardinalities_classes
from utils.evaluate_model import evaluate_model, evaluate_model_and_get_preds
from utils.trn_val_tst_sampling import (
    iterative_stratified_split,
)
from utils.miscellaneous import create_pos_neg_count_matrices
# Import the random baseline function
from utils.miscellaneous import compute_random_baseline, compute_structured_random_baseline, compute_frequency_chain_baseline
from utils.miscellaneous import compute_most_frequent_baseline

# Attempt to import wandb and handle if not installed
try:
    import wandb
except ImportError:
    wandb = None

def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Also configure RDKit logging
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    if not verbose:
        RDLogger.DisableLog('rdApp.warning')
        RDLogger.DisableLog('rdApp.error')
    
    # Return logger instance for use in other functions
    return logging.getLogger(__name__)


def retrain_algorithm(config, n_trainings, wandb_is_active):
    """
    Contains the main data processing logic of the script.
    Logs to W&B if wandb_is_active is True and wandb.run is available.
    """
    # Get logger instance
    logger = logging.getLogger(__name__)

    ################################ LOAD DATA ################################
    ########################### and check the format ##########################

    file_path = config.get('filepath')
    
    try:
        frac_tst = float(config.get('frac_tst'))
    except (ValueError, TypeError) as e:
        err_msg = f"ERROR: 'frac_tst' must be a number. Value: '{config.get('frac_tst')}', Error: {e}"
        logger.error(err_msg)
        if wandb_is_active and wandb.run:
             wandb.log({"error_message": "Invalid frac_tst value"})
             wandb.run.finish(exit_code=1)
        return False # Indicate failure

    random_seed_val = config.get('random_seed')
    random_seed = int(random_seed_val) if random_seed_val is not None else None

    if not file_path:
        err_msg = "ERROR: 'filepath' is a required parameter and was not found in the configuration."
        logger.error(err_msg)
        if wandb_is_active and wandb.run:
            wandb.log({"error_message": "'filepath' missing"})
            wandb.run.finish(exit_code=1)
        return False

    logger.info(f"\n--- Effective Configuration for this Run ---")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    if wandb_is_active and wandb.run: # wandb.run might not exist if init failed earlier or mode is disabled
        logger.info(f"W&B Run Name: {wandb.run.name} (ID: {wandb.run.id})")
    logger.info(f"-------------------------------------------\n")

    if "bh" in file_path:
        config["rtype"] = 'bh'
    elif "sm" in file_path:
        config["rtype"] = 'sm'
    else:
        err_msg = f"ERROR: Type of reaction not recognized from filepath: {file_path}!"
        logger.error(err_msg)
        if wandb_is_active and wandb.run:
            wandb.log({"error_message": err_msg})
            wandb.run.finish(exit_code=1)
        return False

    # Extract the "all/positive" information from the filepath
    if "all_all" in file_path:
        config["data_type"] = "all"
    elif "all_positive" in file_path:
        config["data_type"] = "positive"
    else:
        err_msg = f"ERROR: Data type (all/positive) not recognized from filepath: {file_path}!"
        logger.error(err_msg)
        if wandb_is_active and wandb.run:
            wandb.log({"error_message": err_msg})
            wandb.run.finish(exit_code=1)
        return False

    try:
        loaded_npz = np.load(file_path, allow_pickle=True)
        if 'data' not in loaded_npz:
            err_msg = f"ERROR: 'data' key not found in the .npz file: {file_path}"
            logger.error(err_msg)
            if wandb_is_active and wandb.run:
                wandb.log({"error_message": err_msg})
                wandb.run.finish(exit_code=1)
            return False

        reaction_data_unpacked = loaded_npz['data']
        # print("Reactiond data unpacked: ", reaction_data_unpacked)
        if isinstance(reaction_data_unpacked, np.ndarray) and len(reaction_data_unpacked) == 2:
            reaction_dict, clist = reaction_data_unpacked
            config["clist"] = clist
            if not isinstance(reaction_dict, dict): 
                logger.error("first element should be the reaciton dict (type: dict)")
                if wandb_is_active and wandb.run:
                    wandb.log({"error_message": "first element should be the reaciton dict (type: dict)"})
                    wandb.run.finish(exit_code=1)
                return False
            if not isinstance(clist, list): 
                logger.error("second element should be the clist dict (type: list)")
                if wandb_is_active and wandb.run:
                    wandb.log({"second element should be the clist dict (type: list)"})
                    wandb.run.finish(exit_code=1)
                return False
        else:
            err_msg = (f"ERROR: Expected 'data' in {file_path} to be a 2-element np.array "
                       f"to unpack into reaction_dict and clist. Got type: {type(reaction_data_unpacked)}")
            logger.error(err_msg)
            if wandb_is_active and wandb.run:
                wandb.log({"error_message": err_msg})
                wandb.run.finish(exit_code=1)
            return False
    except FileNotFoundError:
        err_msg = f"ERROR: Data file not found at {file_path}"
        logger.error(err_msg)
        if wandb_is_active and wandb.run:
            wandb.log({"error_message": err_msg})
            wandb.run.finish(exit_code=1)
        return False
    except Exception as e:
        err_msg = f"ERROR: Loading or processing data from {file_path}: {e}"
        logger.error(err_msg)
        if wandb_is_active and wandb.run:
            wandb.log({"error_message": str(e)})
            wandb.run.finish(exit_code=1)
        return False


    if random_seed is not None:
        np.random.seed(random_seed)
        logger.info(f"INFO: Using random seed: {random_seed}")
    else:
        logger.info("INFO: No random seed set.")

    
    device = torch.device("cuda" if (torch.cuda.is_available() and config["device"] == "cuda") else "cpu")
    logger.info(f"INFO: Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}") #prints the GPU name
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB") #prints the amount of GPU memory allocated.
        cuda = device

    os.makedirs('./trained_models/', exist_ok=True)
    
    # Create a unique identifier for this run to prevent model path conflicts
    if wandb_is_active and wandb.run:
        # Use the unique W&B run ID for sweep compatibility
        run_identifier = wandb.run.id
    else:
        # Fallback for non-W&B runs, using a timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_identifier = f'{config["iterid"]}_{timestamp}'
    
    config["run_identifier"] = run_identifier
 
    ################################ GRAPH STRUCTURES ################################
    ################################# generation ####################################

    if config["load_graphs"]:
        try:
            logger.info("--- Loading graphs structures from dataset ---")
            if config["model_type"] in ["emb", "seq_emb"]:
                all_rmol_graphs, all_pmol_graphs, all_reaction_labels, all_reaction_smiles, all_embeddings_mol = load_graph_data(config)
                logger.info(f"INFO: Successfully loaded data")
            else:
                all_rmol_graphs, all_pmol_graphs, all_reaction_labels, all_reaction_smiles, _ = load_graph_data(config)
                logger.info(f"INFO: Successfully loaded data")
        except Exception as e:
            err_msg = f"ERROR: During loading of graph data: {e}"
            logger.error(err_msg)
            if wandb_is_active and wandb.run:
                wandb.log({"error_message": f"get_graph_data error: {str(e)}"})
                wandb.run.finish(exit_code=1)
            return False
    else:
        try:
            logger.info("--- Generating graphs structures from dataset ---")
            reaction_keys = np.array(list(reaction_dict.keys()))
            if config["model_type"] in ["emb", "seq_emb"]:
                all_rmol_graphs, all_pmol_graphs, all_reaction_labels, all_reaction_smiles, all_embeddings_mol = get_graph_data(reaction_dict, reaction_keys, config)
                logger.info("INFO: Graph data processing complete.")
            else:
                all_rmol_graphs, all_pmol_graphs, all_reaction_labels, all_reaction_smiles, _ = get_graph_data(reaction_dict, reaction_keys, config)
                logger.info("INFO: Graph data processing complete.")
        except Exception as e:
            err_msg = f"ERROR: During get_graph_data: {e}"
            logger.error(err_msg)
            if wandb_is_active and wandb.run:
                wandb.log({"error_message": f"get_graph_data error: {str(e)}"})
                wandb.run.finish(exit_code=1)
            return False

    ################################# DATA SPLITS  ##################################
    ################################# generation ####################################

    config["n_classes"] = len(clist)

    # This will check if all lists have the same length before proceeding.
    assert len(all_rmol_graphs) == len(all_pmol_graphs) == len(all_reaction_labels) == len(all_reaction_smiles), \
        "Input lists must all have the same length."

    logger.info("--- Splitting data into a fixed train/test set ---")
    num_reactions = len(all_rmol_graphs)
    
    # --- Custom Data Split based on SMILES ---
    logger.info("--- Splitting data based on provided SMILES list ---")
    test_smiles_list = config['test_smiles_list']
    
    all_smiles_np = np.array(all_reaction_smiles)
    
    # Find indices for the test set
    test_indices_mask = np.isin(all_smiles_np, test_smiles_list)
    test_indices = np.where(test_indices_mask)[0]

    #raise error if test_indices is empty
    if len(test_indices) == 0:
        logger.error("No reactions found for the provided test SMILES list. Aborting.")
        return False
    
    if config['test_injection_percentage'] == 0:
        train_pool_indices = np.where(~test_indices_mask)[0]
    else:
        #will train on everything
        train_pool_indices = np.arange(len(all_rmol_graphs))


    if len(test_indices) == 0:
        logger.error("No reactions found for the provided test SMILES list. Aborting.")
        return False
        
    logger.info(f"Total reactions: {num_reactions}")
    logger.info(f"Training pool size: {len(train_pool_indices)}")
    logger.info(f"Test set size: {len(test_indices)}")


    ############################### MODEL PARAMETERS ################################
    ################################# import ####################################

    if config["model_type"] == 'baseline':
        from models.model_VAE import VAE as Model
        from models.model_VAE import Trainer
        config["use_rxnfp"] = False
        config["collate_fn"] = collate_reaction_graphs
        config["expand_data"] = False
        config["emb_to_use"] = None
    elif config["model_type"] == 'rxnfp':
        from models.model_rxnfp import FNN as Model
        from models.model_rxnfp import Trainer
        config["use_rxnfp"] = True
        config["collate_fn"] = None
        config["expand_data"] = False
        config["emb_to_use"] = None
    elif config["model_type"] == 'seq':
        from models.model_VAE_seq import VAE_seq as Model
        from models.model_VAE_seq import Trainer
        config["use_rxnfp"] = False
        config["collate_fn"] = collate_reaction_graphs
        config["expand_data"] = True
        config["emb_to_use"] = None
    elif config["model_type"] == 'emb':
        from models.model_VAE_emb import VAE_emb as Model
        from models.model_VAE_emb import Trainer
        config["use_rxnfp"] = False
        config["collate_fn"] = collate_graphs_and_embeddings
        config["expand_data"] = False
    elif config["model_type"] == 'seq_emb':
        from models.model_VAE_seq_emb import VAE_seq_emb as Model
        from models.model_VAE_seq_emb import Trainer
        config["use_rxnfp"] = False
        config["collate_fn"] = collate_graphs_and_embeddings
        config["expand_data"] = True
    else:
        err_msg = f"ERROR: Unrecognized model_type '{config['model_type']}'. Valid options are: 'baseline', 'rxnfp', 'seq', 'emb', 'seq_emb'"
        logger.error(err_msg)
        if wandb_is_active and wandb.run:
            wandb.log({"error_message": err_msg})
            wandb.run.finish(exit_code=1)
        return False

    temp_mol_emb = None
    if config["model_type"] in ["emb", "seq_emb"]:
        temp_mol_emb = [all_embeddings_mol[i] for i in train_pool_indices]
    
    temp_trndata = GraphDataset([all_rmol_graphs[i] for i in train_pool_indices], [all_pmol_graphs[i] for i in train_pool_indices], 
                              [all_reaction_labels[i] for i in train_pool_indices], [all_reaction_smiles[i] for i in train_pool_indices], 
                              temp_mol_emb, config, split='trn', device=device)

    # Set n_classes from the dataset before using it
    config["n_classes"] = temp_trndata.n_classes
    config["rmol_max_cnt"] = temp_trndata.rmol_max_cnt
    config["pmol_max_cnt"] = temp_trndata.pmol_max_cnt
    
    n_info = get_cardinalities_classes(config)
    config["n_info"] = n_info

    ################################# RETRAINING LOOP ##################################

    best_performance = -1.0
    best_run_idx = -1
    all_run_metrics = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs('./best_models/', exist_ok=True)
    os.makedirs('./best_results/', exist_ok=True)
        
    best_model_path = f'./best_models/best_model_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{timestamp}.pt'
    results_path = f'./best_results/results_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{timestamp}.json'
    
    # Test set remains fixed
    test_rmol_graphs = [all_rmol_graphs[i] for i in test_indices]
    test_pmol_graphs = [all_pmol_graphs[i] for i in test_indices]
    test_labels = [all_reaction_labels[i] for i in test_indices]
    test_smiles = [all_reaction_smiles[i] for i in test_indices]

    if config['test_injection_percentage'] > 0:
        #select for each set of test labels a percentage of the labels to inject into the training set
        indices_to_inject_for_each_condition = {}
        for i, labels in enumerate(test_labels):
            num_to_inject = int(len(labels) * config['test_injection_percentage'])
            indices_to_inject = np.random.choice(np.arange(len(labels)), size=num_to_inject, replace=False)
            indices_to_keep_in_test_set = [j for j in range(len(labels)) if j not in indices_to_inject]
            test_labels[i] = [labels[j] for j in indices_to_keep_in_test_set]
            indices_to_inject_for_each_condition[test_indices[i]] = indices_to_inject

  

    test_mol_emb = None
    if config["model_type"] in ["emb", "seq_emb"]:
        test_mol_emb = [all_embeddings_mol[i] for i in test_indices]
    
    tstdata = GraphDataset(test_rmol_graphs, test_pmol_graphs, test_labels, test_smiles, test_mol_emb, config, split='tst', device=device)
    tst_loader = DataLoader(dataset=tstdata, batch_size=config["batch_size"], shuffle=False, collate_fn=config["collate_fn"])
    
    tst_y = tstdata.y
    if config["data_type"] == "positive":
        tst_y_pos = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in tst_y]
        tst_y_neg = None
    else:  # "all" data
        tst_y_pos = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in tst_y]
        tst_y_neg = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 0] for sublist in tst_y]


    for run_idx in range(n_trainings):
        logger.info(f"\n--- Starting Training Run {run_idx + 1}/{n_trainings} ---")

        # --- Set random seed for this run for model initialization and data shuffling ---
        if random_seed is not None:
            run_seed = random_seed + run_idx
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            logger.info(f"Run {run_idx+1}: Set random seed to {run_seed}")
        else:
            # To ensure different initializations if no seed is given
            torch.seed()
            np.random.seed(None) # Seed from OS
            logger.info(f"Run {run_idx+1}: No base random seed provided. Using fresh random initialization for this run.")

        # --- Use the full training pool for both training and validation ---
        # As per the request, we train on all data not in the test set.
        # The validation set is made identical to the training set.
        train_indices_run = train_pool_indices
        val_indices_run = train_pool_indices

        logger.info(f"Run {run_idx+1}: Using full training pool for both training and validation.")
        logger.info(f"Run {run_idx+1}: Train size: {len(train_indices_run)}, Validation size: {len(val_indices_run)}")

        # --- Prepare DataLoaders for the run ---
        train_rmol_graphs = [all_rmol_graphs[i] for i in train_indices_run]
        train_pmol_graphs = [all_pmol_graphs[i] for i in train_indices_run]
        train_labels = [all_reaction_labels[i] for i in train_indices_run]
        train_smiles = [all_reaction_smiles[i] for i in train_indices_run]
        
        val_rmol_graphs = [all_rmol_graphs[i] for i in val_indices_run]
        val_pmol_graphs = [all_pmol_graphs[i] for i in val_indices_run]
        val_labels = [all_reaction_labels[i] for i in val_indices_run]
        val_smiles = [all_reaction_smiles[i] for i in val_indices_run]

        if config['test_injection_percentage'] > 0:
            train_index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(train_indices_run)}
            val_index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(val_indices_run)}
            for index, indices_to_inject in indices_to_inject_for_each_condition.items():
                if index in train_index_map:
                    local_train_idx = train_index_map[index]
                    original_train_labels = np.array(train_labels[local_train_idx], dtype=object)
                    train_labels[local_train_idx] = original_train_labels[indices_to_inject].tolist()

                if index in val_index_map:
                    local_val_idx = val_index_map[index]
                    original_val_labels = np.array(val_labels[local_val_idx], dtype=object)
                    val_labels[local_val_idx] = original_val_labels[indices_to_inject].tolist()

        # --- Remove conditions from training set that are in the test set ---

        train_mol_emb, val_mol_emb = None, None
        if config["model_type"] in ["emb", "seq_emb"]:
            train_mol_emb = [all_embeddings_mol[i] for i in train_indices_run]
            val_mol_emb = [all_embeddings_mol[i] for i in val_indices_run]
        
        trndata = GraphDataset(train_rmol_graphs, train_pmol_graphs, train_labels, train_smiles, train_mol_emb, config, split='trn', device=device)
        valdata = GraphDataset(val_rmol_graphs, val_pmol_graphs, val_labels, val_smiles, val_mol_emb, config, split='val', device=device)
        
        trn_loader = DataLoader(dataset=trndata, batch_size=config["batch_size"], shuffle=True, collate_fn=config["collate_fn"])
        val_loader = DataLoader(dataset=valdata, batch_size=config["batch_size"], shuffle=False, collate_fn=config["collate_fn"])

        # --- Model and Trainer setup ---
        model_path_run = f'./trained_models/retrain_run_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{run_identifier}_run_{run_idx}.pt'
        
        # Set model_path in config before initializing trainer
        if config.get('load_model_path') and os.path.exists(config['load_model_path']):
            config['model_path'] = config['load_model_path']
        else:
            config['model_path'] = model_path_run
        
        # Instantiate Model
        if config["model_type"] == 'rxnfp': 
            net = Model(tstdata.fp_dim * 3 + 1, config["n_classes"])
        elif (config["model_type"] == 'baseline'): 
            net = Model(tstdata.node_dim, tstdata.edge_dim, config["n_classes"])
        elif config["model_type"] == 'seq':
            net = Model(config["rtype"], tstdata.node_dim, tstdata.edge_dim, config["n_classes"], config["n_info"])
        elif config["model_type"] == 'emb':
            net = Model(tstdata.node_dim, tstdata.edge_dim, config["n_classes"], tstdata.emb_dim)
        elif config["model_type"] == 'seq_emb':
            net = Model(config["rtype"], tstdata.node_dim, tstdata.edge_dim, config["n_classes"], config["n_info"], tstdata.emb_dim)

        trainer = Trainer(net, device, config)

        # --- Load or Train ---
        if config.get('load_model_path') and os.path.exists(config['load_model_path']):
            logger.info(f"Loading pre-trained model from {config['load_model_path']}")
            trainer.load()
        else:
            logger.info("Starting new training run...")
            if config["class_weights"]:
                pos_count_matrix_1, pos_count_matrix_0, neg_count_matrix_1, neg_count_matrix_0 = create_pos_neg_count_matrices(train_labels, config["n_classes"])
                config["train_pos_count_matrix_1"] = pos_count_matrix_1
                config["train_pos_count_matrix_0"] = pos_count_matrix_0
                config["train_neg_count_matrix_1"] = neg_count_matrix_1
                config["train_neg_count_matrix_0"] = neg_count_matrix_0
            else:
                config["train_pos_count_matrix_1"] = None
                config["train_pos_count_matrix_0"] = None
                config["train_neg_count_matrix_1"] = None
                config["train_neg_count_matrix_0"] = None
                config["train_pos_count_matrix"] = None
                config["train_neg_count_matrix"] = None

            trainer.training(trn_loader, val_loader, config["epochs"])
            # After training, load the best model that was saved during the run
            trainer.load()

        # --- Evaluate on Test Set --- 
        evaluation_results = evaluate_model_and_get_preds(trainer, tst_loader, tst_y_pos, tst_y_neg, config, logger)

        # Determine the performance of the current run based on evaluation results
        current_performance = -1.0 # Default value
        is_simple_model = config["model_type"] in ['rxnfp', 'baselineofbaseline']
        
        try:
            if is_simple_model:
                # Simple models have results at T=1
                pos_metrics = evaluation_results.get('positive', {})
                acc_pos = pos_metrics.get('accuracy', 0.0)
                macro_pos = pos_metrics.get('macro_recall', 0.0)
                micro_pos = pos_metrics.get('micro_recall', 0.0)

                if config["data_type"] == "positive":
                    current_performance = (acc_pos + macro_pos + micro_pos) / 3
                else: # 'all'
                    neg_metrics = evaluation_results.get('negative', {})
                    acc_neg = neg_metrics.get('accuracy', 0.0)
                    macro_neg = neg_metrics.get('macro_recall', 0.0)
                    micro_neg = neg_metrics.get('micro_recall', 0.0)
                    current_performance = (acc_pos + macro_pos + micro_pos + acc_neg + macro_neg + micro_neg) / 6
            else: # VAE-like models
                # Use performance at max T
                max_T = max(evaluation_results.get('T_values', [0]))
                if max_T > 0:
                    pos_metrics = evaluation_results.get('positive', {}).get(max_T, {})
                    acc_pos = pos_metrics.get('accuracy', 0.0)
                    macro_pos = pos_metrics.get('macro_recall', 0.0)
                    micro_pos = pos_metrics.get('micro_recall', 0.0)

                    if config["data_type"] == "positive":
                        current_performance = (acc_pos + macro_pos + micro_pos) / 3
                    else: # 'all'
                        neg_metrics = evaluation_results.get('negative', {}).get(max_T, {})
                        acc_neg = neg_metrics.get('accuracy', 0.0)
                        macro_neg = neg_metrics.get('macro_recall', 0.0)
                        micro_neg = neg_metrics.get('micro_recall', 0.0)
                        current_performance = (acc_pos + macro_pos + micro_pos + acc_neg + macro_neg + micro_neg) / 6
        except Exception as e:
            logger.error(f"Error calculating performance for run {run_idx+1}: {e}")
            current_performance = -1.0
        
        logger.info(f"Run {run_idx+1} Test Performance: {current_performance:.4f}")
        
        evaluation_results['overall_performance'] = current_performance
        all_run_metrics.append(evaluation_results)
        
        if current_performance > best_performance:
            best_performance = current_performance
            best_run_idx = run_idx
            logger.info(f"New best run found: Run {best_run_idx + 1} with performance {best_performance:.4f}")
            if wandb_is_active and wandb.run:
                wandb.summary['best_test_performance'] = best_performance


    # --- Save the best model and clean up other models ---
    if not config.get('load_model_path') and best_run_idx != -1:
        os.makedirs('experiment_2_results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        injection_percentage = config['test_injection_percentage']
        save_path = f'experiment_2_results/model_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{injection_percentage}_{timestamp}.pt'
        
        best_model_run_path = f'./trained_models/retrain_run_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{config["run_identifier"]}_run_{best_run_idx}.pt'
        if os.path.exists(best_model_run_path):
            shutil.copyfile(best_model_run_path, save_path)
            logger.info(f"Copied best model to {save_path}")
        else:
            logger.warning(f"Best model file not found at {best_model_run_path}, could not copy.")

    for run_idx in range(n_trainings):
        model_path_to_delete = f'./trained_models/retrain_run_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{config["run_identifier"]}_run_{run_idx}.pt'
        try:
            if os.path.exists(model_path_to_delete):
                os.remove(model_path_to_delete)
                logger.info(f"Cleaned up model from run {run_idx + 1}: {model_path_to_delete}")
        except OSError as e:
            logger.warning(f"Error deleting model {model_path_to_delete}: {e}")

    reagent_df = pd.read_csv(f"reagents_dfs/bh_treshold_all_all_reagent_df.csv") if config["rtype"] == "bh" else pd.read_csv(f"reagents_dfs/sm_treshold_all_all_reagent_df.csv")
    
    if config["rtype"] == "bh":
        # --- Catalyst Analysis ---
        logger.info("\n--- Catalyst Analysis ---")
        
        catalyst_df = reagent_df[reagent_df['reagent_type'] == 'C']
        catalyst_indices = catalyst_df.index.tolist()
        
        # Get ground truth catalyst distribution
        true_catalyst_indices = []
        for labels in tst_y_pos:
            for condition_set in labels:
                # Find the index in the condition set that is in the catalyst_indices list
                index = next((i for i, x in enumerate(condition_set) if x in catalyst_indices), None)
                if index is not None:
                    true_catalyst_indices.append(condition_set[index])
                else:
                    logger.warning(f"Condition set {condition_set} does not contain any catalyst indices.")

        true_catalyst_counts = pd.Series(true_catalyst_indices).value_counts().nlargest(12)
        true_catalyst_names = catalyst_df.loc[true_catalyst_counts.index]['reagent']
        logger.info("Top 12 Ground Truth Catalysts:")
        for name, count in zip(true_catalyst_names, true_catalyst_counts):
            logger.info(f"{name}: {count}")

        # Get predicted catalyst distribution
        predicted_catalyst_indices = []
        for preds in evaluation_results['positive_predictions']:
            for pred_set in preds:
                index = next((i for i, x in enumerate(pred_set) if x in catalyst_indices), None)
                if index is not None:
                    predicted_catalyst_indices.append(pred_set[index])
                else:
                    logger.warning(f"Condition set {pred_set} does not contain any catalyst indices.")

        predicted_catalyst_counts = pd.Series(predicted_catalyst_indices).value_counts().nlargest(12)
        predicted_catalyst_names = catalyst_df.loc[predicted_catalyst_counts.index]['reagent']
        logger.info("\nTop 12 Predicted Catalysts:")
        for name, count in zip(predicted_catalyst_names, predicted_catalyst_counts):
            logger.info(f"{name}: {count}")

        #Count how many times top-12 predicted catalysts are in the top-12 ground truth catalysts
        top_12_predicted_catalysts = predicted_catalyst_names.tolist()
        top_12_true_catalysts = true_catalyst_names.tolist()
        top_12_predicted_catalysts_in_top_12_true_catalysts = [x for x in top_12_predicted_catalysts if x in top_12_true_catalysts]
        logger.info(f"Number of Top-12 predicted catalysts in top-12 ground truth catalysts: {len(top_12_predicted_catalysts_in_top_12_true_catalysts) / 12:.2f}")
        logger.info(f"Names of Top-12 predicted catalysts in top-12 ground truth catalysts: {top_12_predicted_catalysts_in_top_12_true_catalysts}")

    if config["rtype"] == "sm":
        # --- Solvent and Base Analysis ---
        logger.info("\n--- Solvent and Base Analysis ---")

        # Fixed reference set for the Suzuki-Miyaura target transformation of the paper's case
        # study: the twelve solvent/base pairs used as ground truth in Fig. 6 and Table 2. Unlike
        # the Buchwald-Hartwig branch above, which derives its top 12 from the held-out positives at
        # run time, this list is hard-coded, so it is only meaningful for that one transformation.
        # For any other target, derive the reference set the same way the bh branch does.
        ground_truth_base_solvent_tuples = [("K3PO4", "Dioxane"), ("Cs2CO3", "THF"), ("K2CO3", "Dioxane"), ("K3PO4", "THF"), ("Na2CO3", "EtOH"), ("Cs2CO3", "tAmOH"), ("K3PO4", "PhMe"), ("Na2CO3", "tAmOH"), ("K3PO4", "tAmOH"), ("K3PO4", "MeCN"), ("K3PO4", "DMF"), ("K3PO4", "iPrOH") ]
        
        solvent_df = reagent_df[reagent_df['reagent_type'] == 'S']
        solvent_indices = solvent_df.index.tolist()

        base_df = reagent_df[reagent_df['reagent_type'] == 'B']
        base_indices = base_df.index.tolist()

        # Log ground truth from the provided list
        logger.info("Top 12 Ground Truth Solvent-Base Tuples:")
        for base, solvent in ground_truth_base_solvent_tuples:
            logger.info(f"({base}, {solvent})")

        # Get predicted solvent-base pairs
        predicted_base_solvent_pairs = []
        if 'positive_predictions' in evaluation_results and evaluation_results['positive_predictions'] is not None:
            for preds in evaluation_results['positive_predictions']:
                for pred_set in preds:
                    base_idx = next((x for x in pred_set if x in base_indices), None)
                    solvent_idx = next((x for x in pred_set if x in solvent_indices), None)
                    
                    if base_idx is not None and solvent_idx is not None:
                        base_name = base_df.loc[base_idx]['reagent']
                        solvent_name = solvent_df.loc[solvent_idx]['reagent']
                        predicted_base_solvent_pairs.append((base_name, solvent_name))
        
        if predicted_base_solvent_pairs:
            predicted_base_solvent_counts = pd.Series(predicted_base_solvent_pairs).value_counts().nlargest(12)
            
            logger.info("\nTop 12 Predicted Solvent-Base Tuples:")
            for (base, solvent), count in predicted_base_solvent_counts.items():
                logger.info(f"({base}, {solvent}): {count}")

            top_12_predicted_pairs = predicted_base_solvent_counts.index.tolist()
            
            # Compare with ground truth
            common_pairs = [pair for pair in top_12_predicted_pairs if pair in ground_truth_base_solvent_tuples]
            
            logger.info(f"\nNumber of Top-12 predicted solvent-base pairs in top-12 ground truth pairs: {len(common_pairs)} / 12")
            logger.info(f"Names of common pairs: {common_pairs}")
        else:
            logger.info("\nNo solvent-base pair predictions were made.") 


    # --- Generate Summary Report ---
    summary_report = {}
    if all_run_metrics:
        all_performances = [run_metrics['overall_performance'] for run_metrics in all_run_metrics]
        best_overall_performance = max(all_performances) if all_performances else -1.0
        best_run_index = all_performances.index(best_overall_performance) if all_performances else -1

        metric_values_by_name = defaultdict(list)
        is_simple_model = config["model_type"] in ['rxnfp', 'baselineofbaseline']
        
        for run_metrics in all_run_metrics:
            if 'overall_performance' in run_metrics:
                 metric_values_by_name['overall_performance'].append(run_metrics['overall_performance'])

            if is_simple_model:
                for split in ['positive', 'negative']:
                    if split in run_metrics and run_metrics[split] is not None:
                        for metric, value in run_metrics[split].items():
                            full_metric_name = f"{metric}_{split}_T1"
                            metric_values_by_name[full_metric_name].append(value)
            else: # VAE-like models
                T_values = run_metrics.get('T_values', [])
                for T in T_values:
                    for split in ['positive', 'negative']:
                        if split in run_metrics and T in run_metrics[split]:
                            for metric, value in run_metrics[split][T].items():
                                full_metric_name = f"{metric}_{split}_T{T}"
                                metric_values_by_name[full_metric_name].append(value)
        
        metric_summaries = {}
        for full_metric_name, values_list in metric_values_by_name.items():
            metric_summaries[full_metric_name] = {
                'mean': np.mean(values_list),
                'std_dev': np.std(values_list)
            }

        summary_report = {
            'best_run_index': best_run_index,
            'best_run_performance': best_overall_performance,
            'n_total_runs': len(all_run_metrics),
            'mean_overall_performance': np.mean(all_performances),
            'std_dev_overall_performance': np.std(all_performances),
            'metric_summaries': metric_summaries
        }
    

    # Structure the final report
    final_report = {
        'summary': summary_report,
        'config': {k: v for k, v in config.items() if not k.startswith('_') and 'path' not in k and not callable(v)}, # Cleaned config
        'detailed_runs': all_run_metrics
    }
    
    def numpy_serializer(obj):
        """ Custom JSON encoder for numpy types """
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



    if wandb_is_active and wandb.run and os.path.exists(best_model_path):
        artifact = wandb.Artifact(f'best_model_{config["model_type"]}', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a model on a specific data split and evaluate catalyst prediction.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to base config YAML file.")
    parser.add_argument("--test_smiles_list", type=str, nargs='+', required=True, help="List of reaction SMILES to use as the test set.")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load a pre-trained model.")
    parser.add_argument("--test_injection_percentage", type=float, default=0.0, help="Percentage of test set to inject into training set.")
    
    args, unknown = parser.parse_known_args()
    
    logger = setup_logging(verbose=True)

    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"FATAL: Base configuration file not found at {args.config_file}")
        exit(1)
    except yaml.YAMLError as exc:
        logger.error(f"FATAL: Error parsing YAML file {args.config_file}: {exc}")
        exit(1)

    config = flatten_wandb_config(config)

    for i in range(0, len(unknown), 2):
        key = unknown[i].replace("--", "")
        val = unknown[i+1]
        if key in config:
            config[key] = type(config[key])(val) if config[key] is not None else val

    config['test_smiles_list'] = args.test_smiles_list
    config['load_model_path'] = args.load_model_path
    config['test_injection_percentage'] = args.test_injection_percentage

    WANDB_ACTIVE = config.get("wandb", False) and wandb is not None

    current_wandb_run_object = None
    if WANDB_ACTIVE:
        logger.info("INFO: Weights & Biases is ACTIVE. Attempting to initialize...")
        try:
            wandb_project = config.get("wandb_project", "experiment_2")
            current_wandb_run_object = wandb.init(project=wandb_project, config=config)
            config = dict(wandb.config)
            logger.info(f"INFO: Weights & Biases INITIALIZED. Run: {current_wandb_run_object.name}")
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize Weights & Biases: {e}")
            WANDB_ACTIVE = False
    else:
        logger.info("INFO: Weights & Biases is DISABLED.")

    # Always run for one training/evaluation cycle
    success = retrain_algorithm(config=config, n_trainings=1, wandb_is_active=WANDB_ACTIVE)

    if WANDB_ACTIVE and current_wandb_run_object:
        if success:
            logger.info(f"INFO: W&B run {current_wandb_run_object.name} finishing.")
            current_wandb_run_object.finish(exit_code=0)
        else:
            if current_wandb_run_object._exit_code is None:
                 logger.info(f"INFO: W&B run {current_wandb_run_object.name} finishing due to unhandled error.")
                 current_wandb_run_object.finish(exit_code=1)
            else:
                 logger.info(f"INFO: W&B run {current_wandb_run_object.name} already finished with exit code {current_wandb_run_object._exit_code}.")

    logger.info("INFO: Script execution finished.")
