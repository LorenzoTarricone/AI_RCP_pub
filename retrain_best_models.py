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
from sklearn.model_selection import train_test_split

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
from utils.evaluate_model import evaluate_model
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
    
    # --- Load pre-defined split indices ---
    split_indices_path = f'data/split_indices_{config["rtype"]}_{config["data_type"]}.pkl'
    try:
        with open(split_indices_path, 'rb') as f:
            split_indices = pkl.load(f)
        test_indices = split_indices['test_indices']
        
        # Consolidate all non-test indices into a single training pool
        all_indices = np.arange(num_reactions)
        train_pool_indices = np.setdiff1d(all_indices, test_indices)

        logger.info(f"Loaded split indices from {split_indices_path}")

    except FileNotFoundError:
        logger.warning(f"Split indices file not found at {split_indices_path}. Falling back to random split.")
        # Using iterative stratification for a robust split
        test_indices, train_val_splits = iterative_stratified_split(
            all_reaction_labels, config["n_classes"],
            test_size=frac_tst, n_folds=1, # n_folds=1 gives one train/test split
            random_state=config["random_seed"]
        )
        train_pool_indices = train_val_splits[0][0] # The "training" part of the only split
    
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


    # Initialize test set consistency tracking
    first_run_test_hash = None
    
    for run_idx in range(n_trainings):
        logger.info(f"\n--- Starting Training Run {run_idx + 1}/{n_trainings} ---")
        
        # Verify test set consistency across runs
        test_set_hash = hash(tuple(sorted(test_indices)))
        if run_idx == 0:
            first_run_test_hash = test_set_hash
            logger.info(f"Run {run_idx+1}: Test set established with {len(test_indices)} samples (hash: {test_set_hash})")
        else:
            if test_set_hash != first_run_test_hash:
                logger.error(f"Run {run_idx+1}: Test set inconsistency detected! Hash {test_set_hash} != {first_run_test_hash}")
                raise RuntimeError("Test set changed between runs - this violates fair comparison!")
            logger.debug(f"Run {run_idx+1}: Test set consistency verified (hash: {test_set_hash})")

        # --- Set random seed for this run for model initialization and data shuffling ---
        if random_seed is not None:
            run_seed = random_seed + run_idx
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            # Also set CUDA seed for GPU reproducibility
            if torch.cuda.is_available():
                torch.cuda.manual_seed(run_seed)
                torch.cuda.manual_seed_all(run_seed)
            # Set PyTorch backend deterministic behavior (slower but more reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"Run {run_idx+1}: Set random seed to {run_seed} (with deterministic CUDA)")
        else:
            # To ensure different initializations if no seed is given
            torch.seed()
            np.random.seed(None) # Seed from OS
            logger.info(f"Run {run_idx+1}: No base random seed provided. Using fresh random initialization for this run.")

        # --- Split training pool into train and validation for proper early stopping ---
        # Use a small validation set even during final training to prevent overfitting
        val_size = config.get("final_val_size", 0.1)  # Default to 10% for validation
        use_early_stopping = config.get("use_early_stopping", True)  # Allow disabling if needed
        
        if use_early_stopping and len(train_pool_indices) > 10:  # Only split if we have enough data
            # Use the same stratification method as in train_gen_model.py
            stratification_method = config.get("stratification_method", "iterative")
            
            # Get labels for stratification
            train_pool_labels = [all_reaction_labels[i] for i in train_pool_indices]
            
            try:
                # Use the new dedicated stratified train/val split function
                from utils.trn_val_tst_sampling import stratified_train_val_split
                
                # Get relative indices within the training pool
                temp_train_idx, temp_val_idx = stratified_train_val_split(
                    train_pool_labels, 
                    config["n_classes"],
                    val_size=val_size,
                    stratification_method=stratification_method,
                    top_k=config.get('stratify_top_k', 5),
                    random_state=run_seed if random_seed is not None else None
                )
                
                # Map back to original indices
                train_indices_run = train_pool_indices[temp_train_idx]
                val_indices_run = train_pool_indices[temp_val_idx]
                
                logger.info(f"Run {run_idx+1}: Used {stratification_method} stratification via stratified_train_val_split")
                    
                # Verify no data loss and no overlap
                total_split_size = len(train_indices_run) + len(val_indices_run)
                overlap = len(set(train_indices_run) & set(val_indices_run))
                
                if total_split_size != len(train_pool_indices):
                    logger.error(f"Run {run_idx+1}: Data loss detected! Original: {len(train_pool_indices)}, Split: {total_split_size}")
                elif overlap > 0:
                    logger.error(f"Run {run_idx+1}: Data overlap detected! {overlap} samples in both train and val")
                else:
                    logger.info(f"Run {run_idx+1}: Split training pool using {stratification_method} stratification")
                    logger.info(f"Run {run_idx+1}: Reserved {len(val_indices_run)} samples ({val_size:.1%}) for validation")
                    logger.info(f"Run {run_idx+1}: ✓ No data loss, no overlap confirmed")
                
            except Exception as e:
                logger.warning(f"Run {run_idx+1}: Stratified split failed: {e}. Using simple random split.")
                train_indices_run, val_indices_run = train_test_split(
                    train_pool_indices, 
                    test_size=val_size, 
                    random_state=run_seed if random_seed is not None else None
                )
        else:
            # Fallback: use full training pool for both (original behavior)
            train_indices_run = train_pool_indices
            val_indices_run = train_pool_indices
            logger.warning(f"Run {run_idx+1}: Using full training pool for both training and validation (early stopping disabled or insufficient data)")

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

        train_mol_emb, val_mol_emb = None, None
        if config["model_type"] in ["emb", "seq_emb"]:
            train_mol_emb = [all_embeddings_mol[i] for i in train_indices_run]
            val_mol_emb = [all_embeddings_mol[i] for i in val_indices_run]
        
        trndata = GraphDataset(train_rmol_graphs, train_pmol_graphs, train_labels, train_smiles, train_mol_emb, config, split='trn', device=device)
        valdata = GraphDataset(val_rmol_graphs, val_pmol_graphs, val_labels, val_smiles, val_mol_emb, config, split='val', device=device)
        
        trn_loader = DataLoader(dataset=trndata, batch_size=config["batch_size"], shuffle=True, collate_fn=config["collate_fn"])
        val_loader = DataLoader(dataset=valdata, batch_size=config["batch_size"], shuffle=False, collate_fn=config["collate_fn"])

        # --- Model Training for the run ---
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

        model_path_run = f'./trained_models/retrain_run_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{run_identifier}_run_{run_idx}.pt'
        config["model_path"] = model_path_run
        
        if config["model_type"] == 'rxnfp': 
            net = Model(tstdata.fp_dim * 3 + 1, config["n_classes"])
            trainer = Trainer(net, cuda, config)
        elif (config["model_type"] == 'baseline'): 
            net = Model(tstdata.node_dim, tstdata.edge_dim, config["n_classes"])
            trainer = Trainer(net, cuda, config)
        elif config["model_type"] == 'seq':
            net = Model(config["rtype"], tstdata.node_dim, tstdata.edge_dim, config["n_classes"], config["n_info"])
            trainer = Trainer(net, cuda, config)
        elif config["model_type"] == 'emb':
            net = Model(tstdata.node_dim, tstdata.edge_dim, config["n_classes"], tstdata.emb_dim)
            trainer = Trainer(net, cuda, config)
        elif config["model_type"] == 'seq_emb':
            net = Model(config["rtype"], tstdata.node_dim, tstdata.edge_dim, config["n_classes"], config["n_info"], tstdata.emb_dim)
            trainer = Trainer(net, cuda, config)

        if config["mode"] == 'trn':
            trainer.training(trn_loader, val_loader, config["epochs"])
        elif config["mode"] == 'tst':
            trainer.load()

        # --- Evaluate on Test Set ---
        trainer.load() 
        evaluation_results = evaluate_model(trainer, tst_loader, tst_y_pos, tst_y_neg, config, run_idx, wandb_is_active, logger)

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

    # --- Compute Baselines ---
    if config.get("compute_random_baselines", False):
        logger.info("\n--- Computing Baselines on Test Set ---")
        
        T_values = config.get("T_values", [1, 10, 50, 100, 500, 1000])
        baseline_results_rows = []
        
        train_labels_full_pool = [all_reaction_labels[i] for i in train_pool_indices]

        for T in T_values:
            # --- Most Frequent Baseline (calculated once, used for all T) ---
            mf_acc_pos, mf_macro_pos, mf_micro_pos = compute_most_frequent_baseline(
                tst_y_pos, train_labels_full_pool, config["n_info"], config["rtype"], T=T
            )
            mf_acc_neg, mf_macro_neg, mf_micro_neg = (None, None, None)
            if config["data_type"] == "all":
                mf_acc_neg, mf_macro_neg, mf_micro_neg = compute_most_frequent_baseline(
                    tst_y_neg, train_labels_full_pool, config["n_info"], config["rtype"], T=T
                )

            # --- Random Baseline ---
            rand_acc_pos, rand_macro_pos, rand_micro_pos = compute_random_baseline(
                tst_y_pos, config["n_classes"], T=T, random_seed=config.get("random_seed", None)
            )
            baseline_results_rows.append({'T': T, 'baseline': 'random', 'split': 'pos', 'accuracy': rand_acc_pos, 'macro_recall': rand_macro_pos, 'micro_recall': rand_micro_pos})
            if config["data_type"] == "all":
                rand_acc_neg, rand_macro_neg, rand_micro_neg = compute_random_baseline(
                    tst_y_neg, config["n_classes"], T=T, random_seed=config.get("random_seed", None)
                )
                baseline_results_rows.append({'T': T, 'baseline': 'random', 'split': 'neg', 'accuracy': rand_acc_neg, 'macro_recall': rand_macro_neg, 'micro_recall': rand_micro_neg})
            
            # --- Structured Random Baseline ---
            s_rand_acc_pos, s_rand_macro_pos, s_rand_micro_pos = compute_structured_random_baseline(
                tst_y_pos, config["n_classes"], T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"]
            )
            baseline_results_rows.append({'T': T, 'baseline': 'structured_random', 'split': 'pos', 'accuracy': s_rand_acc_pos, 'macro_recall': s_rand_macro_pos, 'micro_recall': s_rand_micro_pos})
            if config["data_type"] == "all":
                s_rand_acc_neg, s_rand_macro_neg, s_rand_micro_neg = compute_structured_random_baseline(
                    tst_y_neg, config["n_classes"], T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"]
                )
                baseline_results_rows.append({'T': T, 'baseline': 'structured_random', 'split': 'neg', 'accuracy': s_rand_acc_neg, 'macro_recall': s_rand_macro_neg, 'micro_recall': s_rand_micro_neg})
            
            # --- Most Frequent Baseline (append results) ---
            baseline_results_rows.append({'T': T, 'baseline': 'most_frequent', 'split': 'pos', 'accuracy': mf_acc_pos, 'macro_recall': mf_macro_pos, 'micro_recall': mf_micro_pos})
            if config["data_type"] == "all":
                baseline_results_rows.append({'T': T, 'baseline': 'most_frequent', 'split': 'neg', 'accuracy': mf_acc_neg, 'macro_recall': mf_macro_neg, 'micro_recall': mf_micro_neg})

            # --- Frequency Chain Baseline ---
            fc_acc_pos, fc_macro_pos, fc_micro_pos = compute_frequency_chain_baseline(
                tst_y_pos, train_labels_full_pool, T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"]
            )
            baseline_results_rows.append({'T': T, 'baseline': 'freq_chain', 'split': 'pos', 'accuracy': fc_acc_pos, 'macro_recall': fc_macro_pos, 'micro_recall': fc_micro_pos})
            if config["data_type"] == "all":
                fc_acc_neg, fc_macro_neg, fc_micro_neg = compute_frequency_chain_baseline(
                    tst_y_neg, train_labels_full_pool, T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"], sample_neg=True
                )
                baseline_results_rows.append({'T': T, 'baseline': 'freq_chain', 'split': 'neg', 'accuracy': fc_acc_neg, 'macro_recall': fc_macro_neg, 'micro_recall': fc_micro_neg})

        # Save baseline results to CSV
        baseline_df = pd.DataFrame(baseline_results_rows)
        baseline_results_path = os.path.join(os.path.dirname(results_path), f'baselines_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{timestamp}.csv')
        baseline_df.to_csv(baseline_results_path, index=False)
        logger.info(f"Saved baseline results to {baseline_results_path}")

    # --- Save the best model and clean up other models ---
    if best_run_idx != -1:
        best_model_run_path = f'./trained_models/retrain_run_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{config["run_identifier"]}_run_{best_run_idx}.pt'
        if os.path.exists(best_model_run_path):
            shutil.copyfile(best_model_run_path, best_model_path)
            logger.info(f"Copied best model (from run {best_run_idx + 1}) to {best_model_path}")
        else:
            logger.warning(f"Best model file not found at {best_model_run_path}, could not copy.")

    for run_idx in range(n_trainings):
        model_path_to_delete = f'./trained_models/retrain_run_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{config["run_identifier"]}_run_{run_idx}.pt'
        try:
            if os.path.exists(model_path_to_delete):
                os.remove(model_path_to_delete)
                # logger.info(f"Cleaned up model from run {run_idx + 1}: {model_path_to_delete}")
        except OSError as e:
            logger.warning(f"Error deleting model {model_path_to_delete}: {e}")

    # --- Generate Summary Report ---
    summary_report = {}
    if all_run_metrics:
        all_performances = [run_metrics['overall_performance'] for run_metrics in all_run_metrics]
        best_overall_performance = max(all_performances) if all_performances else -1.0
        best_run_index = all_performances.index(best_overall_performance) if all_performances else -1

        metric_values_by_name = defaultdict(list)
        is_simple_model = config["model_type"] == 'rxnfp'
        
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
    
    # --- Generate detailed CSV report for all N runs ---
    if all_run_metrics:
        detailed_rows = []
        is_simple_model = config["model_type"] == 'rxnfp'
        for i, run_metrics in enumerate(all_run_metrics):
            row = {'run_index': i}
            if 'overall_performance' in run_metrics:
                row['overall_performance'] = run_metrics['overall_performance']
            
            if is_simple_model:
                for split in ['positive', 'negative']:
                    if split in run_metrics and run_metrics[split] is not None:
                        for metric, value in run_metrics[split].items():
                            row[f"{metric}_{split}_T1"] = value
            else: # VAE-like models
                T_values = run_metrics.get('T_values', [])
                for T in T_values:
                    for split in ['positive', 'negative']:
                        if split in run_metrics and T in run_metrics[split]:
                            for metric, value in run_metrics[split][T].items():
                                row[f"{metric}_{split}_T{T}"] = value
            detailed_rows.append(row)
            
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv_path = results_path.replace('.json', '_detailed.csv')
        detailed_df.to_csv(detailed_csv_path, index=False)
        logger.info(f"Saved detailed metrics for all runs to {detailed_csv_path}")

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

    with open(results_path, 'w') as f:
        json.dump(final_report, f, indent=4, default=numpy_serializer)
    logger.info(f"Saved all run metrics and summary to {results_path}")

    if wandb_is_active and wandb.run and os.path.exists(best_model_path):
        artifact = wandb.Artifact(f'best_model_{config["model_type"]}', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        logger.info(f"Uploaded best model to W&B: {best_model_path}")

    # Add model comparison guidance
    logger.info(f"\n--- MODEL COMPARISON GUIDANCE ---")
    if all_run_metrics:
        overall_performances = [run_metrics['overall_performance'] for run_metrics in all_run_metrics]
        mean_perf = np.mean(overall_performances)
        std_perf = np.std(overall_performances, ddof=1) if len(overall_performances) > 1 else 0.0
        
        logger.info(f"Overall Performance: {mean_perf:.4f} ± {std_perf:.4f}")
        logger.info(f"Coefficient of Variation: {std_perf/mean_perf:.3f}" if mean_perf != 0 else "Coefficient of Variation: inf")
        logger.info(f"Performance Range: [{min(overall_performances):.4f}, {max(overall_performances):.4f}]")
        
        if std_perf / mean_perf < 0.05:
            logger.info("✓ Model performance is highly stable across runs")
        elif std_perf / mean_perf < 0.10:
            logger.info("⚠ Model performance shows moderate variability")
        else:
            logger.info("⚠ Model performance shows high variability - consider more runs or better regularization")

    logger.info(f"\n--- Retrain Best Models Script Completed ---")
    logger.info(f"Best model from run {best_run_idx + 1} with performance {best_performance:.4f}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"For model comparison, use the detailed CSV: {results_path.replace('.json', '_detailed.csv')}")

    return True

if __name__ == "__main__":
    # Setup logging first, before any other operations
    logger = setup_logging(verbose=True)

    parser = argparse.ArgumentParser(description="Retrain and evaluate the best model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to base config YAML file.")
    parser.add_argument("--n_trainings", type=int, default=10, help="Number of times to train the model.")
    
    args, unknown = parser.parse_known_args()

    # Load the base config from the YAML file
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"FATAL: Base configuration file not found at {args.config_file}")
        exit(1)
    except yaml.YAMLError as exc:
        logger.error(f"FATAL: Error parsing YAML file {args.config_file}: {exc}")
        exit(1)

    # Flatten the config to handle wandb-style {value: ...} entries
    config = flatten_wandb_config(config)

    # This allows overriding config with CLI arguments, useful for wandb sweeps
    for i in range(0, len(unknown), 2):
        key = unknown[i].replace("--", "")
        val = unknown[i+1]
        if key in config:
            config[key] = type(config[key])(val) if config[key] is not None else val

    # 'config' is now the authoritative configuration for this specific run.
    logger = setup_logging(config.get('verbose', False))

    WANDB_ACTIVE = config.get("wandb", False) and wandb is not None

    current_wandb_run_object = None
    if WANDB_ACTIVE:
        logger.info("INFO: Weights & Biases is ACTIVE. Attempting to initialize...")
        try:
            wandb_project = config.get("wandb_project", "retrain_best_model")
            current_wandb_run_object = wandb.init(project=wandb_project, config=config)
            config = dict(wandb.config) # W&B provides the authoritative config
            logger.info(f"INFO: Weights & Biases INITIALIZED. Run: {current_wandb_run_object.name}")
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize Weights & Biases: {e}")
            logger.info("INFO: Proceeding with W&B effectively disabled for this run.")
            WANDB_ACTIVE = False
    else:
        logger.info("INFO: Weights & Biases is DISABLED.")

    # Execute the main processing logic
    success = retrain_algorithm(config=config, n_trainings=args.n_trainings, wandb_is_active=WANDB_ACTIVE)

    # Cleanly finish the W&B run
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
