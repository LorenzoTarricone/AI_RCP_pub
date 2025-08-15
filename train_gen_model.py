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

from utils.create_graphs import get_graph_data, load_graph_data
from utils.collate_functions import collate_reaction_graphs, collate_graphs_and_embeddings
from utils.dataset import GraphDataset, get_cardinalities_classes
from utils.evaluate_model import evaluate_model, update_fold_metrics, update_val_fold_metrics
from utils.trn_val_tst_sampling import (
    iterative_stratified_split, 
    stratify_by_condition_count, 
    stratify_by_frequent_conditions,
    simple_kfold_split,
    analyze_stratification_quality
)
from utils.miscellaneous import create_pos_neg_count_matrices
# Import the random baseline function
from utils.miscellaneous import compute_random_baseline, compute_structured_random_baseline, compute_frequency_chain_baseline
from utils.miscellaneous import compute_most_frequent_baseline

# Attempt to import wandb and handle if not installed
import wandb

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


def algorithm(config, wandb_is_active):
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

    if not os.path.exists('./trained_models/'): 
        os.makedirs('./trained_models/')
        logger.info(f"Created directory: ./trained_models/")
    
    # Create a unique identifier for this run to prevent model path conflicts
    if wandb_is_active and wandb.run:
        # Use the unique W&B run ID for sweep compatibility
        run_identifier = wandb.run.id
    else:
        # Fallback for non-W&B runs, using a timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_identifier = f'{config["iterid"]}_{timestamp}'
    
    base_model_path = f'./trained_models/model_{config["model_type"]}_{config["rtype"]}_{config["data_type"]}_{run_identifier}'
    config["base_model_path"] = base_model_path
    
 
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

    num_reactions = len(all_rmol_graphs)
    n_folds = config.get("n_folds", 5)  # Default to 5-fold CV if not specified
    
    # Get stratification method from config, default to "random" for backward compatibility
    stratification_method = config.get("stratification_method", "random")
    logger.info(f"Using stratification method: {stratification_method}")
    
    # Perform stratified sampling based on selected method
    if stratification_method == "random":
        # Original random approach (for backward compatibility)
        indices = np.arange(num_reactions)
        np.random.seed(config["random_seed"])
        np.random.shuffle(indices)
        
        test_size = int(np.floor(num_reactions * config["frac_tst"]))
        remaining_size = num_reactions - test_size
        
        # Calculate maximum possible folds based on remaining data size
        max_possible_folds = remaining_size
        if n_folds > max_possible_folds:
            logger.warning(f"Warning: Requested {n_folds} folds but only {max_possible_folds} possible")
            logger.warning(f"Reducing number of folds to {max_possible_folds}")
            n_folds = max_possible_folds
        
        # Simple k-fold split on remaining data
        test_indices = indices[remaining_size:]
        remaining_indices = indices[:remaining_size]
        fold_splits = simple_kfold_split(remaining_indices, n_folds, config["random_seed"])
        
    elif stratification_method == "iterative":
        test_indices, fold_splits = iterative_stratified_split(
            all_reaction_labels, config["n_classes"], 
            test_size=config["frac_tst"], n_folds=n_folds, 
            random_state=config["random_seed"]
        )
        test_size = len(test_indices)
        remaining_indices = [i for i in range(num_reactions) if i not in test_indices]
        remaining_size = len(remaining_indices)
    elif stratification_method == "condition_count":
        test_indices, fold_splits = stratify_by_condition_count(
            all_reaction_labels,
            test_size=config["frac_tst"], n_folds=n_folds,
            random_state=config["random_seed"]
        )
        test_size = len(test_indices)
        remaining_indices = [i for i in range(num_reactions) if i not in test_indices]
        remaining_size = len(remaining_indices)
    elif stratification_method == "frequent_conditions":
        test_indices, fold_splits = stratify_by_frequent_conditions(
            all_reaction_labels, config["n_classes"],
            test_size=config["frac_tst"], n_folds=n_folds,
            top_k=config.get('top_k_frequent_conditions', 5),
            random_state=config["random_seed"]
        )
        test_size = len(test_indices)
        remaining_indices = [i for i in range(num_reactions) if i not in test_indices]
        remaining_size = len(remaining_indices)
    else:
        raise ValueError(f"Unknown stratification_method: {stratification_method}")

    # --- SAVE SPLIT INDICES ---
    split_indices_path = config.get("split_indices_path")
    if split_indices_path:
        directory = os.path.dirname(split_indices_path)
        filename = os.path.basename(split_indices_path)
        name, ext = os.path.splitext(filename)
        
        rtype = config.get("rtype")
        data_type = config.get("data_type")
        
        if rtype and data_type:
            new_filename = f"{name}_{rtype}_{data_type}{ext}"
            split_indices_path = os.path.join(directory, new_filename)

        split_data_to_save = {
            'test_indices': test_indices,
            'fold_splits': fold_splits
        }
        os.makedirs(os.path.dirname(split_indices_path), exist_ok=True)
        with open(split_indices_path, 'wb') as f:
            pkl.dump(split_data_to_save, f)
        logger.info(f"Saved split indices to {split_indices_path}")
    # --------------------------

    # Analyze stratification quality if verbose
    if config.get("verbose"):
        try:
            analyze_stratification_quality(all_reaction_labels, config["n_classes"], test_indices, fold_splits)
        except Exception as e:
            logger.warning(f"Could not analyze stratification quality: {e}")
    
    # Update n_folds in case it was adjusted
    n_folds = len(fold_splits)

    # Initialize metrics storage for cross-validation
    # Get T values from config, with fallback to default values
    T_values = config.get("T_values", [1, 10, 50, 100, 500, 1000])
 

    if config["data_type"] == "positive":
        fold_metrics = {
            "accuracy_pos": {T: [] for T in T_values},
            "macro_recall_pos": {T: [] for T in T_values},
            "micro_recall_pos": {T: [] for T in T_values},
            "diversity_pos": {T: [] for T in T_values}
        }
        # Add validation metrics tracking for hyperparameter optimization
        val_fold_metrics = {
            "val_accuracy_pos": {T: [] for T in T_values},
            "val_macro_recall_pos": {T: [] for T in T_values},
            "val_micro_recall_pos": {T: [] for T in T_values},
            "val_diversity_pos": {T: [] for T in T_values}
        }
    else:  # "all" data
        fold_metrics = {
            "accuracy_pos": {T: [] for T in T_values},
            "macro_recall_pos": {T: [] for T in T_values},
            "micro_recall_pos": {T: [] for T in T_values},
            "diversity_pos": {T: [] for T in T_values},
            "accuracy_neg": {T: [] for T in T_values},
            "macro_recall_neg": {T: [] for T in T_values},
            "micro_recall_neg": {T: [] for T in T_values},
            "diversity_neg": {T: [] for T in T_values},
            "avg_inter_diversity": {T: [] for T in T_values}
        }
        # Add validation metrics tracking for hyperparameter optimization
        val_fold_metrics = {
            "val_accuracy_pos": {T: [] for T in T_values},
            "val_macro_recall_pos": {T: [] for T in T_values},
            "val_micro_recall_pos": {T: [] for T in T_values},
            "val_diversity_pos": {T: [] for T in T_values},
            "val_accuracy_neg": {T: [] for T in T_values},
            "val_macro_recall_neg": {T: [] for T in T_values},
            "val_micro_recall_neg": {T: [] for T in T_values},
            "val_diversity_neg": {T: [] for T in T_values},
            "val_avg_inter_diversity": {T: [] for T in T_values}
        }

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

    # Create a temporary dataset to get the dataset information
    # For emb and seq_emb models, we need to load molecular embeddings first
    temp_mol_emb = None
    if config["model_type"] in ["emb", "seq_emb"]:
        temp_mol_emb = [all_embeddings_mol[i] for i in remaining_indices]
    else:
        temp_mol_emb = None

    temp_trndata = GraphDataset([all_rmol_graphs[i] for i in remaining_indices], [all_pmol_graphs[i] for i in remaining_indices], 
                              [all_reaction_labels[i] for i in remaining_indices], [all_reaction_smiles[i] for i in remaining_indices], 
                              temp_mol_emb, config, split='trn', device=device)

    # Set n_classes from the dataset before using it
    config["n_classes"] = temp_trndata.n_classes
    config["rmol_max_cnt"] = temp_trndata.rmol_max_cnt
    config["pmol_max_cnt"] = temp_trndata.pmol_max_cnt

    # Get cardinalities for this fold BEFORE creating datasets (needed for expand_data=True models)
    # These cardinalities we know are always the same across all the dataset, so we can calculate them here
    n_info = get_cardinalities_classes(config)
    config["n_info"] = n_info


    # Print and log dataset information once
    logger.info('INFO: Number of classes: %d', config["n_classes"])
    logger.info('INFO: Number of reaction trn/val/tst: %d/%d/%d' %(remaining_size, remaining_size//n_folds, test_size))
    logger.info('INFO: Base model path: %s', base_model_path)
    logger.info('INFO (training): Total number of conditions: %d', temp_trndata.n_conditions)
    len_list = [len(l) for l in all_reaction_labels[:test_size]]
    logger.info('INFO (training): Number conditions per reaction (min/avg/max): %d/%.2f/%d'%(np.min(len_list), np.mean(len_list), np.max(len_list)))
    logger.info('INFO (training): Number of catalysts: %d' % n_info[0])
    logger.info('INFO (training): Number of solvent 1: %d' % n_info[1])
    logger.info('INFO (training): Number of solvent 2: %d' % n_info[2])
    logger.info('INFO (training): Number of additives: %d' % n_info[3])
    logger.info('INFO (training): Number of bases: %d' % n_info[4])



    if wandb_is_active and wandb.run:
        # Calculate average training and validation set sizes across folds
        avg_train_size = np.mean([len(train_idx) for train_idx, _ in fold_splits])
        avg_val_size = np.mean([len(val_idx) for _, val_idx in fold_splits])
        
        wandb.log({
            "reaction_type": config["rtype"],
            "data_type": config["data_type"],
            "num_total_reactions": num_reactions,
            "n_classes": config["n_classes"],
            "base_model_path": base_model_path,
            "total_n_conditions": temp_trndata.n_conditions, 
            "min_n_conditions": np.min(len_list), 
            "avg_n_conditions": np.mean(len_list), 
            "max_n_conditions": np.max(len_list),
            "n_reactions_train_avg": avg_train_size,
            "n_reactions_val_avg": avg_val_size,
            "n_reactions_test": len(test_indices),
            "n_folds": n_folds,
            "frac_tst_used": config["frac_tst"],
            "stratification_method": stratification_method,
            "stratify_top_k": config.get("stratify_top_k", 5) if stratification_method == "frequent_conditions" else None,
            "random_seed_used": random_seed if random_seed is not None else "None",
            "device": str(device),
            "iterid": config["iterid"],
            "model_type": config["model_type"],
            "use_rxnfp": config["use_rxnfp"],
            "expand_data": config["expand_data"],
            "emb_to_use": config["emb_to_use"],
            "n_cats": n_info[0],
            "n_base": n_info[1],
            "n_sol_1": n_info[2],
            "n_add": n_info[3],
            "n_sol_2": n_info[4]
        })


    # Perform k-fold cross-validation using the pre-computed fold splits
    for fold, (train_indices, val_indices) in enumerate(tqdm(fold_splits, desc="Processing folds", unit="fold")):
        logger.info(f"\n--- Processing Fold {fold + 1}/{n_folds} ---")

        # --- Create a unique model path for this specific fold ---
        model_path = f'{config["base_model_path"]}_fold_{fold}.pt'
        config["model_path"] = model_path
        logger.info(f'Model path for this fold: {model_path}')
        # ---------------------------------------------------------

        # Training sets
        train_rmol_graphs = [all_rmol_graphs[i] for i in train_indices]
        train_pmol_graphs = [all_pmol_graphs[i] for i in train_indices]
        train_labels = [all_reaction_labels[i] for i in train_indices]
        train_smiles = [all_reaction_smiles[i] for i in train_indices]

        # Validation sets
        val_rmol_graphs = [all_rmol_graphs[i] for i in val_indices]
        val_pmol_graphs = [all_pmol_graphs[i] for i in val_indices]
        val_labels = [all_reaction_labels[i] for i in val_indices]
        val_smiles = [all_reaction_smiles[i] for i in val_indices]

        # Test sets
        test_rmol_graphs = [all_rmol_graphs[i] for i in test_indices]
        test_pmol_graphs = [all_pmol_graphs[i] for i in test_indices]
        test_labels = [all_reaction_labels[i] for i in test_indices]
        test_smiles = [all_reaction_smiles[i] for i in test_indices]

        
        logger.info(f"INFO: Number of training samples: {len(train_smiles)}")
        logger.info(f"INFO: Number of validation samples: {len(val_smiles)}")
        logger.info(f"INFO: Number of testing samples: {len(test_smiles)}")


        # do the split also for the molecular embedding if present 
        if config["model_type"] in ["emb", "seq_emb"]:
            train_mol_emb = [all_embeddings_mol[i] for i in train_indices]
            val_mol_emb = [all_embeddings_mol[i] for i in val_indices]
            test_mol_emb = [all_embeddings_mol[i] for i in test_indices]
            if config["verbose"]:
                print(f"INFO: Number of training molecular embeddings: {len(train_mol_emb)}")
                print(f"INFO: Number of validation molecular embeddings: {len(val_mol_emb)}")
                print(f"INFO: Number of testing molecular embeddings: {len(test_mol_emb)}")
        else:
            train_mol_emb = None
            val_mol_emb = None
            test_mol_emb = None

        # Create datasets and dataloaders for this fold
        trndata = GraphDataset(train_rmol_graphs, train_pmol_graphs, train_labels, train_smiles, train_mol_emb, config, split='trn', device=device)
        valdata = GraphDataset(val_rmol_graphs, val_pmol_graphs, val_labels, val_smiles, val_mol_emb, config, split='val', device=device)
        tstdata = GraphDataset(test_rmol_graphs, test_pmol_graphs, test_labels, test_smiles, test_mol_emb, config, split='tst', device=device)

        trn_loader = DataLoader(dataset=trndata, batch_size=config["batch_size"], shuffle=True, collate_fn=config["collate_fn"])
        val_loader = DataLoader(dataset=valdata, batch_size=config["batch_size"], shuffle=False, collate_fn=config["collate_fn"])
        tst_loader = DataLoader(dataset=tstdata, batch_size=config["batch_size"], shuffle=False, collate_fn=config["collate_fn"])

        # Update config with dataset information
        config["rmol_max_cnt"] = tstdata.rmol_max_cnt
        config["pmol_max_cnt"] = tstdata.pmol_max_cnt

        if config["class_weights"]:
            pos_count_matrix_1, pos_count_matrix_0, neg_count_matrix_1, neg_count_matrix_0 = create_pos_neg_count_matrices(train_labels, config["n_classes"])
            config["train_pos_count_matrix_1"] = pos_count_matrix_1
            config["train_pos_count_matrix_0"] = pos_count_matrix_0
            config["train_neg_count_matrix_1"] = neg_count_matrix_1
            config["train_neg_count_matrix_0"] = neg_count_matrix_0
        else:
            config["train_pos_count_matrix"] = None
            config["train_neg_count_matrix"] = None

        # Initialize model and trainer for this fold
        if config["model_type"] == 'rxnfp': 
            net = Model(tstdata.fp_dim * 3 + 1, config["n_classes"])
            trainer = Trainer(net, cuda, config)
        elif (config["model_type"] == 'baseline') or (config["model_type"] == 'baselineofbaseline'): 
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
        else:
            err_msg = f"ERROR: Unrecognized model_type '{config['model_type']}' during model initialization. Valid options are: 'baseline', 'rxnfp', 'seq', 'emb', 'seq_emb'"
            logger.error(err_msg)
            if wandb_is_active and wandb.run:
                wandb.log({"error_message": err_msg})
                wandb.run.finish(exit_code=1)
            return False

        # TODO: IMPORTANT - Consider class imbalance handling
        # Current models don't explicitly handle class imbalance which is common in chemical datasets
        # Consider implementing:
        # - Weighted loss functions (class_weight parameter)
        # - SMOTE or other oversampling techniques
        # - Focal loss for severe imbalance
        # - Balanced accuracy metrics alongside standard metrics

        # Train the model for this fold
        if config["mode"] == 'trn':
            trainer.training(trn_loader, val_loader, config["epochs"])
        elif config["mode"] == 'tst':
            trainer.load()

        # Evaluate on test set
        tst_y = tstdata.y
        if config["data_type"] == "positive":
            tst_y_pos = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in tst_y]
            tst_y_neg = None
        else:  # "all" data
            tst_y_pos = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in tst_y]
            tst_y_neg = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 0] for sublist in tst_y]

        # Evaluate the model on TEST set (for final evaluation)
        evaluation_results = evaluate_model(trainer, tst_loader, tst_y_pos, tst_y_neg, config, fold, wandb_is_active, logger)
        
        # Prepare validation labels for evaluation
        val_y = valdata.y
        if config["data_type"] == "positive":
            val_y_pos_val = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in val_y]
            val_y_neg_val = None
        else:  # "all" data
            val_y_pos_val = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in val_y]
            val_y_neg_val = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 0] for sublist in val_y]

        # Evaluate the model on VALIDATION set (for hyperparameter optimization)
        val_evaluation_results = evaluate_model(trainer, val_loader, val_y_pos_val, val_y_neg_val, config, fold, wandb_is_active, logger)
        
        # Update fold metrics with test results
        fold_metrics = update_fold_metrics(fold_metrics, evaluation_results, config, fold)
        
        # Update validation fold metrics for hyperparameter optimization
        val_fold_metrics = update_val_fold_metrics(val_fold_metrics, val_evaluation_results, config, fold)

    # Calculate and log mean and std of metrics across folds for each T value
    if wandb_is_active and wandb.run:
        # First log the raw fold metrics (flattened for better readability)
        flattened_raw_metrics = {}
        for metric_name, T_dict in fold_metrics.items():
            for T, values in T_dict.items():
                flattened_raw_metrics[f"raw_T{T}_{metric_name}"] = values
        wandb.log({"raw_fold_metrics": flattened_raw_metrics})

        # Find max value of T in fold_metrics
        if config["model_type"] == "rxnfp":
            max_T = 1
        else:
            max_T = max(max(T_dict.keys()) for T_dict in fold_metrics.values())
        
        print("############## TEST METRICS ##############")
        # Calculate aggregated TEST metrics (for reporting final performance)
        aggregated_metrics = {}
        for metric_name, T_dict in fold_metrics.items():
            for T, values in T_dict.items():
                if values:  # Only calculate if we have values for this T
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    aggregated_metrics[f"mean_T{T}_{metric_name}"] = mean_value
                    aggregated_metrics[f"std_T{T}_{metric_name}"] = std_value

                    logger.info(f"\nT={T} {metric_name}:")
                    logger.info(f"Mean: {mean_value:.4f} ± {std_value:.4f}")
                    print(f"\nT={T} {metric_name}:")
                    print(f"Mean: {mean_value:.4f} ± {std_value:.4f}")
        
        print("############## VALIDATION METRICS ##############")
        # Calculate aggregated VALIDATION metrics (for hyperparameter optimization)
        val_aggregated_metrics = {}
        for metric_name, T_dict in val_fold_metrics.items():
            for T, values in T_dict.items():
                if values:  # Only calculate if we have values for this T
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    val_aggregated_metrics[f"mean_T{T}_{metric_name}"] = mean_value
                    val_aggregated_metrics[f"std_T{T}_{metric_name}"] = std_value

        print("###############################################")
        print(aggregated_metrics)
        print(val_aggregated_metrics)
        

        # Log all aggregated metrics at once
        wandb.log(aggregated_metrics)
        wandb.log(val_aggregated_metrics)

        # Log raw metrics for validation set as well
        flattened_raw_val_metrics = {}
        for metric_name, T_dict in val_fold_metrics.items():
            for T, values in T_dict.items():
                flattened_raw_val_metrics[f"raw_T{T}_{metric_name}"] = values
        wandb.log({"raw_val_fold_metrics": flattened_raw_val_metrics})


        # CORRECT: Use VALIDATION performance for hyperparameter optimization
        if config["data_type"] == "positive":
            accuracy_final = (val_aggregated_metrics[f"mean_T{max_T}_val_accuracy_pos"] + val_aggregated_metrics[f"mean_T{max_T}_val_macro_recall_pos"] + val_aggregated_metrics[f"mean_T{max_T}_val_micro_recall_pos"])/3
        else:
            accuracy_final = (val_aggregated_metrics[f"mean_T{max_T}_val_accuracy_pos"] + val_aggregated_metrics[f"mean_T{max_T}_val_macro_recall_pos"] + val_aggregated_metrics[f"mean_T{max_T}_val_micro_recall_pos"] + val_aggregated_metrics[f"mean_T{max_T}_val_accuracy_neg"] + val_aggregated_metrics[f"mean_T{max_T}_val_macro_recall_neg"] + val_aggregated_metrics[f"mean_T{max_T}_val_micro_recall_neg"])/6
        
        # This is using validation performance for hyperparameter optimization
        wandb.log({"accuracy_final": accuracy_final})
        
        # Log test performance separately for final evaluation 
        if config["data_type"] == "positive":
            test_accuracy_final = (aggregated_metrics[f"mean_T{max_T}_accuracy_pos"] + aggregated_metrics[f"mean_T{max_T}_macro_recall_pos"] + aggregated_metrics[f"mean_T{max_T}_micro_recall_pos"])/3
        else:
            test_accuracy_final = (aggregated_metrics[f"mean_T{max_T}_accuracy_pos"] + aggregated_metrics[f"mean_T{max_T}_macro_recall_pos"] + aggregated_metrics[f"mean_T{max_T}_micro_recall_pos"] + aggregated_metrics[f"mean_T{max_T}_accuracy_neg"] + aggregated_metrics[f"mean_T{max_T}_macro_recall_neg"] + aggregated_metrics[f"mean_T{max_T}_micro_recall_neg"])/6
        wandb.log({"test_accuracy_final": test_accuracy_final})

    # Optionally compute and log the random baseline
    if config.get("compute_random_baselines", False):
        # Use the same T values as the main metric
        T_values = config.get("T_values", [1, 10, 50, 100, 500, 1000])
        results_rows = []
        # For frequency chain and most frequent baselines, we need to run n_folds times (using each fold's train set)
        n_folds = len(fold_splits)
        
        for T in T_values:
            # --- Most frequent condition baseline (per fold, then average) ---
            most_freq_acc_pos_list, most_freq_macro_pos_list, most_freq_micro_pos_list = [], [], []
            if config["data_type"] == "all":
                most_freq_acc_neg_list, most_freq_macro_neg_list, most_freq_micro_neg_list = [], [], []

            for fold, (train_indices, val_indices) in enumerate(fold_splits):
                train_labels_fold = [all_reaction_labels[i] for i in train_indices]
                
                mf_acc_pos, mf_macro_pos, mf_micro_pos = compute_most_frequent_baseline(
                    tst_y_pos, train_labels_fold, config["n_info"], config["rtype"], T=T
                )
                most_freq_acc_pos_list.append(mf_acc_pos)
                most_freq_macro_pos_list.append(mf_macro_pos)
                most_freq_micro_pos_list.append(mf_micro_pos)
                
                if config["data_type"] == "all":
                    mf_acc_neg, mf_macro_neg, mf_micro_neg = compute_most_frequent_baseline(
                        tst_y_neg, train_labels_fold, config["n_info"], config["rtype"], T=T
                    )
                    most_freq_acc_neg_list.append(mf_acc_neg)
                    most_freq_macro_neg_list.append(mf_macro_neg)
                    most_freq_micro_neg_list.append(mf_micro_neg)
            
            # --- Random baseline (independent 50% per class) ---
            random_acc_pos, random_macro_pos, random_micro_pos = compute_random_baseline(
                tst_y_pos, config["n_classes"], T=T, random_seed=config.get("random_seed", None)
            )
            results_rows.append({
                "T": T, "baseline": "random", "split": "pos",
                "accuracy_mean": random_acc_pos, "macro_recall_mean": random_macro_pos, "micro_recall_mean": random_micro_pos,
                "accuracy_std": 0, "macro_recall_std": 0, "micro_recall_std": 0
            })
            if config["data_type"] == "all":
                random_acc_neg, random_macro_neg, random_micro_neg = compute_random_baseline(
                    tst_y_neg, config["n_classes"], T=T, random_seed=config.get("random_seed", None)
                )
                results_rows.append({
                    "T": T, "baseline": "random", "split": "neg",
                    "accuracy_mean": random_acc_neg, "macro_recall_mean": random_macro_neg, "micro_recall_mean": random_micro_neg,
                    "accuracy_std": 0, "macro_recall_std": 0, "micro_recall_std": 0
                })

            # --- Structured random baseline ---
            random_structured_acc_pos, random_structured_macro_pos, random_structured_micro_pos = compute_structured_random_baseline(
                tst_y_pos, config["n_classes"], T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"]
            )
            results_rows.append({
                "T": T, "baseline": "structured_random", "split": "pos",
                "accuracy_mean": random_structured_acc_pos, "macro_recall_mean": random_structured_macro_pos, "micro_recall_mean": random_structured_micro_pos,
                "accuracy_std": 0, "macro_recall_std": 0, "micro_recall_std": 0
            })
            if config["data_type"] == "all":
                random_structured_acc_neg, random_structured_macro_neg, random_structured_micro_neg = compute_structured_random_baseline(
                    tst_y_neg, config["n_classes"], T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"]
                )
                results_rows.append({
                    "T": T, "baseline": "structured_random", "split": "neg",
                    "accuracy_mean": random_structured_acc_neg, "macro_recall_mean": random_structured_macro_neg, "micro_recall_mean": random_structured_micro_neg, 
                    "accuracy_std": 0, "macro_recall_std": 0, "micro_recall_std": 0
                })
            
            # --- Most frequent condition baseline (averaged over folds) ---
            results_rows.append({
                "T": T, "baseline": "most_frequent", "split": "pos",
                "accuracy_mean": np.mean(most_freq_acc_pos_list), "accuracy_std": np.std(most_freq_acc_pos_list),
                "macro_recall_mean": np.mean(most_freq_macro_pos_list), "macro_recall_std": np.std(most_freq_macro_pos_list),
                "micro_recall_mean": np.mean(most_freq_micro_pos_list), "micro_recall_std": np.std(most_freq_micro_pos_list),
                "accuracy_raw": most_freq_acc_pos_list, "macro_recall_raw": most_freq_macro_pos_list, "micro_recall_raw": most_freq_micro_pos_list
            })
            if config["data_type"] == "all":
                results_rows.append({
                    "T": T, "baseline": "most_frequent", "split": "neg",
                    "accuracy_mean": np.mean(most_freq_acc_neg_list), "accuracy_std": np.std(most_freq_acc_neg_list),
                    "macro_recall_mean": np.mean(most_freq_macro_neg_list), "macro_recall_std": np.std(most_freq_macro_neg_list),
                    "micro_recall_mean": np.mean(most_freq_micro_neg_list), "micro_recall_std": np.std(most_freq_micro_neg_list),
                    "accuracy_raw": most_freq_acc_neg_list, "macro_recall_raw": most_freq_macro_neg_list, "micro_recall_raw": most_freq_micro_neg_list
                })

            # --- Frequency chain baseline: run n_folds times, each with the corresponding train set ---
            freq_chain_acc_pos_list, freq_chain_macro_pos_list, freq_chain_micro_pos_list = [], [], []
            for fold, (train_indices, val_indices) in enumerate(fold_splits):
                train_labels_fold  = [all_reaction_labels[i] for i in train_indices]
                acc, macro, micro = compute_frequency_chain_baseline(
                    tst_y_pos, train_labels_fold, T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"]
                )
                freq_chain_acc_pos_list.append(acc)
                freq_chain_macro_pos_list.append(macro)
                freq_chain_micro_pos_list.append(micro)
            
            results_rows.append({
                "T": T, "baseline": "freq_chain", "split": "pos",
                "accuracy_mean": np.mean(freq_chain_acc_pos_list), "accuracy_std": np.std(freq_chain_acc_pos_list),
                "macro_recall_mean": np.mean(freq_chain_macro_pos_list), "macro_recall_std": np.std(freq_chain_macro_pos_list),
                "micro_recall_mean": np.mean(freq_chain_micro_pos_list), "micro_recall_std": np.std(freq_chain_micro_pos_list),
                "accuracy_raw": freq_chain_acc_pos_list, "macro_recall_raw": freq_chain_macro_pos_list, "micro_recall_raw": freq_chain_micro_pos_list
            })
            
            if config["data_type"] == "all":
                freq_chain_acc_neg_list, freq_chain_macro_neg_list, freq_chain_micro_neg_list = [], [], []
                for fold, (train_indices, val_indices) in enumerate(fold_splits):
                    train_labels_fold = [all_reaction_labels[i] for i in train_indices]
                    acc, macro, micro = compute_frequency_chain_baseline(
                        tst_y_neg, train_labels_fold, T=T, random_seed=config.get("random_seed", None), rtype=config["rtype"], n_info=config["n_info"], sample_neg=True
                    )
                    freq_chain_acc_neg_list.append(acc)
                    freq_chain_macro_neg_list.append(macro)
                    freq_chain_micro_neg_list.append(micro)
                
                results_rows.append({
                    "T": T, "baseline": "freq_chain", "split": "neg",
                    "accuracy_mean": np.mean(freq_chain_acc_neg_list), "accuracy_std": np.std(freq_chain_acc_neg_list),
                    "macro_recall_mean": np.mean(freq_chain_macro_neg_list), "macro_recall_std": np.std(freq_chain_macro_neg_list),
                    "micro_recall_mean": np.mean(freq_chain_micro_neg_list), "micro_recall_std": np.std(freq_chain_micro_neg_list),
                    "accuracy_raw": freq_chain_acc_neg_list, "macro_recall_raw": freq_chain_macro_neg_list, "micro_recall_raw": freq_chain_micro_neg_list
                })

        # Save results to CSV
        dataset_type = f"{config['rtype']}_{config['data_type']}_{config['frac_tst']}_{config['stratification_method']}"
        df = pd.DataFrame(results_rows)
        df.to_csv(f"{dataset_type}_random_baselines.csv", index=False)
        logger.info(f"INFO: Random baselines saved to {dataset_type}_random_baselines.csv")


        
    return True # Indicate success

if __name__ == "__main__":
    # Setup logging first, before any other operations
    logger = setup_logging(verbose=True)  # Start with verbose logging for initialization

    # 1. Define all possible arguments that can be passed from the sweep.
    # This allows the script to accept hyperparameters from the W&B agent.
    parser = argparse.ArgumentParser(description="Process reaction data with optional W&B tracking.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to base config YAML file.")
    
    # Arguments to be overridden by the wrapper script via the sweep
    parser.add_argument("--model_type", type=str, help="Type of model to use (from sweep).")
    parser.add_argument("--filepath", type=str, help="Path to data file (from sweep).")

    # All tunable hyperparameters from sweep_config.yaml must be defined here
    parser.add_argument("--lr", type=float, help="Learning rate (from sweep).")
    parser.add_argument("--weight_decay", type=float, help="Weight decay (from sweep).")
    parser.add_argument("--weight_bce", type=float, help="BCE weight (from sweep).")
    parser.add_argument("--batch_size", type=int, help="Batch size (from sweep).")
    parser.add_argument("--epochs", type=int, help="Number of epochs (from sweep).")
    parser.add_argument("--treshold", type=float, help="Treshold (from sweep).")
    parser.add_argument("--temperature", type=float, help="Temperature (from sweep).")
    # A small helper function is used to correctly parse boolean values from the command line
    parser.add_argument("--class_weights", type=lambda x: (str(x).lower() == 'true'), help="Use class weights (from sweep).")
    parser.add_argument("--kl_annealing_epochs", type=int, help="KL annealing epochs (from sweep).")
    parser.add_argument("--ss_decay_epochs", type=int, help="Scheduled sampling decay epochs (from sweep).")

    # 2. Parse arguments
    args = parser.parse_args()

    # 3. Load the base config from the YAML file
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"FATAL: Base configuration file not found at {args.config_file}")
        exit(1)
    except yaml.YAMLError as exc:
        logger.error(f"FATAL: Error parsing YAML file {args.config_file}: {exc}")
        exit(1)

    # 4. Override base config with arguments passed from the command line (by the W&B agent).
    # This creates a dictionary from the parsed args, excluding any that were not provided (i.e., are None),
    # and updates the base config with them. This is the crucial step.
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cli_args)
    
    # 'config' is now the authoritative configuration for this specific run.

    # Update logging level based on config
    logger = setup_logging(config.get('verbose', False))

    WANDB_ACTIVE = config["wandb"]

    if WANDB_ACTIVE:
        logger.info("INFO: Weights & Biases is ACTIVE. Attempting to initialize...")
        try:
            # For W&B sweeps, agent passes params which override 'config'.
            # For single runs, 'config' is used directly.
            # Set project/entity via environment variables (WANDB_PROJECT, WANDB_ENTITY)
            # or pass them as arguments: wandb.init(project="my_project", entity="my_entity", config=...)
            wandb_project = config["model_type"]
            current_wandb_run_object = wandb.init(project=wandb_project, config=config)
            effective_config = dict(current_wandb_run_object.config) # W&B provides the authoritative config
            logger.info(f"INFO: Weights & Biases INITIALIZED. Run: {current_wandb_run_object.name}")
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize Weights & Biases: {e}")
            logger.info("INFO: Proceeding with W&B effectively disabled for this run.")
            WANDB_ACTIVE = False # Disable W&B if init fails
            effective_config = config # Fallback to local config
    else:
        effective_config = config
        logger.info("INFO: Weights & Biases is DISABLED (to enable it, please do so through the YAML file).")

    # Execute the main processing logic
    success = algorithm(config=effective_config, wandb_is_active=WANDB_ACTIVE)

    # Cleanly finish the W&B run if it was started and processing ended
    if WANDB_ACTIVE and current_wandb_run_object:
        if success:
            logger.info(f"INFO: W&B run {current_wandb_run_object.name} finishing.")
            current_wandb_run_object.finish(exit_code=0) # Explicitly set exit code for success
        else:
            # If algorithm returned False, it should have already called
            # current_wandb_run_object.finish(exit_code=1) for errors it handled.
            # This is a fallback if it didn't.
            if current_wandb_run_object.exit_code is None : # Check if exit code not already set
                 logger.info(f"INFO: W&B run {current_wandb_run_object.name} finishing due to unhandled error in core logic.")
                 current_wandb_run_object.finish(exit_code=1)
            else:
                 logger.info(f"INFO: W&B run {current_wandb_run_object.name} already finished with exit code {current_wandb_run_object.exit_code}.")

    logger.info("INFO: Script execution finished.")
        