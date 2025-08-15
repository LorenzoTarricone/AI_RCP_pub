import argparse
import numpy as np
import os
import pickle as pkl
import pandas as pd
from datetime import datetime
import yaml
import logging
from utils.create_graphs import get_graph_data, load_graph_data
from utils.dataset import get_cardinalities_classes

from utils.trn_val_tst_sampling import iterative_stratified_split
from utils.miscellaneous import (
    compute_random_baseline,
    compute_structured_random_baseline,
    compute_frequency_chain_baseline,
    compute_most_frequent_baseline,
)

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

def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def compute_baselines(config, n_runs):
    """
    Loads data, splits it, and computes baseline model performances over N runs.
    """
    logger = logging.getLogger(__name__)

    # ############################## LOAD DATA ################################
    file_path = config.get('filepath')
    if not file_path:
        logger.error("ERROR: 'filepath' is a required parameter.")
        return

    try:
        frac_tst = float(config.get('frac_tst'))
    except (ValueError, TypeError):
        logger.error("ERROR: 'frac_tst' must be a number.")
        return
        
    base_random_seed = config.get('random_seed')
    if base_random_seed is not None:
        np.random.seed(base_random_seed)
        logger.info(f"Using base random seed for data splitting: {base_random_seed}")

    if "bh" in file_path:
        config["rtype"] = 'bh'
    elif "sm" in file_path:
        config["rtype"] = 'sm'
    else:
        logger.error(f"ERROR: Reaction type not recognized from filepath: {file_path}")
        return

    if "all_all" in file_path or "all_positive" in file_path:
        config["data_type"] = "all" if "all_all" in file_path else "positive"
    else:
        logger.error(f"ERROR: Data type (all/positive) not recognized from filepath: {file_path}")
        return

    try:
        loaded_npz = np.load(file_path, allow_pickle=True)
        reaction_data_unpacked = loaded_npz['data']
        reaction_dict, clist = reaction_data_unpacked
        config["clist"] = clist
    except Exception as e:
        logger.error(f"ERROR: Loading or processing data from {file_path}: {e}")
        return

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


    num_reactions = len(all_reaction_labels)
    config["n_classes"] = len(clist)

    # ############################# DATA SPLITS (once) ##############################
    logger.info("--- Splitting data into a fixed train/test set ---")
    split_indices_path = f'data/split_indices_{config["rtype"]}_{config["data_type"]}.pkl'
    try:
        with open(split_indices_path, 'rb') as f:
            split_indices = pkl.load(f)
        test_indices = split_indices['test_indices']
        all_indices = np.arange(num_reactions)
        train_pool_indices = np.setdiff1d(all_indices, test_indices)
        logger.info(f"Loaded split indices from {split_indices_path}")
    except FileNotFoundError:
        logger.warning(f"Split indices file not found. Falling back to random split.")
        test_indices, train_val_splits = iterative_stratified_split(
            all_reaction_labels, config["n_classes"],
            test_size=frac_tst, n_folds=1,
            random_state=base_random_seed
        )
        train_pool_indices = train_val_splits[0][0]

    logger.info(f"Total reactions: {num_reactions}")
    logger.info(f"Training pool size: {len(train_pool_indices)}")
    logger.info(f"Test set size: {len(test_indices)}")

    test_labels = [all_reaction_labels[i] for i in test_indices]
    train_labels_full_pool = [all_reaction_labels[i] for i in train_pool_indices]

    if config["data_type"] == "positive":
        tst_y_pos = [[item[0] for item in sublist if item[1] == 1] for sublist in test_labels]
        tst_y_neg = None
    else:  # "all" data
        tst_y_pos = [[item[0] for item in sublist if item[1] == 1] for sublist in test_labels]
        tst_y_neg = [[item[0] for item in sublist if item[1] == 0] for sublist in test_labels]

    config["n_info"] = get_cardinalities_classes(config)

    # ############################ COMPUTE BASELINES (N times) ###########################
    all_results_rows = []
    T_values = config.get("T_values", [1, 10, 50, 100, 500, 1000])

    for run_idx in range(n_runs):
        logger.info(f"\n--- Starting Baseline Run {run_idx + 1}/{n_runs} ---")
        
        run_seed = None
        if base_random_seed is not None:
            run_seed = base_random_seed + run_idx
            np.random.seed(run_seed)
            logger.info(f"Run {run_idx+1}: Set random seed to {run_seed}")
        else:
            np.random.seed(None)
            logger.info(f"Run {run_idx+1}: No base random seed. Using fresh random initialization.")

        for T in T_values:
            run_row_base = {'run_idx': run_idx, 'T': T}
            # --- Most Frequent Baseline ---
            mf_acc_pos, mf_macro_pos, mf_micro_pos = compute_most_frequent_baseline(
                tst_y_pos, train_labels_full_pool, config["n_info"], config["rtype"], T=T
            )
            all_results_rows.append({**run_row_base, 'baseline': 'most_frequent', 'split': 'pos', 'accuracy': mf_acc_pos, 'macro_recall': mf_macro_pos, 'micro_recall': mf_micro_pos})
            if config["data_type"] == "all":
                mf_acc_neg, mf_macro_neg, mf_micro_neg = compute_most_frequent_baseline(
                    tst_y_neg, train_labels_full_pool, config["n_info"], config["rtype"], T=T
                )
                all_results_rows.append({**run_row_base, 'baseline': 'most_frequent', 'split': 'neg', 'accuracy': mf_acc_neg, 'macro_recall': mf_macro_neg, 'micro_recall': mf_micro_neg})

            # --- Random Baseline ---
            rand_acc_pos, rand_macro_pos, rand_micro_pos = compute_random_baseline(
                tst_y_pos, config["n_classes"], T=T, random_seed=run_seed
            )
            all_results_rows.append({**run_row_base, 'baseline': 'random', 'split': 'pos', 'accuracy': rand_acc_pos, 'macro_recall': rand_macro_pos, 'micro_recall': rand_micro_pos})
            if config["data_type"] == "all":
                rand_acc_neg, rand_macro_neg, rand_micro_neg = compute_random_baseline(
                    tst_y_neg, config["n_classes"], T=T, random_seed=run_seed
                )
                all_results_rows.append({**run_row_base, 'baseline': 'random', 'split': 'neg', 'accuracy': rand_acc_neg, 'macro_recall': rand_macro_neg, 'micro_recall': rand_micro_neg})
            
            # --- Structured Random Baseline ---
            s_rand_acc_pos, s_rand_macro_pos, s_rand_micro_pos = compute_structured_random_baseline(
                tst_y_pos, config["n_classes"], T=T, random_seed=run_seed, rtype=config["rtype"], n_info=config["n_info"]
            )
            all_results_rows.append({**run_row_base, 'baseline': 'structured_random', 'split': 'pos', 'accuracy': s_rand_acc_pos, 'macro_recall': s_rand_macro_pos, 'micro_recall': s_rand_micro_pos})
            if config["data_type"] == "all":
                s_rand_acc_neg, s_rand_macro_neg, s_rand_micro_neg = compute_structured_random_baseline(
                    tst_y_neg, config["n_classes"], T=T, random_seed=run_seed, rtype=config["rtype"], n_info=config["n_info"]
                )
                all_results_rows.append({**run_row_base, 'baseline': 'structured_random', 'split': 'neg', 'accuracy': s_rand_acc_neg, 'macro_recall': s_rand_macro_neg, 'micro_recall': s_rand_micro_neg})
            
            # --- Frequency Chain Baseline ---
            fc_acc_pos, fc_macro_pos, fc_micro_pos = compute_frequency_chain_baseline(
                tst_y_pos, train_labels_full_pool, T=T, random_seed=run_seed, rtype=config["rtype"], n_info=config["n_info"]
            )
            all_results_rows.append({**run_row_base, 'baseline': 'freq_chain', 'split': 'pos', 'accuracy': fc_acc_pos, 'macro_recall': fc_macro_pos, 'micro_recall': fc_micro_pos})
            if config["data_type"] == "all":
                fc_acc_neg, fc_macro_neg, fc_micro_neg = compute_frequency_chain_baseline(
                    tst_y_neg, train_labels_full_pool, T=T, random_seed=run_seed, rtype=config["rtype"], n_info=config["n_info"], sample_neg=True
                )
                all_results_rows.append({**run_row_base, 'baseline': 'freq_chain', 'split': 'neg', 'accuracy': fc_acc_neg, 'macro_recall': fc_macro_neg, 'micro_recall': fc_micro_neg})

    # ############################ SAVE RESULTS ##############################
    os.makedirs('./best_results/', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_type = config.get("model_type", "baseline")
    
    baseline_df = pd.DataFrame(all_results_rows)
    # Reorder columns to have run_idx first
    cols = ['run_idx'] + [col for col in baseline_df.columns if col != 'run_idx']
    baseline_df = baseline_df[cols]

    baseline_results_path = f'./best_results/baselines_{model_type}_{config["rtype"]}_{config["data_type"]}_{timestamp}.csv'
    baseline_df.to_csv(baseline_results_path, index=False)
    logger.info(f"Saved all baseline results to {baseline_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute baseline model performances over multiple runs.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to base config YAML file.")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of times to run baseline computations.")
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

    # Flatten the config to handle wandb-style {value: ...} entries
    config = flatten_wandb_config(config)

    # Override config with any command-line arguments
    for i in range(0, len(unknown), 2):
        key = unknown[i].replace("--", "")
        val = unknown[i+1]
        if key in config:
            try:
                config[key] = type(config[key])(val) if config[key] is not None else val
            except (ValueError, TypeError):
                config[key] = val
    
    compute_baselines(config, args.n_runs)
    logger.info("Script execution finished.")