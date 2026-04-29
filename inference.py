#!/usr/bin/env python3
"""
Inference script for reaction condition prediction models.

This script loads a trained model and generates positive and negative reaction conditions
for a given set of SMILES strings (starting materials and product).

Usage:
    python inference.py --config inference_config.yaml
"""

import argparse
import yaml
import os
import sys
import numpy as np
import torch
import logging
from tqdm import tqdm
import pickle as pkl
from collections import Counter
import pandas as pd
import pulp
import itertools
import json
import time


# Ensure the active Python environment's bin directory (where xtb lives) is
# on PATH. Derives the path from sys.executable so it works for any conda
# env / virtualenv the script is launched with.
_env_bin = os.path.dirname(sys.executable)
if _env_bin and _env_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_env_bin}{os.pathsep}{os.environ.get('PATH', '')}"

# Fix Numba-NumPy compatibility issue
os.environ['NUMBA_NUMNPY_COMPAT'] = '1'

# Import necessary modules from the codebase
from utils.create_graphs import mol_to_graph
from utils.collate_functions import collate_reaction_graphs, collate_graphs_and_embeddings
from utils.dataset import GraphDataset, get_cardinalities_classes
from utils.miscellaneous import create_pos_neg_count_matrices

# RDKit imports
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# XGBoost scoring model
from models.model_xgboost_yield import XGBoostYieldRegressor

# Optional embedding models for feature construction
from transformers import AutoTokenizer, AutoModel

def setup_logging(verbose=False):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def configure_external_loggers(config):
    """Suppress noisy third-party DEBUG logs unless explicitly enabled in config."""
    show_debug = bool(config.get('show_debug_logs', False))
    level = logging.DEBUG if show_debug else logging.WARNING
    for name in [
        'urllib3',
        'urllib3.connectionpool',
        'transformers',
        'huggingface_hub',
        'numba',
        'numba.cuda',
        'numba.core',
        'rdkit',
        'pulp',
        'Auto3D',
        'aimnet',
        'xgboost',
    ]:
        logging.getLogger(name).setLevel(level)
        logging.getLogger(name).propagate = False

def _banner(msg: str) -> str:
    line = '=' * 72
    return f"\n{line}\n{msg}\n{line}"

def _resolve_model_and_xgb_paths(config, logger):
    """Resolve model/xgb paths from rtype, supporting both legacy single-path and new per-type keys.

    Expected optional keys in config:
      - model_path (legacy)
      - model_path_bh, model_path_sm (new)
      - xgb_model_path, xgb_config_path (legacy)
      - xgb_model_path_bh, xgb_model_path_sm, xgb_config_path_bh, xgb_config_path_sm (new)
    """
    rtype = str(config.get('rtype', '')).lower()
    if rtype not in {'bh', 'sm'}:
        raise ValueError("Config 'rtype' must be 'bh' or 'sm'.")

    # Resolve generative model path
    if not config.get('model_path'):
        candidate_key = 'model_path_bh' if rtype == 'bh' else 'model_path_sm'
        if config.get(candidate_key):
            config['model_path'] = config[candidate_key]
            logger.info(f"Using model_path from '{candidate_key}': {config['model_path']}")
        else:
            # Keep as-is; downstream loader will error if missing
            logger.warning("No 'model_path' provided and no per-type model_path key found."
                           " Set 'model_path' or 'model_path_bh'/'model_path_sm'.")

    # Resolve XGBoost paths for pruning
    xgb_model_key = 'xgb_model_path'
    xgb_config_key = 'xgb_config_path'

    if not config.get(xgb_model_key):
        candidate_key = 'xgb_model_path_bh' if rtype == 'bh' else 'xgb_model_path_sm'
        if config.get(candidate_key):
            config[xgb_model_key] = config[candidate_key]
            logger.info(f"Using {xgb_model_key} from '{candidate_key}': {config[xgb_model_key]}")

    if not config.get(xgb_config_key):
        candidate_key = 'xgb_config_path_bh' if rtype == 'bh' else 'xgb_config_path_sm'
        if config.get(candidate_key):
            config[xgb_config_key] = config[candidate_key]
            logger.info(f"Using {xgb_config_key} from '{candidate_key}': {config[xgb_config_key]}")

    return config

def _initialize_chemberta(config):
    """Initialize the ChemBERTa model and tokenizer for molecular embeddings."""
    chemberta_model_name = "seyonec/ChemBERTa-zinc-base-v1"
    print(f"Initializing ChemBERTa model ({chemberta_model_name})...")
    try:
        from utils.bootstrap import ensure_chemberta_safetensors
        ensure_chemberta_safetensors(chemberta_model_name)
        tokenizer = AutoTokenizer.from_pretrained(chemberta_model_name)
        model = AutoModel.from_pretrained(chemberta_model_name)
        embedding_dim = model.config.hidden_size

        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        device_val = config.get("device")
        use_cuda = torch.cuda.is_available() and (str(device_val).startswith('cuda'))
        device = torch.device("cuda" if use_cuda else "cpu")
        model.to(device)
        config['device'] = device
        print(f"ChemBERTa model moved to {device}.")
    except Exception as e:
        print(f"Failed to load ChemBERTa model: {e}")
        return None, None, None

    return tokenizer, model, embedding_dim

def _get_chemberta_embeddings(tokenizer, model, smiles_list, config):
    """Generate ChemBERTa embeddings for a list of SMILES strings."""
    if not model or not tokenizer:
        return torch.zeros(768, device=config['device'])

    valid_smiles = [s for s in smiles_list if s]
    if not valid_smiles:
        return torch.zeros(model.config.hidden_size, device=config['device'])

    inputs = tokenizer(valid_smiles, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(config['device']) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.flatten().unsqueeze(0).cpu()

def _get_morgan_embeddings(smiles_list, config):
    """Generate Morgan fingerprints for a list of SMILES strings."""
    from rdkit.Chem import AllChem
    morgan_embeddings = []
    fp_size = config.get('morgan_fp_size', 512)

    for smiles in smiles_list:
        mol = AllChem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size)
            arr = np.array(fp)
            morgan_embeddings.append(arr)
        else:
            morgan_embeddings.append(np.zeros(fp_size, dtype=int))

    return torch.tensor(np.concatenate(morgan_embeddings), dtype=torch.float32).unsqueeze(0)

def load_condition_list(data_type):
    """
    Load the condition list (clist) from the processed data file.
    
    Args:
        data_type (str): Either 'bh' or 'sm'
        
    Returns:
        list: List of condition names
    """
    # Try to load from the processed data file
    filepath = f"data/{data_type}_treshold_all_all_processed.npz"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    try:
        loaded_npz = np.load(filepath, allow_pickle=True)
        if 'data' not in loaded_npz:
            raise ValueError(f"'data' key not found in {filepath}")
        
        reaction_data_unpacked = loaded_npz['data']
        if isinstance(reaction_data_unpacked, np.ndarray) and len(reaction_data_unpacked) == 2:
            reaction_dict, clist = reaction_data_unpacked
            return clist
        else:
            raise ValueError(f"Expected 2-element array in 'data', got {type(reaction_data_unpacked)}")
    except Exception as e:
        raise RuntimeError(f"Error loading condition list: {e}")

def create_reaction_graphs(starting_material_1, starting_material_2, product, config):
    """
    Create graph representations for the reaction components.
    
    Args:
        starting_material_1 (str): SMILES of first starting material
        starting_material_2 (str): SMILES of second starting material
        product (str): SMILES of product
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (rmol_graphs, pmol_graphs, embeddings_mol)
    """
    logger = logging.getLogger(__name__)
    
    # Process reactants
    rmol_graphs = []
    embeddings_mol = []
    
    for i, smi in enumerate([starting_material_1, starting_material_2]):
        if smi == '':
            raise ValueError(f"Reactant SMILES is empty for reactant {i+1}")
        
        # Create RDKit mol object
        rmol = Chem.MolFromSmiles(smi, sanitize=False)
        if rmol is None:
            raise ValueError(f"Couldn't create RDKit mol for SMILES: {smi}")
        
        try:
            Chem.SanitizeMol(rmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=False)
        except Exception as e:
            logger.warning(f"Standard sanitization failed for SMILES: {smi}. Error: {e}")
            try:
                rmol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(rmol, 
                                Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                catchErrors=False)
            except Exception as e:
                raise ValueError(f"Failed partial sanitization with error {e}")
        
        try:
            Chem.AssignStereochemistry(rmol, flagPossibleStereoCenters=True, force=True)
            Chem.AssignAtomChiralTagsFromStructure(rmol, -1)
        except Exception as e:
            raise ValueError(f"Stereochemistry assignment failed for SMILES: {smi}. Error: {e}")
        
        # Handle stereochemistry
        rs = Chem.FindPotentialStereo(rmol)
        for element in rs:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                rmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                rmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
        
        # Add/remove hydrogens
        if config["addh"]:
            rmol = Chem.AddHs(rmol)
        else:
            rmol = Chem.RemoveHs(rmol)
        
        # Create graph and embeddings
        graph_obj, emb_info = mol_to_graph(rmol, config)
        rmol_graphs.append(graph_obj)
        embeddings_mol.append(emb_info)
    
    # Process product
    if product == '':
        raise ValueError("Product SMILES is empty")
    
    pmol = Chem.MolFromSmiles(product, sanitize=False)
    if pmol is None:
        raise ValueError(f"Couldn't create RDKit mol for product SMILES: {product}")
    
    try:
        Chem.SanitizeMol(pmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=False)
    except Exception as e:
        logger.warning(f"Standard sanitization failed for product SMILES: {product}. Error: {e}")
        try:
            pmol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(pmol, 
                            Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                            Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                            Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                            catchErrors=False)
        except Exception as e:
            raise ValueError(f"Failed partial sanitization with error {e}")
    
    try:
        Chem.AssignStereochemistry(pmol, flagPossibleStereoCenters=True, force=True)
        Chem.AssignAtomChiralTagsFromStructure(pmol, -1)
    except Exception as e:
        raise ValueError(f"Stereochemistry assignment failed for product SMILES: {product}. Error: {e}")
    
    # Handle stereochemistry
    ps = Chem.FindPotentialStereo(pmol)
    for element in ps:
        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
            pmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
            pmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
    
    # Add/remove hydrogens
    if config["addh"]:
        pmol = Chem.AddHs(pmol)
    else:
        pmol = Chem.RemoveHs(pmol)
    
    # Create graph and embeddings
    graph_obj, emb_info = mol_to_graph(pmol, config)
    pmol_graphs = [graph_obj]
    embeddings_mol.append(emb_info)
    
    return rmol_graphs, pmol_graphs, embeddings_mol

def finalize_reaction_graphs(rmol_graphs, pmol_graphs, embeddings_mol_dict, config):
    """
    Finalize the reaction graphs.
    """
    if config["model_type"] in ["emb", "seq_emb"]:
        
            #Normalize the values of sasa
        if "sasa" in config["atom_emb_list"]:
            print("Normalizing sasa values")
            atom_sasa_mean = torch.mean(torch.cat([embeddings_mol_dict[i][j]["atom_emb"]["sasa"] for i in range(len(embeddings_mol_dict)) for j in range(3)]))
            atom_sasa_std = torch.std(torch.cat([embeddings_mol_dict[i][j]["atom_emb"]["sasa"] for i in range(len(embeddings_mol_dict)) for j in range(3)]))
            mol_sasa_mean = torch.mean(torch.cat([embeddings_mol_dict[i][j]["mol_emb"]["sasa"] for i in range(len(embeddings_mol_dict)) for j in range(3)]))
            mol_sasa_std = torch.std(torch.cat([embeddings_mol_dict[i][j]["mol_emb"]["sasa"] for i in range(len(embeddings_mol_dict)) for j in range(3)]))
            for i in range(len(embeddings_mol_dict)):
                for j in range(3):
                    embeddings_mol_dict[i][j]["atom_emb"]["sasa"] = (embeddings_mol_dict[i][j]["atom_emb"]["sasa"] - atom_sasa_mean) / (atom_sasa_std + 1e-6)
                    embeddings_mol_dict[i][j]["mol_emb"]["sasa"] = (embeddings_mol_dict[i][j]["mol_emb"]["sasa"] - mol_sasa_mean) / (mol_sasa_std + 1e-6)

        embeddings_mol = [[None, None, None]]

        
        if len(config["atom_emb_list"]) > 0:

            for i in range(len(rmol_graphs)):
                for j in range(len(rmol_graphs[i])):
                    if config["atom_emb_list"] == ["xtb"]:
                        rmol_graphs[i][j].ndata['node_attr'] = torch.cat([rmol_graphs[i][j].ndata['node_attr'].float(), embeddings_mol_dict[i][j]["atom_emb"]["xtb"].float()], dim=1)
                    elif config["atom_emb_list"] == ["sasa"]:
                        rmol_graphs[i][j].ndata['node_attr'] = torch.cat([rmol_graphs[i][j].ndata['node_attr'].float(), embeddings_mol_dict[i][j]["atom_emb"]["sasa"].float()], dim=1)
                    elif config["atom_emb_list"] == ["xtb", "sasa"]:
                        rmol_graphs[i][j].ndata['node_attr'] = torch.cat([rmol_graphs[i][j].ndata['node_attr'].float(), embeddings_mol_dict[i][j]["atom_emb"]["xtb"].float(), embeddings_mol_dict[i][j]["atom_emb"]["sasa"].float()], dim=1)

            for i in range(len(pmol_graphs)):
                for j in range(len(pmol_graphs[i])):
                    if config["atom_emb_list"] == ["xtb"]:
                        pmol_graphs[i][j].ndata['node_attr'] = torch.cat([pmol_graphs[i][j].ndata['node_attr'].float(), embeddings_mol_dict[i][2]["atom_emb"]["xtb"].float()], dim=1)
                    elif config["atom_emb_list"] == ["sasa"]:
                        pmol_graphs[i][j].ndata['node_attr'] = torch.cat([pmol_graphs[i][j].ndata['node_attr'].float(), embeddings_mol_dict[i][2]["atom_emb"]["sasa"].float()], dim=1)
                    elif config["atom_emb_list"] == ["xtb", "sasa"]:
                        pmol_graphs[i][j].ndata['node_attr'] = torch.cat([pmol_graphs[i][j].ndata['node_attr'].float(), embeddings_mol_dict[i][2]["atom_emb"]["xtb"].float(), embeddings_mol_dict[i][2]["atom_emb"]["sasa"].float()], dim=1)

            for i in range(len(embeddings_mol_dict)):
                for j in range(3):
                    if embeddings_mol_dict[i][j] is not None:
                        if config["atom_emb_list"] == ["xtb"]:
                            embeddings_mol[i][j] = embeddings_mol_dict[i][j]["mol_emb"]["xtb"]
                        elif config["atom_emb_list"] == ["sasa"]:
                            embeddings_mol[i][j] = embeddings_mol_dict[i][j]["mol_emb"]["sasa"]
                        elif config["atom_emb_list"] == ["xtb", "sasa"] or config["atom_emb_list"] == ["sasa", "xtb"]:
                            embeddings_mol[i][j] = torch.cat([embeddings_mol_dict[i][j]["mol_emb"]["xtb"], embeddings_mol_dict[i][j]["mol_emb"]["sasa"]], dim=1)

        rsmi = [f"{config['starting_material_1']}.{config['starting_material_2']}>>{config['product']}"]
        return rmol_graphs, pmol_graphs, embeddings_mol, rsmi
    else:
        return rmol_graphs, pmol_graphs, embeddings_mol_dict, rsmi


def setup_params_model_and_trainer(config, device, clist):
    """
    Load the appropriate model and trainer based on model type.
    
    Args:
        config (dict): Configuration dictionary
        device (torch.device): Device to load model on
        clist (list): Condition list
        
    """
    # Set up model path if not provided
    if config["model_path"] is None:
        config["model_path"] = f'./best_models/model_{config["model_type"]}_{config["rtype"]}_all_0.pt'
    
    # Set up configuration for model loading
    config["clist"] = clist
    config["n_classes"] = len(clist)
    config["rmol_max_cnt"] = 2
    config["pmol_max_cnt"] = 1
    config["wandb"] = False
    
    # Import appropriate model based on type
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
        raise ValueError(f"Unrecognized model_type '{config['model_type']}'. Valid options are: 'baseline', 'rxnfp', 'seq', 'emb', 'seq_emb'")
    
    # Get cardinalities for the model
    n_info = get_cardinalities_classes(config)
    config["n_info"] = n_info

    return Model, Trainer

def load_model_and_trainer(config, device, dataset, Model, Trainer):
    """
    Load the model and trainer.
    """
    # Initialize model
    if config["model_type"] == 'rxnfp':
        net = Model(dataset.fp_dim * 3 + 1, config["n_classes"])
    elif config["model_type"] in ['baseline', 'baselineofbaseline']:
        net = Model(dataset.node_dim, dataset.edge_dim, config["n_classes"])
    elif config["model_type"] == 'seq':
        net = Model(config["rtype"], dataset.node_dim, dataset.edge_dim, config["n_classes"], config["n_info"])
    elif config["model_type"] == 'emb':
        net = Model(dataset.node_dim, dataset.edge_dim, config["n_classes"], dataset.emb_dim)
    elif config["model_type"] == 'seq_emb':
        net = Model(config["rtype"], dataset.node_dim, dataset.edge_dim, config["n_classes"], config["n_info"], dataset.emb_dim)
    
    # Create trainer and load model
    trainer = Trainer(net, device, config)
    success = trainer.load()
    
    if not success:
        raise RuntimeError(f"Failed to load model from {config['model_path']}")
    
    return net, trainer


def map_predictions_to_conditions(predictions, config):
    """
    Map prediction indices to actual condition names using the reagents_df.
    """
    # Get condition type ranges as before
    n_info = config["n_info"]
    n_cats, n_sol_1, n_sol_2, n_add, n_base = n_info

    reagents_df = pd.read_csv(f"./reagents_dfs/{config['rtype']}_treshold_all_all_reagent_df.csv")

    if config["rtype"] == 'bh':
        ranges = {
            'catalyst': (0, n_cats),
            'base': (n_cats, n_cats + n_base),
            'solvent_1': (n_cats + n_base, n_cats + n_base + n_sol_1),
            'water': (n_cats + n_base + n_sol_1, n_cats + n_base + n_sol_1 + 1),
            'additives': (n_cats + n_base + n_sol_1 + 1, n_cats + n_base + n_sol_1 + 1 + n_add)
        }
    elif config["rtype"] == 'sm':
        ranges = {
            'solvent_1': (0, n_sol_1),
            'solvent_2': (n_sol_1, n_sol_1 + n_sol_2),
            'water': (n_sol_1 + n_sol_2, n_sol_1 + n_sol_2 + 1),
            'additives': (n_sol_1 + n_sol_2 + 1, n_sol_1 + n_sol_2 + 1 + n_add),
            'catalyst': (n_sol_1 + n_sol_2 + 1 + n_add, n_sol_1 + n_sol_2 + 1 + n_add + n_cats),
            'base': (n_sol_1 + n_sol_2 + 1 + n_add + n_cats, n_sol_1 + n_sol_2 + 1 + n_add + n_cats + n_base)
        }

    # Map predictions to conditions using the DataFrame
    conditions = {}
    for cond_type, (start_idx, end_idx) in ranges.items():
        selected_indices = [idx for idx in predictions if start_idx <= idx < end_idx]
        # Use .iloc to get the reagent name by index
        conditions[cond_type] = [reagents_df.iloc[idx]["reagent"] for idx in selected_indices]
    return conditions

def print_conditions_summary(positive_conditions, negative_conditions, sample_size):
    """
    Print a summary of the generated conditions.
    
    Args:
        positive_conditions (list): List of positive condition predictions
        negative_conditions (list): List of negative condition predictions
        sample_size (int): Number of samples generated
    """
    print(f"\n{'='*80}")
    print(f"CONDITION PREDICTION SUMMARY")
    print(f"{'='*80}")
    print(f"Generated {sample_size} positive and {sample_size} negative condition sets")
    
    # Analyze positive conditions
    print(f"\n{'='*40}")
    print(f"POSITIVE CONDITIONS (Top 10 most frequent)")
    print(f"{'='*40}")
    
    # Count frequency of each condition type in positive samples
    pos_condition_counts = {}
    for sample in positive_conditions:
        for cond_type, conditions in sample.items():
            if cond_type not in pos_condition_counts:
                pos_condition_counts[cond_type] = Counter()
            for condition in conditions:
                pos_condition_counts[cond_type][condition] += 1
    
    for cond_type, counter in pos_condition_counts.items():
        print(f"\n{cond_type.upper()}:")
        for condition, count in counter.most_common(10):
            percentage = (count / sample_size) * 100
            print(f"  {condition}: {count} times ({percentage:.1f}%)")
    
    # Analyze negative conditions
    print(f"\n{'='*40}")
    print(f"NEGATIVE CONDITIONS (Top 10 most frequent)")
    print(f"{'='*40}")
    
    # Count frequency of each condition type in negative samples
    neg_condition_counts = {}
    for sample in negative_conditions:
        for cond_type, conditions in sample.items():
            if cond_type not in neg_condition_counts:
                neg_condition_counts[cond_type] = Counter()
            for condition in conditions:
                neg_condition_counts[cond_type][condition] += 1
    
    for cond_type, counter in neg_condition_counts.items():
        print(f"\n{cond_type.upper()}:")
        for condition, count in counter.most_common(10):
            percentage = (count / sample_size) * 100
            print(f"  {condition}: {count} times ({percentage:.1f}%)")
    
    print(f"\n{'='*80}")

def construct_plate_bh(positive_conditions, negative_conditions, config):
    """
    Construct a plate design from the positive and negative conditions.
    """
    ranking_cats = ["XantPhos", "tBuBrettPhos", "XPhos", "AlPhos", "QPhos", "BrettPhos", "BINAP", "cBridP", "BippyPhos", "IPENT Cl", "RuPhos", "P(tBu)3", "DiMeIHept Cl", "GPhos", "JosiPhosSL J009-1", "SPhos", "dppf", "Amphos", "dtbpf", "CPhos", "IHept", "DPEPhos", "CataCXiumA", "Triisobutyl-Phosphatrane", "CyJohnPhos", "RockPhos", "meCgPPh", "dppp", "P(tBu)(Cy)2", "Piperidinyl-amino-pincer", "JohnPhos", "EPhos", "P(oTol)3", "dppdtbpf", "SIPr", "VPhos", "dcypf", "P(Cy3)", "tBuPhosphinito", "CyMPhos"]
    reagents_df = pd.read_csv(f"./reagents_dfs/{config['rtype']}_treshold_all_all_reagent_df.csv")

    #get indices of ranking cats in reagent_df if not present in reagent_df just put nan
    ranking_cats_indices = [reagents_df[reagents_df["reagent"] == cat].index[0] if cat in reagents_df["reagent"].values else None for cat in ranking_cats]
    ranking_cats_indices = [index for index in ranking_cats_indices if index is not None]
    

    ranking_cats = ["XantPhos", "tBuBrettPhos", "XPhos", "AlPhos", "QPhos", "BrettPhos", "BINAP", "cBridP", "BippyPhos", "IPENT Cl", "RuPhos", "P(tBu)3", "DiMeIHept Cl", "GPhos", "JosiPhosSL J009-1", "SPhos", "dppf", "Amphos", "dtbpf", "CPhos", "IHept", "DPEPhos", "CataCXiumA", "Triisobutyl-Phosphatrane", "CyJohnPhos", "RockPhos", "meCgPPh", "dppp", "P(tBu)(Cy)2", "Piperidinyl-amino-pincer", "JohnPhos", "EPhos", "P(oTol)3", "dppdtbpf", "SIPr", "VPhos", "dcypf", "P(Cy3)", "tBuPhosphinito", "CyMPhos"]
    reagents_df = pd.read_csv(f"./reagents_dfs/{config['rtype']}_treshold_all_all_reagent_df.csv")

    #get indices of ranking cats in reagent_df if not present in reagent_df just put nan
    ranking_cats_indices = [reagents_df[reagents_df["reagent"] == cat].index[0] if cat in reagents_df["reagent"].values else None for cat in ranking_cats]
    ranking_cats_indices = [index for index in ranking_cats_indices if index is not None]

    #copy or ranking cat indices
    ranking_cats_indices_copy = ranking_cats_indices.copy()

    #count which unique cats appear in positive conditions iterating over all samoles
    cat_used_in_positive_conditions = []
    for sample in positive_conditions:
        for cat in ranking_cats_indices_copy:
            if cat in sample:
                cat_used_in_positive_conditions.append(cat)
                ranking_cats_indices_copy.remove(cat)

    number_of_unique_cats_in_positive_conditions = len(cat_used_in_positive_conditions)

    if number_of_unique_cats_in_positive_conditions > 12:
        enough_cats = True
    else:
        enough_cats = False
        #add enough cats to make it 12 by adding the most used cats
        for cat in ranking_cats_indices_copy:
            cat_used_in_positive_conditions.append(cat)
            if len(cat_used_in_positive_conditions) == 12:
                break


    #get number of solvents and number of bases in positive conditions
    ranking_solvents = ["PhMe", "tAmOH", "MeCN", "MeTHF", "Dioxane", "DMAc", "anisole", "THF", "Ph2O", "DMI", "CPME", "TBME", "iPrOAc", "iPrOH", "Propionitrile", "nBuOAc", "MeOH", "xylenes", "acetone", "o-Xylene", "DMSO", "PhCN", "DMF", "cyclohexane", "NMP", "DMPU", "chlorobenzene", "TPGS-750M", "EtOH", "TFE"]

    ranking_bases = ["Cs2CO3", "NaOtBu", "K3PO4", "KHMDS", "K2CO3", "LiHMDS", "KOH", "Na2CO3", "KOtBu", "DBU", "KOtPent", "TMSOK", "KOAc", "NaHCO3", "CsOH.H2O", "Potassium 2-ethylhexanoate", "NaOtPent", "NaH", "CsF", "lutidine", "KF", "NaBHT", "Li2CO3", "LiOtBu", "DIPEA", "TMG", "NaOH", "K2HPO4", "LiOH anh", "Sodium phenoxide", "NMI", "Zinc acetate"]

    unique_solvents = []
    unique_bases = []

    for sample in positive_conditions:
        for condition in sample:
            line_of_df = reagents_df.iloc[condition]
            if line_of_df["reagent_type"] == "S" and condition not in unique_solvents:
                unique_solvents.append(condition)
            elif line_of_df["reagent_type"] == "B" and condition not in unique_bases:
                unique_bases.append(condition)


    if len(unique_solvents) * len(unique_bases) > 8:
        enough_solvents_and_bases = True
    else:
        enough_solvents_and_bases = False
        #add enough solvents and bases to make it 8 by adding the most used solvents and bases adding one by one
        for solvent in ranking_solvents:
            if len(unique_solvents) < 8:
                unique_solvents.append(solvent)
        for base in ranking_bases:
            if len(unique_bases) < 8:
                unique_bases.append(base)
        

    sb_couples_that_should_never_mix = [("NaHMDS", "tAmOH"), ("KOtBu","MeCN"), ("NaH","DMSO")]
    sb_couples_that_shouldnt_mix_without_water = [("Na2CO3", "PhMe"), ("K3PO4", "Dioxane")]

    #Check if any of these forbidden couples elements is not in the readeng_df, if so remove it from the list
    for couple in sb_couples_that_should_never_mix:
        if couple[0] not in reagents_df["reagent"].values or couple[1] not in reagents_df["reagent"].values:
            sb_couples_that_should_never_mix.remove(couple)
    for couple in sb_couples_that_shouldnt_mix_without_water:
        if couple[0] not in reagents_df["reagent"].values or couple[1] not in reagents_df["reagent"].values:
            sb_couples_that_shouldnt_mix_without_water.remove(couple)

    #find indices for these tuples from the reagents_df
    sb_couples_that_should_never_mix_indices = [(reagents_df[reagents_df["reagent"] == couple[1]].index[0], reagents_df[reagents_df["reagent"] == couple[0]].index[0]) for couple in sb_couples_that_should_never_mix]
    sb_couples_that_shouldnt_mix_without_water_indices = [(reagents_df[reagents_df["reagent"] == couple[1]].index[0], reagents_df[reagents_df["reagent"] == couple[0]].index[0]) for couple in sb_couples_that_shouldnt_mix_without_water]

    forbidded_sb_couples_final = sb_couples_that_should_never_mix_indices + sb_couples_that_shouldnt_mix_without_water_indices  

    ######################## ILP problem ########################

    #Aliases for the ILP problem
    class_list = list(reagents_df["reagent_type"])
    desirable_items = positive_conditions
    undesirable_items = negative_conditions

    # Map each component index to its class ('C', 'B', 'S', etc.)
    class_map = {i: cls for i, cls in enumerate(class_list)}

    # Get lists of indices for each relevant class
    catalyst_indices = [i for i, cls in class_map.items() if cls == 'C']
    base_indices = [i for i, cls in class_map.items() if cls == 'B']
    solvent_indices = [i for i, cls in class_map.items() if cls == 'S']

    # Generate all possible solvent-base pairs
    sb_pairs = list(itertools.product(solvent_indices, base_indices))

    # Define the ILP problem
    # We want to maximize the objective function
    prob = pulp.LpProblem("Maximize_Item_Coverage", pulp.LpMaximize)

    # --- 3. Define Decision Variables ---

    # Variables for selecting catalysts
    # x_vars[i] = 1 if catalyst i is chosen, 0 otherwise
    x_vars = pulp.LpVariable.dicts("Catalyst", catalyst_indices, cat='Binary')

    # Variables for selecting solvent-base pairs
    # p_vars[(s,b)] = 1 if pair (s,b) is chosen, 0 otherwise
    p_vars = pulp.LpVariable.dicts("SB_Pair", sb_pairs, cat='Binary')

    # Variables for tracking if an item is built
    # y_vars[i] = 1 if desirable_item i is built
    # z_vars[j] = 1 if undesirable_item j is built
    y_vars = pulp.LpVariable.dicts("Desirable", range(len(desirable_items)), cat='Binary')
    z_vars = pulp.LpVariable.dicts("Undesirable", range(len(undesirable_items)), cat='Binary')

    # --- 4. Define the Objective Function ---

    # Objective: Maximize (desirable items) - (undesirable items)
    prob += pulp.lpSum(y_vars) - pulp.lpSum(z_vars), "Total_Score"

    # --- 5. Define Constraints ---

    # Budget Constraints
    prob += pulp.lpSum(x_vars) == 12, "Catalyst_Budget"
    prob += pulp.lpSum(p_vars) == 8, "SB_Pair_Budget"

    # Item Building Logic Constraints
    def add_item_constraints(item_list, prob, item_vars, prefix):
        for i, item in enumerate(item_list):
            # Find required components for this item
            required_catalysts = [c for c in item if class_map.get(c) == 'C']
            required_solvents = [s for s in item if class_map.get(s) == 'S']
            required_bases = [b for b in item if class_map.get(b) == 'B']

            # An item is built only if all its required components are selected
            # Add a constraint for each required catalyst
            for cat in required_catalysts:
                prob += item_vars[i] <= x_vars[cat], f"{prefix}_{i}_req_cat_{cat}"

            # Add a constraint for the required solvent-base pair
            # This assumes an item has at most one solvent and one base
            if required_solvents and required_bases:
                s = required_solvents[0]
                b = required_bases[0]
                if (s,b) in p_vars and (s,b) not in forbidded_sb_couples_final: # Check if the pair is valid
                    prob += item_vars[i] <= p_vars[(s, b)], f"{prefix}_{i}_req_sb_{s}_{b}"
                else: # This item can never be made
                    prob += item_vars[i] == 0, f"{prefix}_{i}_impossible_sb"


    # Add constraints for both desirable and undesirable items
    add_item_constraints(desirable_items, prob, y_vars, "Desirable")
    add_item_constraints(undesirable_items, prob, z_vars, "Undesirable")


    # --- 6. Solve the Problem ---

    print("Solving the ILP problem...")
    # The CBC solver is used by default. It's fast and reliable.
    prob.solve()
    print("Solver finished.")

    # --- 7. Display Results ---

    print("\n" + "="*40)
    print(f"Solver Status: {pulp.LpStatus[prob.status]}")

    if prob.status == pulp.LpStatusOptimal:
        # --- Selected Components ---
        selected_catalysts = [i for i in catalyst_indices if x_vars[i].varValue > 0.9]
        selected_sb_pairs = [p for p in sb_pairs if p_vars[p].varValue > 0.9]

        #check for each selected sb pair how many times water appeared in the positive conditions and in the negative conditions that contained that sb pair
        water_index = reagents_df[reagents_df["reagent_type"] == "W"].index[0]
        water_dict = {}
        for p in selected_sb_pairs:
            water_in_positive = 0
            water_in_negative = 0
            for sample in positive_conditions:
                if set(list(p) + [water_index]).issubset(set(sample)):
                    water_in_positive += 1
            for sample in negative_conditions:
                if set(list(p) + [water_index]).issubset(set(sample)):
                    water_in_negative += 1
            water_dict[p] = {"positive": water_in_positive, "negative": water_in_negative}

        selected_catalysts_names = [reagents_df.iloc[i]["reagent"] for i in selected_catalysts]
        selected_sb_pairs_names = [reagents_df.iloc[p[0]]["reagent"] + "-" + reagents_df.iloc[p[1]]["reagent"] for p in selected_sb_pairs]

        # Check if selected conditions have any additive and if yes, which one
        additive_indexes = list(reagents_df[reagents_df["reagent_type"] == "A"].index)
        
        # Take all possible combinations of the selected catalysts and the selected sb pairs
        add_info = {}
        for catalyst in selected_catalysts:
            for sb_pair in selected_sb_pairs:
                for additive in additive_indexes:
                    for condition in positive_conditions:
                        # FIX 1: Create the set of components correctly
                        # [catalyst] instead of list(catalyst)
                        # list(sb_pair) assumes sb_pair is a tuple, e.g., (sm_index, base_index)
                        component_set = set([catalyst] + list(sb_pair) + [additive])

                        if component_set.issubset(set(condition)):
                            # FIX 2: Index the sb_pair tuple to get each reagent name
                            catalyst_reagent = reagents_df.iloc[catalyst]["reagent"]
                            sm_reagent = reagents_df.iloc[sb_pair[0]]["reagent"]
                            base_reagent = reagents_df.iloc[sb_pair[1]]["reagent"]
                            additive_reagent = reagents_df.iloc[additive]["reagent"]
                            print(
                                f"Selected condition has additive: "
                                f"{catalyst_reagent} + "
                                f"{sm_reagent} + {base_reagent} + "
                                f"{additive_reagent}"
                            )
                            add_info[(catalyst, sb_pair)] = additive_reagent

        print("\n--- Optimal Selection ---")
        print(f"✅ Selected Catalysts ({len(selected_catalysts)}):")
        print(selected_catalysts_names)
        print(f"\n✅ Selected Solvent-Base Pairs ({len(selected_sb_pairs)}):")
        print(selected_sb_pairs_names)
        print(f"\n💧 Water distribution in sampled conditions for selected Solvent-Base Pairs:")
        print(water_dict)

        # --- Score Calculation ---
        num_desirable_made = sum(y_vars[i].varValue for i in range(len(desirable_items)))
        num_undesirable_made = sum(z_vars[j].varValue for j in range(len(undesirable_items)))

        print("\n--- Performance Score ---")
        print(f"👍 Desirable Items Built: {int(num_desirable_made)}")
        print(f"👎 Undesirable Items Built: {int(num_undesirable_made)}")
        print(f"🎯 Final Score (Desirable - Undesirable): {pulp.value(prob.objective)}")

        return (selected_catalysts_names, selected_sb_pairs_names, water_dict, add_info)
    else:
        print("Could not find an optimal solution.")

    print("="*40)


def construct_plate_ilp(positive_conditions, negative_conditions, config, logger, negative_penalty_weight=1.0, minimum_pos_covered=None):
    """Construct a plate using ILP as done in experiment_3 (general for 'bh' or 'sm')."""
    reagents_df = pd.read_csv(f"./reagents_dfs/{config['rtype']}_treshold_all_all_reagent_df.csv")
    class_map = {i: cls for i, cls in enumerate(list(reagents_df["reagent_type"]))}

    catalyst_indices = [i for i, cls in class_map.items() if cls == 'C']
    base_indices = [i for i, cls in class_map.items() if cls == 'B']
    solvent_indices = [i for i, cls in class_map.items() if cls == 'S']

    all_sb_pairs = list(itertools.product(solvent_indices, base_indices))

    prob = pulp.LpProblem("Maximize_Coverage", pulp.LpMaximize)

    x_vars = pulp.LpVariable.dicts("Catalyst", catalyst_indices, cat='Binary')
    p_vars = pulp.LpVariable.dicts("SB_Pair", all_sb_pairs, cat='Binary')

    y_vars = pulp.LpVariable.dicts("Desirable", range(len(positive_conditions)), cat='Binary')
    z_vars = pulp.LpVariable.dicts("Undesirable", range(len(negative_conditions)), cat='Binary')

    prob += pulp.lpSum(y_vars) - negative_penalty_weight * pulp.lpSum(z_vars), "Total_Score"
    prob += pulp.lpSum(x_vars) == 12, "Catalyst_Budget"
    prob += pulp.lpSum(p_vars) == 8, "SB_Pair_Budget"

    for i, cond in enumerate(positive_conditions):
        cat = next(c for c in cond if class_map.get(c) == 'C')
        sol = next(s for s in cond if class_map.get(s) == 'S')
        bas = next(b for b in cond if class_map.get(b) == 'B')
        prob += y_vars[i] <= x_vars[cat]
        prob += y_vars[i] <= p_vars[(sol, bas)]
        prob += y_vars[i] >= x_vars[cat] + p_vars[(sol, bas)] - 1

    for i, cond in enumerate(negative_conditions):
        cat = next(c for c in cond if class_map.get(c) == 'C')
        sol = next(s for s in cond if class_map.get(s) == 'S')
        bas = next(b for b in cond if class_map.get(b) == 'B')
        prob += z_vars[i] <= x_vars[cat]
        prob += z_vars[i] <= p_vars[(sol, bas)]
        prob += z_vars[i] >= x_vars[cat] + p_vars[(sol, bas)] - 1

    if minimum_pos_covered is not None:
        prob += pulp.lpSum(y_vars) >= minimum_pos_covered, "Minimum_Positive_Coverage"
        logger.info(f"Added constraint: minimum positive coverage = {minimum_pos_covered}")

    logger.info("Solving the ILP problem...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    logger.info("Solver finished.")

    if prob.status == pulp.LpStatusOptimal:
        selected_catalysts = [i for i in catalyst_indices if x_vars[i].varValue > 0.9]
        selected_sb_pairs = [p for p in all_sb_pairs if p_vars[p].varValue > 0.9]
        selected_catalysts_names = sorted([reagents_df.iloc[i]["reagent"] for i in selected_catalysts])
        selected_sb_pairs_names = sorted([(reagents_df.iloc[p[0]]["reagent"], reagents_df.iloc[p[1]]["reagent"]) for p in selected_sb_pairs])
        return selected_catalysts_names, selected_sb_pairs_names
    else:
        logger.error(f"ILP solver failed with status: {pulp.LpStatus[prob.status]}")
        return None, None

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference with a trained reaction condition prediction model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to inference configuration file")
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config_file}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return
    
    # Setup logging
    logger = setup_logging(config.get('verbose', False))
    
    # Normalize paths per rtype (supports model_path_bh/sm and xgb_*_bh/sm)
    config = _resolve_model_and_xgb_paths(config, logger)
    configure_external_loggers(config)
    
    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load condition list
        logger.info("Loading condition list...")
        clist = load_condition_list(config["rtype"])  # rtype is 'bh' or 'sm'
        logger.info(f"Loaded {len(clist)} conditions")
        
        # Create reaction graphs + dataset
        logger.info(_banner("🧩 Creating reaction graphs and dataset"))
        t_graph_ds_start = time.time()
        rmol_graphs, pmol_graphs, embeddings_mol_dict = create_reaction_graphs(
            config["starting_material_1"],
            config["starting_material_2"], 
            config["product"],
            config
        )

        rmol_graphs, pmol_graphs, embeddings_mol, rsmi = finalize_reaction_graphs(
            [rmol_graphs], [pmol_graphs], [embeddings_mol_dict], config
        )

        logger.info("✅ Reaction graphs created successfully")
        
        # Setup params, model and trainer
        Model, Trainer = setup_params_model_and_trainer(config, device, clist)
        
        # Create dataset for inference
        logger.info("Creating dataset for inference...")
        dataset = GraphDataset(
            rmol_graphs, 
            pmol_graphs, 
            [[0]],  # Dummy labels
            rsmi,  # Reaction SMILES
            embeddings_mol, 
            config, 
            split='tst', 
            device=device
        )

        config["n_info"] = get_cardinalities_classes(config)
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=config["batch_size"], 
            shuffle=False, 
            collate_fn=config["collate_fn"]
        )
        logger.info(f"⏱️ Graphs + dataset prepared in {time.time() - t_graph_ds_start:.2f}s")

        # Load model and trainer
        logger.info("Loading model and trainer...")
        net, trainer = load_model_and_trainer(config, device, dataset, Model, Trainer)
        logger.info("Model loaded successfully")
        
        # Run inference (sampling)
        logger.info(_banner(f"🎲 Generating conditions (n={config['sample_size']})"))
        t_sampling_start = time.time()
        positive_predictions, negative_predictions = trainer.inference(
            dataloader, 
            n_sampling=config["sample_size"], 
            temperature=config["temperature"]
        )
        logger.info(f"⏱️ Condition generation completed in {time.time() - t_sampling_start:.2f}s")

        # We only have one reaction in the dataloader; take index 0
        raw_positive_conditions = positive_predictions[0]
        raw_negative_conditions = negative_predictions[0]

        logger.info(f"Sampled {len(raw_positive_conditions)} positive and {len(raw_negative_conditions)} negative condition instances.")

        # --- Optional Scoring and Pruning with XGBoost (experiment_1 logic) ---
        pruned_positive_conditions = raw_positive_conditions
        pruned_negative_conditions = raw_negative_conditions

        enable_pruning = ("xgb_model_path" in config) and ("xgb_config_path" in config)
        if enable_pruning:
            logger.info(_banner("📈 Scoring sampled conditions with XGBoost"))

            # Load XGB config and model
            with open(config["xgb_config_path"], 'r') as f:
                xgb_config = json.load(f)

            # Device/setup for XGB
            if device.type == 'cuda':
                if 'model_params' not in xgb_config:
                    xgb_config['model_params'] = {}
                if 'device' not in xgb_config['model_params']:
                    logger.info("Setting XGBoost 'device' to 'cuda' for inference.")
                    xgb_config['model_params']['device'] = 'cuda'
                if 'tree_method' not in xgb_config['model_params']:
                    logger.info("Setting XGBoost 'tree_method' to 'gpu_hist' for inference.")
                    xgb_config['model_params']['tree_method'] = 'gpu_hist'

            xgb_model = XGBoostYieldRegressor(
                quantiles=xgb_config['quantiles'],
                n_ensemble=xgb_config['n_ensemble'],
                scale_features=xgb_config['scale_features'],
                random_state=xgb_config['random_state'],
                verbose=config.get('verbose', False),
                config=xgb_config
            )
            xgb_model.load_model(config["xgb_model_path"])

            # Initialize ChemBERTa if needed
            chemberta_tokenizer, chemberta_model, chemberta_embedding_dim = None, None, 0
            if xgb_config.get('representation') == 'chemberta' or xgb_config.get('condition_representation') == 'chemberta':
                chemberta_tokenizer, chemberta_model, chemberta_embedding_dim = _initialize_chemberta(xgb_config)
                if chemberta_embedding_dim:
                    xgb_config['chemberta_embedding_dim'] = chemberta_embedding_dim

            # Build base reaction embeddings once
            reaction_smiles = rsmi[0]
            reactants, product = reaction_smiles.split('>>')
            sm_1, sm_2 = reactants.split('.')
            smiles_list = [product, sm_1, sm_2]

            if xgb_config['representation'] == 'chemberta':
                base_embs_reaction = _get_chemberta_embeddings(chemberta_tokenizer, chemberta_model, smiles_list, xgb_config)
            elif xgb_config['representation'] == 'morgan':
                base_embs_reaction = _get_morgan_embeddings(smiles_list, xgb_config)
            else:
                raise ValueError("Unsupported representation in xgb_config: expected 'chemberta' or 'morgan'")

            # Load reagents DF
            # Use 'all' data split by default for scoring artifacts
            reagents_df_path = os.path.join(xgb_config['reagents_dir'], f"{config['rtype']}_treshold_all_all_reagent_df.csv")
            reagent_df = pd.read_csv(reagents_df_path)

            # Prepare features for all conditions
            feature_vectors = []
            meta = []

            t_scoring_start = time.time()

            def append_condition_feature(condition_indices, cond_type):
                nonlocal feature_vectors, meta
                if xgb_config['condition_representation'] == 'multihot':
                    n_conditions = reagent_df.shape[0]
                    multihot_tensor = torch.zeros((1, n_conditions))
                    multihot_tensor[0, condition_indices] = 1
                    feature_vector = torch.cat([base_embs_reaction.clone(), multihot_tensor.clone()], dim=1)
                else:
                    embs_reaction = base_embs_reaction.clone()
                    embs_reagent_dict = {}
                    for reagent_idx in condition_indices:
                        reagent_info = reagent_df.loc[reagent_idx]
                        reagent_type = reagent_info['reagent_type']
                        emb_col = 'embedding' if xgb_config['condition_representation'] == 'chemberta' else 'morgan_fp'
                        emb_str = reagent_info[emb_col]
                        reagent_emb = torch.tensor([float(num) for num in str(emb_str).replace(" .. ", "").strip().rstrip(',').split(',') if num != ''], dtype=torch.float32).unsqueeze(0)
                        embs_reagent_dict[reagent_type] = reagent_emb

                    # Zero-embedding must match the CONDITION representation dimensionality
                    emb_dim_type = 'chemberta_embedding_dim' if xgb_config.get('condition_representation') == 'chemberta' else 'morgan_fp_size'
                    emb_dim = xgb_config.get(emb_dim_type)
                    zero_emb = torch.zeros((1, emb_dim))

                    conditions_list = ['C', 'B', 'S', 'W', 'A'] if config['rtype'] == 'bh' else ['S', 'L', 'W', 'A', 'C', 'B']
                    for r_type in conditions_list:
                        if r_type in embs_reagent_dict:
                            embs_reaction = torch.cat([embs_reaction, embs_reagent_dict[r_type], torch.ones(1, 1)], dim=1)
                        else:
                            embs_reaction = torch.cat([embs_reaction, zero_emb, torch.zeros(1, 1)], dim=1)
                    feature_vector = embs_reaction

                feature_vectors.append(feature_vector.numpy())
                meta.append({
                    "type": cond_type,
                    "indices": condition_indices
                })

            for cond in raw_positive_conditions:
                append_condition_feature(cond, "positive")
            for cond in raw_negative_conditions:
                append_condition_feature(cond, "negative")

            # Predict yields
            if feature_vectors:
                all_features = np.concatenate(feature_vectors, axis=0)
                prediction_results = xgb_model.predict(all_features)
                predicted_yields = prediction_results['ensemble_mean']
            else:
                predicted_yields = np.array([])

            logger.info(f"⏱️ XGBoost scoring (features+predict) completed in {time.time() - t_scoring_start:.2f}s")

            # Apply pruning
            epsilon = float(config.get('epsilon', 0.05))
            alpha = float(config.get('alpha', 0.0))
            logger.info(f"✂️ Pruning with epsilon={epsilon}, alpha={alpha}")
            
            pruned_pos = []
            pruned_neg = []
            rng = np.random.default_rng(config.get('random_seed', None))

            for i, m in enumerate(meta):
                score = predicted_yields[i]
                x = rng.uniform()
                if m["type"] == "positive":
                    if (score >= epsilon) and (x < alpha):
                        pruned_pos.append(m["indices"]) 
                else:
                    if (score <= epsilon) and (x < alpha):
                        pruned_neg.append(m["indices"]) 

            logger.info(f"🧮 Pruned positives: {len(pruned_pos)} (from {len(raw_positive_conditions)})")
            logger.info(f"🧮 Pruned negatives: {len(pruned_neg)} (from {len(raw_negative_conditions)})")

            pruned_positive_conditions = pruned_pos
            pruned_negative_conditions = pruned_neg

        # Map to names and/or save if requested (post-pruning if enabled)
        if config.get("report_results_sampling", False):
            logger.info("Mapping predictions to condition names (post-pruning if enabled)...")
            positive_conditions_named = []
            negative_conditions_named = []
            for pred_indices in pruned_positive_conditions:
                conditions = map_predictions_to_conditions(pred_indices, config)
                positive_conditions_named.append(conditions)
            for pred_indices in pruned_negative_conditions:
                conditions = map_predictions_to_conditions(pred_indices, config)
                negative_conditions_named.append(conditions)
            print_conditions_summary(positive_conditions_named, negative_conditions_named, len(pruned_positive_conditions))

            if config.get("save_results", False):
                output_file = f"inference_results_{config['rtype']}_{config['model_type']}.pkl"
                results = {
                    'positive_conditions': positive_conditions_named,
                    'negative_conditions': negative_conditions_named,
                    'config': config,
                    'clist': clist
                }
                with open(output_file, 'wb') as f:
                    pkl.dump(results, f)
                logger.info(f"Results saved to {output_file}")

        # For plate construction, use unique tuples and filter to valid C/S/B presence
        reagents_df = pd.read_csv(f"./reagents_dfs/{config['rtype']}_treshold_all_all_reagent_df.csv")
        class_map = {i: cls for i, cls in enumerate(list(reagents_df["reagent_type"]))}

        def is_valid_csb(cond):
            cats = [c for c in cond if class_map.get(c) == 'C']
            sols = [s for s in cond if class_map.get(s) == 'S']
            bas = [b for b in cond if class_map.get(b) == 'B']
            return len(cats) >= 1 and len(sols) >= 1 and len(bas) >= 1

        unique_pos = list({tuple(sorted(c)) for c in pruned_positive_conditions if is_valid_csb(c)})
        unique_neg = list({tuple(sorted(c)) for c in pruned_negative_conditions if is_valid_csb(c)})
        
    except Exception as e:
        logger.error(f"Error during inference (Sampling/Scoring): {e}")
        raise

    # try:
    #     #TODO: add xgboost pruning of predictions
    #     logger.info("Pruning predictions...")
    #     logger.info("Pruning predictions done...")
    # except Exception as e:
    #     logger.error(f"Error during inference (Pruning): {e}")
    #     raise

    try:
        logger.info(_banner("🧪 Plate construction via ILP"))
        t_plate_start = time.time()
        neg_weight = float(config.get('negative_penalty_weight', 1.0))
        ilp_catalysts, ilp_sb_pairs = construct_plate_ilp(unique_pos, unique_neg, config, logger, neg_weight)
        if ilp_catalysts and ilp_sb_pairs:
            logger.info(f"✅ Selected Catalysts ({len(ilp_catalysts)}): {ilp_catalysts}")
            logger.info(f"✅ Selected Solvent-Base Pairs ({len(ilp_sb_pairs)}): {ilp_sb_pairs}")
        else:
            logger.error("Failed to construct a plate with ILP.")
        logger.info(f"⏱️ Plate construction done in {time.time() - t_plate_start:.2f}s.")
    except Exception as e:
        logger.error(f"Error during inference (Plate construction): {e}")
        raise

if __name__ == "__main__":
    main() 