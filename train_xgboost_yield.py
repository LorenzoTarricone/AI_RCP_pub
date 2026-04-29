#!/usr/bin/env python3
"""
Training script for XGBoost yield prediction model with indexed condition embeddings.

This script:
1. Uses pre-computed condition embeddings for efficiency
2. Loads reaction data and extracts ChemBERTa embeddings for molecules only
3. Trains XGBoost models for yield prediction with uncertainty
4. Evaluates model performance and uncertainty calibration
5. Saves models and generates reports

Usage:
    python train_xgboost_yield_indexed.py --data_path data/bh_data_clean_all.csv --reaction_type bh
"""

import argparse
import os
import json
import yaml  # Add yaml import
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from models.model_xgboost_yield import XGBoostYieldRegressor, hyperparameter_search
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Optional imports
try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import pickle as pkl
from tqdm import tqdm
import joblib # Added for joblib.load/dump
import shutil


# The _get_reaction_labels function is now removed.

def get_condition_cache_path(reaction_type: str, data_type: str, cache_dir: str = "condition_embeddings_cache") -> str:
    """Get the path to the appropriate condition embeddings cache."""
    dataset_id = f"{reaction_type}_treshold_all_{data_type}"
    cache_path = os.path.join(cache_dir, f"{dataset_id}_condition_embeddings.pkl")
    
    # Fallback to unified cache if specific cache doesn't exist
    if not os.path.exists(cache_path):
        unified_cache_path = os.path.join(cache_dir, "unified_condition_embeddings.pkl")
        if os.path.exists(unified_cache_path):
            return unified_cache_path
        else:
            raise FileNotFoundError(
                f"Neither specific cache ({cache_path}) nor unified cache ({unified_cache_path}) found. "
                f"Please run precompute_condition_embeddings.py first."
            )
    
    return cache_path


def setup_wandb(config: dict, project_name: str = "xgboost_yield_prediction_indexed"):
    """Initialize Weights & Biases if available."""
    if wandb is None:
        print("WandB not available, skipping logging")
        return False
    
    try:
        wandb.init(
            project=project_name,
            config=config,
            name=f"xgb_yield_indexed_{config['reaction_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return True
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")
        return False


def numpy_json_serializer(obj):
    """ Custom JSON encoder for numpy types """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Let the default encoder raise the TypeError for other types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def create_output_directories(base_dir: str) -> dict:
    """Create output directories for models, plots, and reports."""
    directories = {
        'base': base_dir,
        'models': os.path.join(base_dir, 'models'),
        'plots': os.path.join(base_dir, 'plots'),
        'reports': os.path.join(base_dir, 'reports'),
        'cache': os.path.join(base_dir, 'cache')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def _initialize_chemberta(config):
    """Initialize the ChemBERTa model and tokenizer for molecular embeddings.
    
    This method loads the ChemBERTa model and tokenizer, moves them to the specified device,
    and freezes the model parameters. It handles error cases gracefully and provides
    appropriate warnings if initialization fails.

    Raises:
        RuntimeError: If model/tokenizer loading fails and use_embedd is True
    """
    chemberta_model_name = "seyonec/ChemBERTa-zinc-base-v1"
    print("\n")
    print(f"Initializing ChemBERTa model ({chemberta_model_name}) in Dataset...")
    # Load tokenizer and model onto CPU for safety with DataLoader workers
    try:
        from utils.bootstrap import ensure_chemberta_safetensors
        ensure_chemberta_safetensors(chemberta_model_name)
        tokenizer = AutoTokenizer.from_pretrained(chemberta_model_name, force_download=True)
        chemberta_model = AutoModel.from_pretrained(chemberta_model_name, force_download=True)
        chemberta_embedding_dim = chemberta_model.config.hidden_size

        # Freeze ChemBERTa parameters & set to eval mode
        for param in chemberta_model.parameters():
            param.requires_grad = False
        chemberta_model.eval()

        # --- Move model to the specified device ---
        if config['device'] and chemberta_model:
            try:
                chemberta_model.to(config['device'])
                if config['verbose']: print(f"ChemBERTa model moved to {config['device']}.")
            except Exception as e:
                warnings.warn(f"Failed to move ChemBERTa model to device {config['device']}. Using CPU instead. Error: {e}")
                config['device'] = torch.device("cpu") # Fallback to CPU
                chemberta_model.to(config['device']) # Ensure it's on CPU
        elif chemberta_model:
                # If no device specified, ensure it's on CPU
                config['device'] = torch.device("cpu")
                chemberta_model.to(config['device'])
                if config['verbose']: print("ChemBERTa model loaded on CPU.")
        # -----------------------------------------
        if config['verbose']: print("ChemBERTa model loaded and frozen.")

    except Exception as e:
        warnings.warn(f"Failed to load ChemBERTa model '{chemberta_model_name}'. "
                        f"Setting use_embedd=False. Error: {e}")
        print(f"Error: {e}")
        tokenizer = None
        chemberta_model = None
        chemberta_embedding_dim = None

    return tokenizer, chemberta_model, chemberta_embedding_dim


def _get_chemberta_embeddings(tokenizer, chemberta_model, chemberta_embedding_dim, smiles_list, config):
        """Generate ChemBERTa embeddings for a list of SMILES strings.
        
        Args:
            smiles_list (list): List of SMILES strings to generate embeddings for

        Returns:
            torch.Tensor: Tensor containing the generated embeddings

        Note:
            Requires transformers library and ChemBERTa model to be initialized.
        """
        # Determine the device to use (fallback to CPU if self.device is None)
        target_device = config['device'] if config['device'] else torch.device("cpu")

        if not chemberta_model or not tokenizer:
            # Should not happen if checks in __init__ and __getitem__ are done
            print("No ChemBERTa model or tokenizer found")
            return torch.zeros(chemberta_embedding_dim or 768, device=target_device) # Default size if unknown

        if not smiles_list or all(s == '' for s in smiles_list): # Handle empty or list of empty strings
            print("Found empty smile string")
            return torch.zeros(chemberta_embedding_dim, device = target_device)

        # Filter out any empty strings from the list before tokenizing
        valid_smiles = [s for s in smiles_list if s]
        if not valid_smiles:
                print("No valid SMILES strings found at all")
                return torch.zeros(chemberta_embedding_dim, device=target_device)

        try:
            # Tokenize the valid SMILES strings
            inputs = tokenizer(valid_smiles, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            
            # Get model outputs (inference on CPU or specified device)
            with torch.no_grad():
                outputs = chemberta_model(**inputs)

            # Extract CLS token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :] # Shape: [num_valid_smiles, embedding_dim]

            # Aggregate embeddings with concatenation to pass from shape [num_valid_smiles, embedding_dim] to [embedding_dim * num_valid_smiles]
            aggregated_embedding = cls_embeddings.flatten().unsqueeze(0) # Shape: [1,embedding_dim * num_valid_smiles]

            return aggregated_embedding

        except Exception as e:
            warnings.warn(f"Error during ChemBERTa embedding generation for SMILES {valid_smiles}: {e}")
            return torch.zeros(chemberta_embedding_dim) # Return zeros on error
        
def _get_morgan_embeddings(smiles_list, config=None, n_bits=512, radius=2):
    """Generate Morgan fingerprints for a list of SMILES strings, matching precompute_condition_embeddings.py."""
    morgan_embeddings = []
    fp_size = config.get('morgan_fp_size', n_bits) # Use config value, fallback to default
    fpgen = GetMorganGenerator(radius=radius, fpSize=fp_size)
        
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Could not convert SMILES to Morgan fingerprint: {smiles}")
            arr = np.zeros((fp_size,), dtype=int)
        else:
            fp = fpgen.GetFingerprint(mol)
            arr = np.array(fp)
        morgan_embeddings.append(arr)
    # Concatenate all fingerprints (as in ChemBERTa embeddings)
    morgan_embeddings = np.concatenate(morgan_embeddings)
    return torch.tensor(morgan_embeddings, dtype=torch.float32).unsqueeze(0)


def find_duplicate_reactions(reaction_dict):
    # Map: (reaction_key, tuple(sorted(condition_indices))) -> list of yields
    seen = defaultdict(list)
    for rxn_key, conds in reaction_dict.items():
        for cond in conds:
            # cond[0] is the list of reagent indices (condition signature)
            # cond[-1] is the yield
            cond_signature = tuple(sorted(cond[0]))  # sort to avoid order issues
            seen[(rxn_key, cond_signature)].append(cond[-1])

    # Find duplicates: same reaction and condition signature, more than one yield
    duplicates = {k: v for k, v in seen.items() if len(v) > 1}

    print(f"Found {len(duplicates)} duplicate reaction-condition pairs.")
    for (rxn_key, cond_signature), yields in list(duplicates.items())[:10]:  # print first 10
        print(f"Reaction: {rxn_key}")
        print(f"  Condition indices: {cond_signature}")
        print(f"  Yields: {yields}")
        print("")

    return duplicates

def aggregate_reaction_dict(reaction_dict, aggfunc='mean'):
    """
    Returns a new reaction_dict with duplicate reaction-condition pairs aggregated.
    aggfunc: 'mean' or 'median'
    """

    new_reaction_dict = defaultdict(list)
    seen = defaultdict(list)
    for rxn_key, conds in reaction_dict.items():
        for cond in conds:
            cond_signature = tuple(sorted(cond[0]))
            seen[(rxn_key, cond_signature)].append(cond)
    for (rxn_key, cond_signature), cond_list in seen.items():
        # Aggregate yields
        yields = [c[-1] for c in cond_list]
        if aggfunc == 'mean':
            agg_yield = float(np.mean(yields))
        elif aggfunc == 'median':
            agg_yield = float(np.median(yields))
        else:
            raise ValueError("aggfunc must be 'mean' or 'median'")
        # Use the first cond as template, but replace yield with aggregated value
        new_cond = list(cond_list[0])
        new_cond[-1] = agg_yield
        new_reaction_dict[rxn_key].append(tuple(new_cond))

    return dict(new_reaction_dict)

def _get_data_split(split: np.ndarray, reaction_dict: dict, reagent_df: pd.DataFrame, tokenizer, chemberta_model, config) -> Tuple[List[torch.Tensor], List[float]]:
    """Get data split from split indices."""
    embs = []
    yields = []
    chemberta_embedding_dim = config.get('chemberta_embedding_dim')

    #get the ith key from reaction_dict
    for i in tqdm(split, desc="Processing reactions"):
        smiles_list = []
        key = list(reaction_dict.keys())[i]
        #separe the reaction smile key into reactants and products
        reactants, product = key.split('>>')
        smiles_list.append(product)
        #split the two reactnatns
        starting_material_1, starting_material_2 = reactants.split('.')
        smiles_list.append(starting_material_1)
        smiles_list.append(starting_material_2)

        if config['representation'] == 'chemberta':
            base_embs_reaction = _get_chemberta_embeddings(tokenizer, chemberta_model, chemberta_embedding_dim, smiles_list, config).detach().cpu()
        elif config['representation'] == 'morgan':
            base_embs_reaction = _get_morgan_embeddings(smiles_list, config).detach().cpu()

        conditions_list = ['C', 'B', 'S', 'W', 'A']  if config['rxn_type'] == 'bh' else ['S', 'L', 'W', 'A', 'C', 'B']

        #get the embeddings for the reaction conditions
        conditions = reaction_dict[key]
        for condition in conditions:

            if config['condition_representation'] == 'multihot':
                n_conditions = reagent_df.shape[0]
                multihot_tensor = torch.zeros((1, n_conditions))
                multihot_tensor[0, condition[0]] = 1

                embs.append(torch.cat([base_embs_reaction.clone(), multihot_tensor.clone()], dim=1))
                yields.append(condition[2])
                continue
                
           
            embs_reaction = base_embs_reaction.clone()
            embs_reagent_dict = {}
            for reagents_idxs in condition[0]:
                reagent_type = reagent_df.loc[reagents_idxs, 'reagent_type']
                if config['condition_representation'] == 'chemberta':
                    reagent_emb_str = reagent_df.loc[reagents_idxs, 'embedding']
                    reagent_emb = torch.tensor([float(num) for num in reagent_emb_str.replace(" .. ", "").strip().rstrip(',').split(',')], dtype=torch.float32).unsqueeze(0) #shape [1, embedding_dim]
                elif config['condition_representation'] == 'morgan':
                    reagent_emb_str = reagent_df.loc[reagents_idxs, 'morgan_fp']
                    reagent_emb = torch.tensor([float(num) for num in reagent_emb_str.replace(" .. ", "").strip().rstrip(',').split(',')], dtype=torch.float32).unsqueeze(0) #shape [1, embedding_dim]
                embs_reagent_dict[reagent_type] = reagent_emb

            if config['representation'] == 'chemberta':
                emb_dim = config.get('chemberta_embedding_dim')
            else: # morgan
                emb_dim = config.get('morgan_fp_size')

            zero_emb = torch.zeros((1, emb_dim))

            # This is the new part: concatenate embedding and a presence flag
            for reagent_type in conditions_list:
                if reagent_type in embs_reagent_dict:
                    embs_reaction = torch.cat([embs_reaction, embs_reagent_dict[reagent_type]], dim=1) # Add embedding
                    embs_reaction = torch.cat([embs_reaction, torch.ones(1, 1)], dim=1) # Add presence flag (1)
                else:
                    embs_reaction = torch.cat([embs_reaction, zero_emb], dim=1) # Add zero embedding
                    embs_reaction = torch.cat([embs_reaction, torch.zeros(1, 1)], dim=1) # Add presence flag (0)
                
            yield_condition = condition[2]
            embs.append(embs_reaction)
            yields.append(yield_condition)
    
    return embs, yields

def train_and_evaluate_model(
    datasets: dict,
    config: dict,
    output_dirs: dict,
    use_wandb: bool = False,
    reagent_df: Optional[pd.DataFrame] = None
) -> Tuple[dict, str, str, str, Optional[str], dict, Optional[str]]:
    """Train and evaluate the XGBoost yield prediction model."""
    
    print("\n" + "="*60)
    print("TRAINING XGBOOST YIELD PREDICTION MODEL")
    print("="*60)
    plot_path = None
    hyperparameter_search_path = None
    
    # Get data and convert to numpy
    print("\n1. Preparing data...")
    X_train, y_train = datasets['train']
    X_val, y_val = datasets['val']
    X_test, y_test = datasets['test']

    if config['transform_target']:
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
        # The test target is NOT transformed here, because the model will predict
        # in the transformed space. We will compare apples to apples.
        # However, for metrics, we will need to inverse transform predictions.
        # Let's keep y_test as is and transform predictions back later.

    if config['verbose']:
        print("X_train.shape:", X_train.shape)
        print("X_val.shape:", X_val.shape)
        print("X_test.shape:", X_test.shape)
        print("y_train.shape:", y_train.shape)
        print("y_val.shape:", y_val.shape)
        print("y_test.shape:", y_test.shape)
        #print distribution of yields min/avg/median/max
        print("y_train min:", y_train.min())
        print("y_train avg:", y_train.mean())
        print("y_train median:", np.median(y_train))
        print("y_train max:", y_train.max())
        print("y_val min:", y_val.min())
        print("y_val avg:", y_val.mean())
        print("y_val median:", np.median(y_val))
        print("y_val max:", y_val.max())
        print("y_test min:", y_test.min())
        print("y_test avg:", y_test.mean())
        print("y_test median:", np.median(y_test))
        print("y_test max:", y_test.max())
        #print first row of first 10 elements of X_train
        print("First row of first 10 elements of X_train:", X_train[0:10])
        #how average number of zero features in X_train
        print("Average number of zero features in X_train:", np.mean(np.sum(X_train == 0, axis=1)))
        #how average number of zero features in X_val
        print("Average number of zero features in X_val:", np.mean(np.sum(X_val == 0, axis=1)))
        #how average number of zero features in X_test
        print("Average number of zero features in X_test:", np.mean(np.sum(X_test == 0, axis=1)))
        print("Unique rows in X_train:", np.unique(X_train, axis=0).shape[0])

    data_info = {}
    
    print(f"Dataset info:")
    if not data_info:
        print("  No dataset info available.")
    else:
        for key, value in data_info.items():
            if key != 'yield_stats':
                print(f"  {key}: {value}")
        if 'yield_stats' in data_info:
            print(f"Yield statistics:")
            for key, value in data_info['yield_stats'].items():
                print(f"  {key}: {value:.3f}")

    print(f"\nData splits loaded:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    #For BH reaction n.features is 6144 = 3*768 + 5*768 = 3*chemberta_embedding_dim + 5*chemberta_embedding_dim = 
    # (startingmat_1 + startingmat_2 + product_1) * chemberta_embedding_dim + 
    # (catalyst + base + solvent_1 + additives + water) * chemberta_embedding_dim

    #For SM reaction n.features is 6912 = 3*768 + 6*768 = 3*chemberta_embedding_dim + 6*chemberta_embedding_dim = 
    # (startingmat_1 + startingmat_2 + product_1) * chemberta_embedding_dim + 
    # (solvent_1 + solvent_2 + water + additives + catalyst + base) * chemberta_embedding_dim

    # --- Hyperparameter Handling ---
    xgb_params = {}
    
    if config['hyperparameter_search']:
        print("\n2. Performing hyperparameter search to find optimal model settings...")
        # Define a more focused grid for the search. We encourage the model
        # to be more complex, as this is a complex task.
        local_param_grid = {
            'n_estimators': [500, 800, 1200, 1500],
            'max_depth': [9, 11, 13, 15],
            'learning_rate': [0.02, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'alpha': [0, 0.1, 0.5] # L1 regularization
        }
        search_results = hyperparameter_search(
            X_train, y_train,
            param_grid=local_param_grid,
            cv_folds=config.get('cv_folds', 3), # Reduced folds for speed
            random_state=config['random_state'],
            verbose=True,
            config=config
        )
        xgb_params = search_results['best_params']
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hyperparameter_search_path = os.path.join(output_dirs['reports'], f'hyperparameter_search_{timestamp}.json')
        with open(hyperparameter_search_path, 'w') as f:
            search_results_json = {
                'best_params': search_results['best_params'],
                'best_score': float(search_results['best_score'])
            }
            json.dump(search_results_json, f, indent=2)
    
    if use_wandb:
        wandb.log({"best_cv_score": search_results['best_score']})
        wandb.log({"best_params": search_results['best_params']})


    # --- GPU Acceleration and Parameter Merging ---
    # Start with default model parameters from config
    final_xgb_params = config.get('model_params', {}).copy()
    # Update with parameters from hyperparameter search (these take precedence)
    final_xgb_params.update(xgb_params)

    # Check for GPU device configuration
    # Priority: params -> top-level config -> auto-detect
    if 'device' not in final_xgb_params:
        if config.get('device') == 'cuda' and torch.cuda.is_available():
            print("\nUsing 'cuda' device from top-level config.")
            final_xgb_params['device'] = 'cuda'
        elif torch.cuda.is_available():
            print("\nGPU detected. Enabling XGBoost GPU acceleration.")
            final_xgb_params['device'] = 'cuda'
        else:
            print("\nNo GPU detected, using CPU for training.")

    # Set tree_method if using GPU
    if final_xgb_params.get('device') == 'cuda':
        if 'tree_method' not in final_xgb_params:
            final_xgb_params['tree_method'] = 'gpu_hist'
        print(f"XGBoost will train on GPU with tree_method='{final_xgb_params.get('tree_method')}'.")
    else:
        print("XGBoost will train on CPU.")


    # --- Sample Weighting ---
    sample_weight = None
    if config.get('use_sample_weighting', False):
        print("\nCalculating sample weights to address skewed distribution...")
        # Bin yields to calculate weights - this prevents over-weighting unique high values
        yield_bins = pd.cut(y_train, bins=np.linspace(0, 1, 21), labels=False, include_lowest=True)
        class_counts = np.bincount(yield_bins)
        
        # Calculate weights using a smoothed inverse frequency to avoid extreme values.
        # The previous method was too aggressive for the highly skewed data.
        total_samples = len(y_train)
        # We take the square root to make the weighting less extreme.
        class_weights = np.sqrt(total_samples / (class_counts + 1))

        # Optionally cap max weight
        class_weights = np.clip(class_weights, a_min=None, a_max=100)

        # Map weights back to each sample
        sample_weight = class_weights[yield_bins]
        
        # Normalize weights
        sample_weight /= sample_weight.mean()
        
        print(f"Sample weights calculated. Min weight: {sample_weight.min():.2f}, Max weight: {sample_weight.max():.2f}")


    # Initialize model
    print("\n3. Training XGBoost model...")
    model = XGBoostYieldRegressor(
        quantiles=config['quantiles'],
        n_ensemble=config['n_ensemble'],
        scale_features=config['scale_features'],
        random_state=config['random_state'],
        verbose=True,
        config=config
    )

    # Generate feature names
    # This logic is based on the data construction in _get_data_split
    reaction_type = config.get('rxn_type')
    molecule_names = ['product_1_smiles', 'startingmat_1_smiles', 'startingmat_2_smiles']

    if reaction_type == 'bh':
        # Order from _get_data_split: ['C', 'B', 'S', 'W', 'A']
        condition_names = ['catalyst', 'base', 'solvent_1', 'water', 'additives']
    elif reaction_type == 'sm':
        # Order from _get_data_split: ['S', 'L', 'W', 'A', 'C', 'B']
        condition_names = ['solvent_1', 'solvent_2', 'water', 'additives', 'catalyst', 'base']
    else:
        # Fallback for unknown reaction types, infer from data shape
        if config['representation'] == 'chemberta':
            mol_emb_dim = config.get('chemberta_embedding_dim', 768)
        else:
            mol_emb_dim = config.get('morgan_fp_size', 512)

        num_molecule_features = len(molecule_names) * mol_emb_dim
        num_condition_features = X_train.shape[1] - num_molecule_features
        
        # For hybrid representation, each condition is embedding + 1 flag
        if config['condition_representation'] == 'chemberta':
            cond_emb_dim = config.get('chemberta_embedding_dim', 768)
        else:
            cond_emb_dim = config.get('morgan_fp_size', 512)

        features_per_condition = cond_emb_dim + 1
        
        if num_condition_features > 0 and num_condition_features % features_per_condition == 0:
            num_reagents = num_condition_features // features_per_condition
            condition_names = [f'condition_{i+1}' for i in range(num_reagents)]
        else:
            condition_names = []
            if num_condition_features > 0:
                warnings.warn("Could not determine condition feature names for unknown reaction type.")
    
    all_component_keys = molecule_names + condition_names
    
    feature_names = []
    if X_train.shape[1] > 0:
        if config['representation'] == 'chemberta':
            emb_dim = config.get('chemberta_embedding_dim', 768)
        else: # morgan
            emb_dim = config.get('morgan_fp_size', 512)
        
        # Molecule features
        for component_name in molecule_names:
            feature_names.extend([f"{component_name}_emb_{i}" for i in range(emb_dim)])
        
        # Condition features (embedding + flag)
        if config['condition_representation'] != 'multihot':
            if config['condition_representation'] == 'chemberta':
                condition_emb_dim = config.get('chemberta_embedding_dim', 768)
            else: # morgan
                condition_emb_dim = config.get('morgan_fp_size', 512)
                
            for component_name in condition_names:
                feature_names.extend([f"{component_name}_emb_{i}" for i in range(condition_emb_dim)])
                feature_names.append(f"{component_name}_present_flag")
        else:
            feature_names.extend([f"reagent_{i}" for i in range(reagent_df.shape[0])])


    if not feature_names or len(feature_names) != X_train.shape[1]:
        warnings.warn(
            f"Generated feature names ({len(feature_names)}) do not match data dimension "
            f"({X_train.shape[1]}). Falling back to generic names."
        )
        feature_names = [f'feat_{i}' for i in range(X_train.shape[1])]


    # Train model
    model.fit(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_names,
        sample_weight=sample_weight,
        use_early_stopping=config.get('use_early_stopping', True),
        **final_xgb_params
    )
    
    # Save model
    model_path = os.path.join(output_dirs['models'], f"xgboost_yield_{config['rxn_type']}_{config['representation']}_{config.get('data_type', 'all')}_{config.get('timestamp', '')}.pkl")
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on test set
    print("\n4. Evaluating model...")
    test_metrics = model.evaluate(X_test, y_test, transform_target=config['transform_target'])
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    if 'uncertainty_error_correlation' in test_metrics:
        print(f"  Uncertainty-Error Correlation: {test_metrics['uncertainty_error_correlation']:.4f}")
        print(f"  Mean Uncertainty: {test_metrics['mean_uncertainty']:.4f}")
    
    # Feature importance analysis by component type
    print("\n5. Analyzing feature importance by component...")
    feature_importance = model.get_feature_importance()
    
    # Analyze importance by molecule vs condition components
    molecule_importance = feature_importance[
        feature_importance['feature'].str.startswith(tuple(f"{n}_emb_" for n in molecule_names))
    ]['importance'].sum()

    condition_importance = feature_importance[
        feature_importance['feature'].str.startswith(tuple(f"{n}_emb_" for n in condition_names))
    ]['importance'].sum()
    
    print(f"\nFeature importance by type:")
    print(f"  Molecule components: {molecule_importance:.4f}")
    print(f"  Condition components: {condition_importance:.4f}")
    if condition_importance > 0:
        ratio = molecule_importance / condition_importance
        print(f"  Ratio (molecules/conditions): {ratio:.2f}")
    else:
        ratio = float('inf')
        print("  Ratio (molecules/conditions): inf (condition importance is zero)")

    
    # Component-wise importance
    component_importance = {}
    component_display_map = {
        'product_1_smiles': 'Product',
        'startingmat_1_smiles': 'Starting Material 1',
        'startingmat_2_smiles': 'Starting Material 2',
        'catalyst': 'Catalyst',
        'base': 'Base',
        'solvent_1': 'Solvent 1',
        'solvent_2': 'Solvent 2',
        'water': 'Water',
        'additives': 'Additives'
    }
    
    for component_key in all_component_keys:
        component_features = feature_importance[
            feature_importance['feature'].str.startswith(f"{component_key}_emb_")
        ]
        display_name = component_display_map.get(component_key, component_key.replace('_', ' ').title())
        component_importance[display_name] = component_features['importance'].sum()
    
    print(f"\nFeature importance by reaction component:")
    for component, importance in sorted(component_importance.items(), 
                                      key=lambda x: x[1], reverse=True):
        print(f"  {component}: {importance:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed feature importance
    feature_importance_path = os.path.join(output_dirs['reports'], f'feature_importance_detailed_{timestamp}.csv')
    feature_importance.to_csv(
        feature_importance_path,
        index=False
    )
    
    # Save component importance summary
    component_importance_df = pd.DataFrame([
        {'component': k, 'importance': v} for k, v in component_importance.items()
    ]).sort_values('importance', ascending=False)
    component_importance_path = os.path.join(output_dirs['reports'], f'component_importance_summary_{timestamp}.csv')
    component_importance_df.to_csv(
        component_importance_path,
        index=False
    )
    
    # Generate predictions plot with uncertainty
    if plt is not None:
        print("\n6. Generating prediction plots...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(output_dirs['plots'], f"predictions_vs_actual_{config.get('split_strategy')}_{timestamp}.png")
            model.plot_predictions(X_test, y_test, save_path=plot_path, transform_target=config['transform_target'])
            print(f"Prediction plot saved to: {plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate prediction plot: {e}")
    
    # Save metrics
    metrics_dict = {
        'test_metrics': test_metrics,
        'component_importance': component_importance,
        'molecule_vs_condition_importance': {
            'molecule_importance': float(molecule_importance),
            'condition_importance': float(condition_importance),
            'molecule_condition_ratio': float(ratio)
        },
        'data_info': data_info
    }
    
    # Save metrics summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dirs['reports'], f'metrics_summary_{timestamp}.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=numpy_json_serializer)
    
    # Log to WandB
    if use_wandb:
        log_payload = {
            "test_rmse": test_metrics['rmse'],
            "test_mae": test_metrics['mae'],
            "test_r2": test_metrics['r2'],
            "molecule_importance": molecule_importance,
            "condition_importance": condition_importance,
            "molecule_condition_ratio": ratio,
            **component_importance
        }
        
        if 'uncertainty_error_correlation' in test_metrics:
            log_payload.update({
                "uncertainty_error_correlation": test_metrics['uncertainty_error_correlation'],
                "mean_uncertainty": test_metrics['mean_uncertainty']
            })

        wandb.log(log_payload)
    
    return metrics_dict, model_path, feature_importance_path, component_importance_path, plot_path, final_xgb_params, hyperparameter_search_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train XGBoost model for yield prediction using indexed embeddings')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    config.setdefault('n_test_runs', 5)

    # Load split indices
    split_path = config['split_indices_path']
    directory = os.path.dirname(split_path)
    filename = os.path.basename(split_path)
    name, ext = os.path.splitext(filename)
    
    rtype = config.get("rxn_type")
    data_type = config.get("data_type")
    
    if rtype and data_type:
        new_filename = f"{name}_{rtype}_{data_type}{ext}"
        split_path = os.path.join(directory, new_filename)

    with open(split_path, 'rb') as f:
        split_indices = pkl.load(f)
        
    config['split_indices'] = split_indices
    
    # Initialize wandb if enabled
    use_wandb = False
    if not config.get('no_wandb', False):
        try:
            # The main wandb.init will be handled by the sweep agent or called later
            # For now, we just prepare the config
            print("WandB logging is enabled in config.")
            use_wandb = True
        except Exception as e:
            print(f"Warning: Could not initialize Weights & Biases: {e}")
    # --- End of new logic ---


    print("="*60)
    print("XGBOOST YIELD PREDICTION WITH INDEXED EMBEDDINGS")
    print("="*60)

    config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up base output directory
    base_output_dir = config.get('output_dir')
    if base_output_dir is None:
        base_output_dir = f"outputs/xgboost_yield_{config['rxn_type']}_{config['representation']}_{config['data_type']}_{config.get('split_strategy')}_{config['timestamp']}"
    config['output_dir'] = base_output_dir
    print(f"Base output directory: {base_output_dir}")

    # Check if condition embeddings are available
    # This logic can be simplified as cache path is in config
    condition_cache_path = config.get('condition_cache_path')
    reagents_dir = config.get('reagents_dir')
    embeddings_available = False
    if condition_cache_path and os.path.exists(condition_cache_path):
        print(f"✓ Condition cache found: {condition_cache_path}")
        embeddings_available = True
    elif reagents_dir and os.path.exists(reagents_dir):
        # Check if reagent dataframes have embeddings
        print(f"✓ Checking for embeddings in reagent dataframes: {reagents_dir}")
        
        # Quick check for embedding columns
        has_embeddings = False
        for filename in os.listdir(reagents_dir):
            #take the file for the reaction and data type
            if filename == f'{config['rxn_type']}_treshold_all_{config['data_type']}_reagent_df.csv':
                reagent_df = pd.read_csv(os.path.join(reagents_dir, filename))
                if 'embedding' in reagent_df.columns:
                    has_embeddings = True
                    break
        
        if has_embeddings:
            print("✓ Found embeddings in reagent dataframes")
            embeddings_available = True
        else:
            print("✗ No embeddings found in reagent dataframes")
    
    if not embeddings_available:
        print("\n" + "="*60)
        print("ERROR: No condition embeddings found!")
        print("Please run one of the following:")
        print("1. python precompute_condition_embeddings.py  # Add embeddings to dataframes")
        print("2. Provide --condition_cache_path with valid cache file")
        print("="*60)
        return
    
    
    #initialize chemberta
    if config['representation'] == 'chemberta' or config['condition_representation'] == 'chemberta':
        tokenizer, chemberta_model, chemberta_embedding_dim = _initialize_chemberta(config)
    else:
        chemberta_embedding_dim = 0
        tokenizer = None
        chemberta_model = None
    
    # --- Auto-detect embedding/fingerprint dimensions ---
    reagent_df_path = os.path.join(config['reagents_dir'], f"{config['rxn_type']}_treshold_all_{config['data_type']}_reagent_df.csv")
    if not os.path.exists(reagent_df_path):
        raise FileNotFoundError(f"Reagent dataframe not found at: {reagent_df_path}")
    
    reagent_df = pd.read_csv(reagent_df_path)
    
    # Detect ChemBERTa embedding dimension
    if 'embedding' in reagent_df.columns and pd.notna(reagent_df['embedding'].iloc[0]):
        # Safely parse the string representation of the list
        first_emb_str = reagent_df['embedding'].iloc[0]
        detected_chemberta_dim = len(first_emb_str.replace(" .. ", "").strip().rstrip(',').split(','))
        config['chemberta_embedding_dim'] = detected_chemberta_dim
        if chemberta_embedding_dim != detected_chemberta_dim:
            warnings.warn(f"Mismatch between model config dim ({chemberta_embedding_dim}) and detected dim ({detected_chemberta_dim}). Using detected dim.")
            chemberta_embedding_dim = detected_chemberta_dim
        print(f"✓ Auto-detected ChemBERTa embedding dimension: {chemberta_embedding_dim}")

    # Detect Morgan fingerprint dimension
    if 'morgan_fp' in reagent_df.columns and pd.notna(reagent_df['morgan_fp'].iloc[0]):
        first_fp_str = reagent_df['morgan_fp'].iloc[0]
        detected_morgan_dim = len(first_fp_str.replace(" .. ", "").strip().rstrip(',').split(','))
        config['morgan_fp_size'] = detected_morgan_dim
        print(f"✓ Auto-detected Morgan fingerprint dimension: {detected_morgan_dim}")

    # Load dataset
    print(f"\nLoading dataset for {config['rxn_type']} - {config['data_type']}")
    try:
        file_path = f"data/{config['rxn_type']}_treshold_all_{config['data_type']}_processed.npz"
        loaded_npz = np.load(file_path, allow_pickle=True)
        if 'data' not in loaded_npz:
            err_msg = f"ERROR: 'data' key not found in the .npz file: {file_path}"
            raise ValueError(err_msg)

        reaction_data_unpacked = loaded_npz['data']
        # print("Reactiond data unpacked: ", reaction_data_unpacked)
        if isinstance(reaction_data_unpacked, np.ndarray) and len(reaction_data_unpacked) == 2:
            reaction_dict, _ = reaction_data_unpacked
            if not isinstance(reaction_dict, dict): 
                raise ValueError("first element should be the reaciton dict (type: dict)")

        else:
            err_msg = (f"ERROR: Expected 'data' in {file_path} to be a 2-element np.array "
                       f"to unpack into reaction_dict and clist. Got type: {type(reaction_data_unpacked)}")
            raise ValueError(err_msg)
    except Exception as e:
        raise ValueError(f"Error loading reaction data: {e}")
    
    if config["remove_duplicates"]:
        #check for duplicates in reaction_dict
        duplicates = find_duplicate_reactions(reaction_dict)
        if len(duplicates) > 0:
            print("Found duplicates in reaction_dict")
            reaction_dict = aggregate_reaction_dict(reaction_dict, aggfunc='mean')
            print("Aggregated reaction_dict")

    all_test_metrics = []
    best_r2 = -float('inf')
    best_model_path = None
    best_feature_importance_path = None
    best_component_importance_path = None
    best_plot_path = None
    best_hyperparameter_search_path = None
    best_config_path = None
    n_runs = config.get('n_test_runs', 5)
    best_hyperparameters = {}

    for i in range(n_runs):
        print(f"\n\n{'='*25} RUN {i+1}/{n_runs} {'='*25}")

        run_config = config.copy()
        run_config['random_state'] = config['random_state'] + i
        run_config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Each run will save to a temporary directory
        run_output_dir = os.path.join(base_output_dir, f"run_{i+1}")
        output_dirs = create_output_directories(run_output_dir)

        config_path = os.path.join(output_dirs['reports'], f"config_{run_config['timestamp']}.json")
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=2, default=numpy_json_serializer)

        # --- DATA SPLITTING ---
        split_strategy = run_config.get('split_strategy', 'by_reaction') # Default to original strategy
        print(f"\nSplitting data using '{split_strategy}' strategy...")

        if split_strategy == 'by_reaction':
            # --- ORIGINAL STRATEGY: Split by unique reactions ---
            print("\nSplitting data using pre-computed indices...")
            
            split_indices = run_config['split_indices']
            test_indices = split_indices['test_indices']

            # Consolidate all non-test data 
            n_total = len(reaction_dict)
            all_indices = np.arange(n_total)
            dev_indices = np.setdiff1d(all_indices, test_indices)

            # Stratified split with safe fallback for small datasets
            print("\nPerforming stratified split on development data...")
            dev_keys = [list(reaction_dict.keys())[i] for i in dev_indices]
            dev_max_yields = np.array([max(cond[2] for cond in reaction_dict[key]) for key in dev_keys])

            n_bins = min(10, len(np.unique(dev_max_yields)))
            yield_strata = pd.qcut(dev_max_yields, q=n_bins, labels=False, duplicates='drop') if n_bins > 1 else None

            try:
                train_indices, val_indices = train_test_split(
                    dev_indices,
                    test_size=0.15,
                    random_state=run_config['random_state'],
                    stratify=yield_strata
                )
            except ValueError as e:
                print(f"Stratified split failed ({e}). Falling back to non-stratified split.")
                train_indices, val_indices = train_test_split(
                    dev_indices,
                    test_size=0.15,
                    random_state=run_config['random_state'],
                    stratify=None
                )

            print(f"Data split into: "
                  f"{len(train_indices)} train reactions, "
                  f"{len(val_indices)} validation reactions, "
                  f"{len(test_indices)} test reactions.")
            
            # Generate embeddings for each split
            train_embs, train_yields = _get_data_split(train_indices, reaction_dict, reagent_df, tokenizer, chemberta_model, run_config)
            val_embs, val_yields = _get_data_split(val_indices, reaction_dict, reagent_df, tokenizer, chemberta_model, run_config)
            test_embs, test_yields = _get_data_split(test_indices, reaction_dict, reagent_df, tokenizer, chemberta_model, run_config)

            X_train = torch.cat(train_embs, dim=0).numpy()
            y_train = np.array(train_yields)
            X_val = torch.cat(val_embs, dim=0).numpy()
            y_val = np.array(val_yields)
            X_test = torch.cat(test_embs, dim=0).numpy()
            y_test = np.array(test_yields)

        elif split_strategy == 'point_wise':
            # --- DIAGNOSTIC STRATEGY: Split by individual data points ---
            print("\nGenerating all embeddings for point-wise splitting...")
            all_reaction_indices = np.arange(len(reaction_dict))
            all_embs, all_yields = _get_data_split(all_reaction_indices, reaction_dict, reagent_df, tokenizer, chemberta_model, run_config)

            X_all = torch.cat(all_embs, dim=0).numpy()
            y_all = np.array(all_yields)

            print(f"Total data points generated: {len(y_all)}")
            print("\nPerforming stratified split on all data points...")

            n_bins = min(10, len(np.unique(y_all)))
            stratify_col = pd.qcut(y_all, q=n_bins, labels=False, duplicates='drop') if n_bins > 1 else None

            try:
                X_dev, X_test, y_dev, y_test = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=run_config['random_state'], stratify=stratify_col
                )
            except ValueError as e:
                print(f"Stratified split (dev/test) failed ({e}). Falling back to non-stratified split.")
                X_dev, X_test, y_dev, y_test = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=run_config['random_state'], stratify=None
                )
            
            n_bins_dev = min(10, len(np.unique(y_dev)))
            stratify_dev_col = pd.qcut(y_dev, q=n_bins_dev, labels=False, duplicates='drop') if n_bins_dev > 1 else None

            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_dev, y_dev, test_size=0.15, random_state=run_config['random_state'], stratify=stratify_dev_col
                )
            except ValueError as e:
                print(f"Stratified split (train/val) failed ({e}). Falling back to non-stratified split.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_dev, y_dev, test_size=0.15, random_state=run_config['random_state'], stratify=None
                )
        else:
            raise ValueError(f"Invalid split_strategy: '{split_strategy}'. Must be 'by_reaction' or 'point_wise'.")


        print(f"Data shapes ready for training: \n"
              f"  Train: X={X_train.shape}, y={y_train.shape}\n"
              f"  Val:   X={X_val.shape}, y={y_val.shape}\n"
              f"  Test:  X={X_test.shape}, y={y_test.shape}")

        datasets = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

        # Setup WandB for this run
        if use_wandb and wandb is not None:
            if wandb.run is not None:
                wandb.finish() # Finish previous run

            wandb.init(
                project=run_config.get('wandb_project', 'xgboost-yield-indexed'),
                config=run_config,
                name=f"xgb_yield_{run_config.get('reaction_type', 'unknown')}_{run_config['representation']}_{run_config['timestamp']}_run{i+1}",
                group=f"run_group_{config['timestamp']}",
                reinit=True
            )
            if wandb.run is not None:
                run_config.update(wandb.config)
                
                if run_config.get('hyperparameter_search', False):
                    sweep_params_for_grid = {
                        k: [v] for k, v in wandb.config.items() if k in run_config.get('param_grid', {})
                    }
                    if sweep_params_for_grid:
                        print("Updating param_grid with sweep parameters...")
                        run_config['param_grid'].update(sweep_params_for_grid)
        try:
            if i == 0 and run_config.get('hyperparameter_search', False):
                run_config['hyperparameter_search'] = True
            else:
                run_config['hyperparameter_search'] = False
                if best_hyperparameters:
                    run_config['model_params'] = best_hyperparameters

            # Train and evaluate model
            results, model_path, feature_importance_path, component_importance_path, plot_path, final_xgb_params, hyperparameter_search_path = train_and_evaluate_model(
                datasets=datasets,
                config=run_config,
                output_dirs=output_dirs,
                use_wandb=use_wandb,
                reagent_df=reagent_df
            )
            
            if i == 0:
                best_hyperparameters = final_xgb_params

            all_test_metrics.append(results['test_metrics'])
            
            # Check for best model
            if results['test_metrics']['r2'] > best_r2:
                best_r2 = results['test_metrics']['r2']
                best_model_path = model_path
                best_feature_importance_path = feature_importance_path
                best_component_importance_path = component_importance_path
                best_plot_path = plot_path
                best_hyperparameter_search_path = hyperparameter_search_path
                best_config_path = config_path

            
            print("\n" + "="*60)
            print(f"RUN {i+1} COMPLETE")
            print("="*60)
            print(f"Results for this run saved to: {run_output_dir}")
            print(f"Model performance for this run:")
            print(f"  RMSE: {results['test_metrics']['rmse']:.4f}")
            print(f"  MAE: {results['test_metrics']['mae']:.4f}")
            print(f"  R²: {results['test_metrics']['r2']:.4f}")
            
        except Exception as e:
            print(f"\nError during run {i+1}: {e}")
            if use_wandb and wandb is not None and wandb.run:
                wandb.log({"error": str(e)})
            # Continue to next run
        finally:
            if use_wandb and wandb is not None and wandb.run:
                wandb.finish()

    # --- AGGREGATE RESULTS & SAVE BEST MODEL ---
    if all_test_metrics:
        print("\n\n" + "="*60)
        print("AGGREGATED RESULTS FROM ALL RUNS")
        print("="*60)
        
        df_metrics = pd.DataFrame(all_test_metrics)
        
        data_type = config['data_type']
        rxn_type = config['rxn_type']
        representation = config['representation']
        
        split_strategy = config.get('split_strategy', 'by_reaction')
        # Save detailed metrics for all runs
        df_metrics.to_csv(os.path.join(base_output_dir, f'all_runs_test_metrics_{data_type}_{rxn_type}_{representation}_{split_strategy}.csv'), index=False)
        
        # Calculate and print summary
        summary = df_metrics.describe().transpose()
        print("\nTest Metrics Summary:")
        print(summary[['mean', 'std', 'min', 'max']])
        summary.to_csv(os.path.join(base_output_dir, f'summary_test_metrics_{data_type}_{rxn_type}_{representation}_{split_strategy}.csv'))
        
        # --- Save Best Model and Reports ---
        main_reports_dir = os.path.join(base_output_dir, 'reports')
        os.makedirs(main_reports_dir, exist_ok=True)
        main_plots_dir = os.path.join(base_output_dir, 'plots')
        os.makedirs(main_plots_dir, exist_ok=True)


        # Save the best model to the main directory and clean up
        if best_model_path and os.path.exists(best_model_path):
            final_model_name = f"best_model_{data_type}_{rxn_type}_{representation}_{split_strategy}.pkl"
            final_model_dest = os.path.join(base_output_dir, 'models', final_model_name)
            os.makedirs(os.path.dirname(final_model_dest), exist_ok=True)
            
            shutil.copy(best_model_path, final_model_dest)
            print(f"\nBest model (R²={best_r2:.4f}) saved to: {final_model_dest}")

            # Save corresponding reports
            if best_feature_importance_path and os.path.exists(best_feature_importance_path):
                shutil.copy(best_feature_importance_path, os.path.join(main_reports_dir, f"best_feature_importance_{data_type}_{rxn_type}_{representation}_{split_strategy}.csv"))
            if best_component_importance_path and os.path.exists(best_component_importance_path):
                shutil.copy(best_component_importance_path, os.path.join(main_reports_dir, f"best_component_importance_{data_type}_{rxn_type}_{representation}_{split_strategy}.csv"))
            if best_plot_path and os.path.exists(best_plot_path):
                shutil.copy(best_plot_path, os.path.join(main_plots_dir, f"best_plot_{data_type}_{rxn_type}_{representation}_{split_strategy}.png"))
            if best_hyperparameter_search_path and os.path.exists(best_hyperparameter_search_path):
                shutil.copy(best_hyperparameter_search_path, os.path.join(main_reports_dir, f"best_hyperparameter_search_{data_type}_{rxn_type}_{representation}_{split_strategy}.json"))
            if best_config_path and os.path.exists(best_config_path):
                shutil.copy(best_config_path, os.path.join(main_reports_dir, f"best_config_{data_type}_{rxn_type}_{representation}_{split_strategy}.json"))


            
            # Save hyperparams used for the best model
            with open(os.path.join(main_reports_dir, f'best_model_hyperparameters_{data_type}_{rxn_type}_{representation}_{split_strategy}.json'), 'w') as f:
                json.dump(best_hyperparameters, f, indent=2)

            # Clean up run directories
            for i in range(n_runs):
                run_dir = os.path.join(base_output_dir, f"run_{i+1}")
                if os.path.isdir(run_dir):
                    shutil.rmtree(run_dir)
            print("Cleaned up temporary run directories.")

        else:
            print("\nWarning: Could not find the best model file to save.")
            
    else:
        print("\nNo successful runs to aggregate results.")


if __name__ == "__main__":
    main() 