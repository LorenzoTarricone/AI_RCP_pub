#!/usr/bin/env python3
"""
Pre-compute ChemBERTa embeddings for all reaction conditions.

This script:
1. Loads all reagent dataframes 
2. Extracts unique SMILES for each condition type
3. Computes ChemBERTa embeddings for all conditions
4. Saves embeddings with proper indexing for efficient retrieval

Usage:
    python precompute_condition_embeddings.py
"""

import os
import pandas as pd
import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
import warnings

# ChemBERTa imports
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Warning: Transformers library not found. ChemBERTa embeddings will not work.")
    AutoTokenizer, AutoModel = None, None

# RDKit imports
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("Warning: RDKit not found. SMILES validation will be skipped.")
    Chem = None

from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_morgan_fp(smiles, n_bits=512, radius=2):
    if not smiles or pd.isna(smiles):
        return np.zeros(n_bits, dtype=int)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Could not convert SMILES to Morgan fingerprint: {smiles}")
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class ConditionEmbeddingComputer:
    """
    Computes and manages ChemBERTa embeddings for reaction conditions.
    """
    
    def __init__(
        self, 
        device: str = "cuda",
        max_length: int = 512,
        validate_smiles: bool = True,
        verbose: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.validate_smiles = validate_smiles
        self.verbose = verbose
        
        # Initialize ChemBERTa
        self._initialize_chemberta()
        
        # Storage for mappings
        self.condition_mappings = {}  # Map condition codes to SMILES
        self.condition_embeddings = {}  # Map condition codes to embeddings
        
    def _initialize_chemberta(self):
        """Initialize ChemBERTa model and tokenizer."""
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("Transformers library is required for ChemBERTa embeddings")
        
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        
        if self.verbose:
            print(f"Initializing ChemBERTa model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.embedding_dim = self.model.config.hidden_size
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
            if self.verbose:
                print(f"ChemBERTa initialized on {self.device}, embedding dim: {self.embedding_dim}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChemBERTa: {e}")
    
    def _validate_smiles(self, smiles: str) -> bool:
        """Validate a SMILES string using RDKit."""
        if not self.validate_smiles or Chem is None:
            return True
        
        if pd.isna(smiles) or smiles == '' or smiles == 'NoAdditive':
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _get_smiles_from_condition_name(self, condition_name: str, reaction_data: pd.DataFrame) -> str:
        """
        Extract SMILES for a condition name from reaction data.
        
        This method maps condition names back to their SMILES representations
        by looking them up in the main reaction dataset.
        """
        # Handle special cases
        if pd.isna(condition_name) or condition_name == '' or condition_name == 'NoAdditive':
            return ''
        
        if condition_name == 'True':  # Water case
            return 'O'  # Water SMILES
        
        # Strip suffixes like _A, _B, _S from condition names before lookup
        search_name = condition_name
        if isinstance(condition_name, str) and len(condition_name) > 2 and condition_name[-2] == '_':
            suffix = condition_name[-1]
            if suffix.isalpha() and suffix.isupper():
                search_name = condition_name[:-2]
        
        # Special cases
        if search_name == 'Alphos':
            return 'CCCCC1=C(F)C(F)=C(C2=C(C(C)C)C=C(C(C)C)C(C3=C(OC)C=CC=C3P(C45CC6CC(C4)CC(C6)C5)C78CC9CC(C7)CC(C9)C8)=C2C(C)C)C(F)=C1F'
        if search_name == 'SPhos':
            return 'COc1cccc(OC)c1-c2ccccc2P(C3CCCCC3)C4CCCCC4'
        
        
        # Search through different condition types in the reaction data
        condition_mappings = {
            'catalyst_name': 'catalyst_smiles',
            'reagent_1_name': 'reagent_1_smiles', 
            'solvent_1_name': 'solvent_1_smiles',
            'additives_name_merged': 'additives_smiles_merged'
        }
        
        for name_col, smiles_col in condition_mappings.items():
            if name_col in reaction_data.columns and smiles_col in reaction_data.columns:
                matches = reaction_data[reaction_data[name_col] == search_name]
                if not matches.empty:
                    smiles_values = matches[smiles_col].dropna().unique()
                    if len(smiles_values) > 0:
                        return smiles_values[0]  # Return first valid SMILES
        
        if self.verbose:
            print(f"Warning: Could not find SMILES for condition '{condition_name}'")
        return ''
    
    def _get_embedding_from_smiles(self, smiles: str) -> torch.Tensor:
        """
        Generate ChemBERTa embedding for a SMILES string.
        Handles comma-separated SMILES by embedding each part and summing them.
        
        Args:
            smiles (str): SMILES string, potentially comma-separated.
            
        Returns:
            torch.Tensor: ChemBERTa embedding of shape [embedding_dim].
        """
        # Handle empty/invalid SMILES
        if pd.isna(smiles) or smiles == '' or smiles == 'NoAdditive':
            return torch.zeros(self.embedding_dim, device='cpu')
        
        smiles_parts = [s.strip() for s in smiles.split(',') if s.strip()]
        
        if not smiles_parts:
            return torch.zeros(self.embedding_dim, device='cpu')
            
        total_embedding = torch.zeros(self.embedding_dim, device='cpu')
        
        for part in smiles_parts:
            # Validate SMILES if requested
            if not self._validate_smiles(part):
                if self.verbose:
                    print(f"Warning: Invalid SMILES part: '{part}' in '{smiles}'")
                continue  # Skip invalid parts, effectively adding a zero vector.
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    part,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                
                total_embedding += embedding.cpu()
                
            except Exception as e:
                if self.verbose:
                    print(f"Error generating embedding for SMILES part '{part}': {e}")
                # Also skip parts that cause errors
        
        return total_embedding
    
    def load_reagent_dataframes(self, reagents_dir: str = "reagents_dfs") -> Dict[str, pd.DataFrame]:
        """Load all reagent dataframes."""
        reagent_dfs = {}
        
        if not os.path.exists(reagents_dir):
            raise FileNotFoundError(f"Reagents directory not found: {reagents_dir}")
        
        for filename in os.listdir(reagents_dir):
            if filename.endswith('_reagent_df.csv'):
                file_path = os.path.join(reagents_dir, filename)
                # Extract dataset identifier from filename
                dataset_id = filename.replace('_reagent_df.csv', '')
                reagent_dfs[dataset_id] = pd.read_csv(file_path)
                if self.verbose:
                    print(f"Loaded {dataset_id}: {len(reagent_dfs[dataset_id])} conditions")
        
        return reagent_dfs
    
    def load_reaction_data(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """Load reaction data files for SMILES lookup."""
        reaction_data = {}
        
        data_files = {
            'bh_treshold_all_all': 'bh_data_clean_all_whitelisted.csv',
            'bh_treshold_all_positive': 'bh_data_clean_positive_whitelisted.csv', 
            'sm_treshold_all_all': 'sm_data_clean_all_whitelisted.csv',
            'sm_treshold_all_positive': 'sm_data_clean_positive_whitelisted.csv'
        }
        
        for dataset_id, filename in data_files.items():
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                if self.verbose:
                    print(f"Loading reaction data: {filename}")
                reaction_data[dataset_id] = pd.read_csv(file_path)
            else:
                print(f"Warning: Reaction data file not found: {file_path}")
        
        return reaction_data
    
    def compute_all_embeddings(
        self, 
        reagents_dir: str = "reagents_dfs",
        data_dir: str = "data",
        output_dir: str = "condition_embeddings_cache",
        add_to_dataframes: bool = True,
        save_cache_files: bool = True
    ):
        """
        Compute embeddings for all conditions across all datasets.
        
        Args:
            reagents_dir: Directory containing reagent dataframes
            data_dir: Directory containing reaction data files  
            output_dir: Directory to save cached embeddings
            add_to_dataframes: If True, add embeddings as a column to reagent dataframes
            save_cache_files: If True, save separate cache files for backward compatibility
        """
        # Create output directory if saving cache files
        if save_cache_files:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load reagent dataframes and reaction data
        reagent_dfs = self.load_reagent_dataframes(reagents_dir)
        reaction_data = self.load_reaction_data(data_dir)
        
        # Process each dataset
        for dataset_id, reagent_df in reagent_dfs.items():
            if self.verbose:
                print(f"\nProcessing dataset: {dataset_id}")
            
            # Get corresponding reaction data for SMILES lookup
            if dataset_id not in reaction_data:
                print(f"Warning: No reaction data found for {dataset_id}, skipping...")
                continue
            
            df_reaction = reaction_data[dataset_id]
            
            # Initialize storage for this dataset (for cache files if needed)
            condition_mappings = {}
            condition_embeddings = {}
            
            # Initialize list to store embeddings for dataframe column
            embeddings_for_df = []
            morgan_fps_for_df = []
            
            # Process each condition in the reagent dataframe
            for idx, row in tqdm(reagent_df.iterrows(), total=len(reagent_df), 
                               desc=f"Computing embeddings for {dataset_id}"):
                
                condition_code = row['reagent_type_numbered']  # e.g., C1, B2, S3
                condition_name = row['reagent']
                
                # Get SMILES for this condition
                smiles = self._get_smiles_from_condition_name(condition_name, df_reaction)
                
                # Compute embedding
                embedding = self._get_embedding_from_smiles(smiles)
                embedding_numpy = embedding.numpy()
                
                # Compute Morgan fingerprint
                morgan_fp = smiles_to_morgan_fp(smiles)
                
                # Store for dataframe
                embeddings_for_df.append(embedding_numpy)
                morgan_fps_for_df.append(morgan_fp)
                
                # Store for cache files if needed
                if save_cache_files:
                    condition_mappings[condition_code] = {
                        'name': condition_name,
                        'smiles': smiles,
                        'type': row['reagent_type'],
                        'count': row['count']
                    }
                    condition_embeddings[condition_code] = embedding_numpy
            
            # Add embeddings as a new column to the reagent dataframe
            if add_to_dataframes:
                # Convert list of numpy arrays to a list for storage in dataframe
                # Note: pandas can store numpy arrays in cells
                reagent_df['embedding'] = embeddings_for_df
                reagent_df['morgan_fp'] = morgan_fps_for_df
                
                # Save the modified dataframe back to the original file
                original_file_path = os.path.join(reagents_dir, f"{dataset_id}_reagent_df.csv")
                
                # For CSV storage, we need to serialize the numpy arrays
                # We'll store them as comma-separated strings
                reagent_df_to_save = reagent_df.copy()
                reagent_df_to_save['embedding'] = reagent_df_to_save['embedding'].apply(
                    lambda x: ','.join(map(str, x))
                )
                reagent_df_to_save['morgan_fp'] = reagent_df_to_save['morgan_fp'].apply(lambda x: ','.join(map(str, x)))
                
                reagent_df_to_save.to_csv(original_file_path, index=False)
                
                if self.verbose:
                    print(f"Added embeddings column to {original_file_path}")
                    print(f"Embedding shape per condition: {embeddings_for_df[0].shape}")
            
            # Save cache files for backward compatibility if requested
            if save_cache_files:
                dataset_cache = {
                    'mappings': condition_mappings,
                    'embeddings': condition_embeddings,
                    'embedding_dim': self.embedding_dim,
                    'dataset_id': dataset_id
                }
                
                cache_path = os.path.join(output_dir, f"{dataset_id}_condition_embeddings.pkl")
                with open(cache_path, 'wb') as f:
                    pkl.dump(dataset_cache, f)
                
                if self.verbose:
                    print(f"Saved {len(condition_embeddings)} condition embeddings to {cache_path}")
    
    def create_unified_embedding_cache(self, output_dir: str = "condition_embeddings_cache"):
        """
        Create a unified cache with all unique conditions across all datasets.
        This allows sharing embeddings between datasets when conditions are the same.
        """
        if not os.path.exists(output_dir):
            print(f"Output directory {output_dir} does not exist. Run compute_all_embeddings first.")
            return
        
        # Load all individual dataset caches
        unified_mappings = {}
        unified_embeddings = {}
        
        for filename in os.listdir(output_dir):
            if filename.endswith('_condition_embeddings.pkl'):
                cache_path = os.path.join(output_dir, filename)
                with open(cache_path, 'rb') as f:
                    dataset_cache = pkl.load(f)
                
                # Merge into unified cache
                unified_mappings.update(dataset_cache['mappings'])
                unified_embeddings.update(dataset_cache['embeddings'])
        
        # Create unified cache
        unified_cache = {
            'mappings': unified_mappings,
            'embeddings': unified_embeddings,
            'embedding_dim': self.embedding_dim
        }
        
        unified_cache_path = os.path.join(output_dir, 'unified_condition_embeddings.pkl')
        with open(unified_cache_path, 'wb') as f:
            pkl.dump(unified_cache, f)
        
        if self.verbose:
            print(f"Created unified cache with {len(unified_embeddings)} unique conditions")
            print(f"Saved to: {unified_cache_path}")

    def load_reagent_dataframes_with_embeddings(self, reagents_dir: str = "reagents_dfs") -> Dict[str, pd.DataFrame]:
        """
        Load reagent dataframes and parse embeddings from string format back to numpy arrays.
        
        Returns:
            Dict mapping dataset_id to dataframe with 'embedding' column containing numpy arrays
        """
        reagent_dfs = {}
        
        if not os.path.exists(reagents_dir):
            raise FileNotFoundError(f"Reagents directory not found: {reagents_dir}")
        
        for filename in os.listdir(reagents_dir):
            if filename.endswith('_reagent_df.csv'):
                file_path = os.path.join(reagents_dir, filename)
                # Extract dataset identifier from filename
                dataset_id = filename.replace('_reagent_df.csv', '')
                
                df = pd.read_csv(file_path)
                
                # Check if embeddings column exists
                if 'embedding' in df.columns:
                    # Parse embeddings from string format back to numpy arrays
                    df['embedding'] = df['embedding'].apply(
                        lambda x: np.fromstring(x, sep=',') if isinstance(x, str) else x
                    )
                if 'morgan_fp' in df.columns:
                    df['morgan_fp'] = df['morgan_fp'].apply(
                        lambda x: np.fromstring(x, sep=',') if isinstance(x, str) else x
                    )
                
                if self.verbose:
                    print(f"Loaded {dataset_id}: {len(df)} conditions with embeddings")
                else:
                    print(f"Loaded {dataset_id}: {len(df)} conditions (no embeddings found)")
                
                reagent_dfs[dataset_id] = df
        
        return reagent_dfs

    def check_dataframe_embeddings(self, reagents_dir: str = "reagents_dfs") -> Dict[str, bool]:
        """
        Check which reagent dataframes already have embeddings.
        
        Returns:
            Dict mapping dataset_id to boolean indicating if embeddings exist
        """
        embedding_status = {}
        
        if not os.path.exists(reagents_dir):
            print(f"Reagents directory not found: {reagents_dir}")
            return embedding_status
        
        for filename in os.listdir(reagents_dir):
            if filename.endswith('_reagent_df.csv'):
                dataset_id = filename.replace('_reagent_df.csv', '')
                file_path = os.path.join(reagents_dir, filename)
                
                try:
                    df = pd.read_csv(file_path)
                    has_embeddings = 'embedding' in df.columns
                    embedding_status[dataset_id] = has_embeddings
                    
                    if self.verbose:
                        status = "✓" if has_embeddings else "✗"
                        print(f"{status} {dataset_id}: {'Has embeddings' if has_embeddings else 'No embeddings'}")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"✗ {dataset_id}: Error reading file - {e}")
                    embedding_status[dataset_id] = False
        
        return embedding_status
    
    def verify_embedding_integrity(self, reagents_dir: str = "reagents_dfs") -> bool:
        """
        Verify that embeddings in dataframes are valid and properly formatted.
        
        Returns:
            True if all embeddings are valid, False otherwise
        """
        if self.verbose:
            print("Verifying embedding integrity...")
        
        all_valid = True
        
        for filename in os.listdir(reagents_dir):
            if filename.endswith('_reagent_df.csv'):
                dataset_id = filename.replace('_reagent_df.csv', '')
                file_path = os.path.join(reagents_dir, filename)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    if 'embedding' not in df.columns:
                        continue
                    
                    # Check each embedding
                    for idx, row in df.iterrows():
                        embedding_str = row['embedding']
                        
                        if not isinstance(embedding_str, str) or not embedding_str.strip():
                            if self.verbose:
                                print(f"✗ {dataset_id}: Invalid embedding at row {idx}")
                            all_valid = False
                            continue
                        
                        try:
                            embedding = np.fromstring(embedding_str, sep=',')
                            
                            # Check if embedding has the expected dimension
                            if hasattr(self, 'embedding_dim') and len(embedding) != self.embedding_dim:
                                if self.verbose:
                                    print(f"✗ {dataset_id}: Wrong embedding dimension at row {idx} "
                                         f"(expected {self.embedding_dim}, got {len(embedding)})")
                                all_valid = False
                            
                        except Exception as e:
                            if self.verbose:
                                print(f"✗ {dataset_id}: Could not parse embedding at row {idx} - {e}")
                            all_valid = False
                    
                    if self.verbose:
                        print(f"✓ {dataset_id}: All embeddings valid")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"✗ {dataset_id}: Error reading file - {e}")
                    all_valid = False
        
        return all_valid


def main():
    parser = argparse.ArgumentParser(description="Pre-compute ChemBERTa embeddings for reaction conditions")
    parser.add_argument("--reagents_dir", type=str, default="reagents_dfs", 
                       help="Directory containing reagent dataframes")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing reaction data files")
    parser.add_argument("--output_dir", type=str, default="condition_embeddings_cache",
                       help="Directory to save computed embeddings")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for ChemBERTa computations")
    parser.add_argument("--no_validation", action="store_true",
                       help="Skip SMILES validation")
    parser.add_argument("--unified_only", action="store_true",
                       help="Only create unified cache (assumes individual caches exist)")
    parser.add_argument("--no_dataframe_embeddings", action="store_true",
                       help="Don't add embeddings as columns to reagent dataframes")
    parser.add_argument("--no_cache_files", action="store_true",
                       help="Don't save separate cache files (only modify dataframes)")
    parser.add_argument("--check_embeddings", action="store_true",
                       help="Check which dataframes already have embeddings")
    parser.add_argument("--verify_embeddings", action="store_true",
                       help="Verify integrity of embeddings in dataframes")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CONDITION EMBEDDING PRE-COMPUTATION")
    print("="*60)
    
    # Initialize computer
    computer = ConditionEmbeddingComputer(
        device=args.device,
        validate_smiles=not args.no_validation,
        verbose=True
    )
    
    if args.check_embeddings:
        # Check which dataframes have embeddings
        print("\nChecking embedding status in reagent dataframes:")
        print("-" * 50)
        embedding_status = computer.check_dataframe_embeddings(args.reagents_dir)
        
        if embedding_status:
            n_with_embeddings = sum(embedding_status.values())
            n_total = len(embedding_status)
            print(f"\nSummary: {n_with_embeddings}/{n_total} dataframes have embeddings")
        
        return
    
    if args.verify_embeddings:
        # Verify embedding integrity
        print("\nVerifying embedding integrity:")
        print("-" * 50)
        is_valid = computer.verify_embedding_integrity(args.reagents_dir)
        
        if is_valid:
            print("\n✓ All embeddings are valid and properly formatted")
        else:
            print("\n✗ Some embeddings have issues - see details above")
        
        return
    
    if args.unified_only:
        # Only create unified cache
        computer.create_unified_embedding_cache(args.output_dir)
    else:
        # Compute all embeddings
        print(f"Computing embeddings for {args.reagents_dir} and {args.data_dir} and saving to {args.output_dir}")
        computer.compute_all_embeddings(
            reagents_dir=args.reagents_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            add_to_dataframes=not args.no_dataframe_embeddings,
            save_cache_files=not args.no_cache_files
        )
        
        # Also create unified cache if cache files were created
        if not args.no_cache_files:
            computer.create_unified_embedding_cache(args.output_dir)
    
    print("\n" + "="*60)
    print("CONDITION EMBEDDING COMPUTATION COMPLETE")
    if not args.no_dataframe_embeddings:
        print("✓ Embeddings added as columns to reagent dataframes")
    if not args.no_cache_files and not args.unified_only:
        print("✓ Cache files saved for backward compatibility")
    print("="*60)


if __name__ == "__main__":
    main() 