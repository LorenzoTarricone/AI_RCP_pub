import os
import numpy as np
import pandas as pd
import torch
import warnings
from tqdm import tqdm
import pickle as pkl
from typing import List, Dict, Optional, Tuple

# Import necessary libraries for ChemBERTa
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Warning: Transformers library not found. ChemBERTa embeddings will not work.")
    AutoTokenizer, AutoModel = None, None

# Import RDKit for SMILES validation
try:
    from rdkit import Chem
except ImportError:
    print("Warning: RDKit not found. SMILES validation will be skipped.")
    Chem = None


class YieldRegressionDataset:
    """
    Dataset class for reaction yield regression using ChemBERTa embeddings.
    
    This dataset extracts ChemBERTa embeddings from all reaction components
    (reactants, products, reagents, catalysts, solvents, additives) and uses
    reaction yield as the target variable.
    
    Args:
        data_path (str): Path to CSV file containing reaction data
        embedding_cache_dir (str): Directory to cache computed embeddings
        device (str or torch.device): Device for ChemBERTa computations
        max_length (int): Maximum sequence length for tokenization
        validate_smiles (bool): Whether to validate SMILES strings
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        data_path: str,
        embedding_cache_dir: str = "./cache_embeddings",
        device: str = "cuda",
        max_length: int = 512,
        validate_smiles: bool = True,
        verbose: bool = True
    ):
        self.data_path = data_path
        self.embedding_cache_dir = embedding_cache_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.validate_smiles = validate_smiles
        self.verbose = verbose
        
        # Initialize ChemBERTa
        self.tokenizer = None
        self.model = None
        self.embedding_dim = None
        
        # Data storage
        self.data = None
        self.embeddings = None
        self.yields = None
        self.smiles_columns = [
            'startingmat_1_smiles', 'startingmat_2_smiles', 'product_1_smiles',
            'reagent_1_smiles', 'catalyst_smiles', 'solvent_1_smiles', 
            'additives_smiles_merged'
        ]
        self.yield_column = 'product_1_area%'
        
        # Create cache directory
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        
        # Initialize
        self._initialize_chemberta()
        self._load_data()
        self._load_or_compute_embeddings()
    
    def _initialize_chemberta(self):
        """Initialize ChemBERTa model and tokenizer."""
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("Transformers library is required for ChemBERTa embeddings")
        
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        
        if self.verbose:
            print(f"Initializing ChemBERTa model: {model_name}")
        
        try:
            from utils.bootstrap import ensure_chemberta_safetensors
            ensure_chemberta_safetensors(model_name)
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
    
    def _load_data(self):
        """Load reaction data from CSV file."""
        if self.verbose:
            print(f"Loading data from: {self.data_path}")
        
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Check required columns
            missing_cols = []
            for col in self.smiles_columns + [self.yield_column]:
                if col not in self.data.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
            
            # Clean yield data
            self.data[self.yield_column] = pd.to_numeric(
                self.data[self.yield_column], errors='coerce'
            )
            
            # Remove rows with invalid yields
            initial_count = len(self.data)
            self.data = self.data.dropna(subset=[self.yield_column])
            final_count = len(self.data)
            
            if self.verbose:
                print(f"Loaded {final_count} reactions ({initial_count - final_count} removed due to invalid yields)")
                print(f"Yield range: {self.data[self.yield_column].min():.3f} - {self.data[self.yield_column].max():.3f}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
    
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
    
    def _get_embedding_from_smiles(self, smiles: str) -> torch.Tensor:
        """
        Generate ChemBERTa embedding for a single SMILES string.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            torch.Tensor: ChemBERTa embedding of shape [embedding_dim]
        """
        # Handle empty/invalid SMILES
        if pd.isna(smiles) or smiles == '' or smiles == 'NoAdditive':
            return torch.zeros(self.embedding_dim, device='cpu')
        
        # Validate SMILES if requested
        if not self._validate_smiles(smiles):
            if self.verbose:
                print(f"Warning: Invalid SMILES: {smiles}")
            return torch.zeros(self.embedding_dim, device='cpu')
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                smiles,
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
            
            return embedding.cpu()
            
        except Exception as e:
            if self.verbose:
                print(f"Error generating embedding for SMILES '{smiles}': {e}")
            return torch.zeros(self.embedding_dim, device='cpu')
    
    def _get_embeddings_for_reaction(self, row_idx: int) -> torch.Tensor:
        """
        Generate concatenated embeddings for all components of a reaction.
        
        Args:
            row_idx (int): Index of the reaction in the dataframe
            
        Returns:
            torch.Tensor: Concatenated embeddings of shape [len(smiles_columns) * embedding_dim]
        """
        row = self.data.iloc[row_idx]
        embeddings = []
        
        for col in self.smiles_columns:
            smiles = row.get(col, '')
            embedding = self._get_embedding_from_smiles(smiles)
            embeddings.append(embedding)
        
        # Concatenate all embeddings
        return torch.cat(embeddings, dim=0)
    
    def _get_cache_path(self) -> str:
        """Get path for cached embeddings."""
        data_basename = os.path.basename(self.data_path).replace('.csv', '')
        return os.path.join(self.embedding_cache_dir, f"{data_basename}_embeddings.pkl")
    
    def _load_or_compute_embeddings(self):
        """Load cached embeddings or compute them if not available."""
        cache_path = self._get_cache_path()
        
        # Try to load cached embeddings
        if os.path.exists(cache_path):
            if self.verbose:
                print(f"Loading cached embeddings from: {cache_path}")
            
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pkl.load(f)
                    
                # Verify cache is valid for current data
                if (cached_data['data_shape'] == self.data.shape and 
                    cached_data['smiles_columns'] == self.smiles_columns):
                    
                    self.embeddings = cached_data['embeddings']
                    self.yields = cached_data['yields']
                    
                    if self.verbose:
                        print(f"Loaded {len(self.embeddings)} cached embeddings")
                    return
                else:
                    if self.verbose:
                        print("Cache invalid, recomputing embeddings...")
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache: {e}, recomputing embeddings...")
        
        # Compute embeddings
        if self.verbose:
            print("Computing ChemBERTa embeddings for all reactions...")
        
        embeddings = []
        yields = []
        
        for idx in tqdm(range(len(self.data)), desc="Computing embeddings"):
            embedding = self._get_embeddings_for_reaction(idx)
            yield_value = self.data.iloc[idx][self.yield_column]
            
            embeddings.append(embedding.numpy())
            yields.append(yield_value)
        
        self.embeddings = np.array(embeddings)
        self.yields = np.array(yields)
        
        # Cache the results
        cache_data = {
            'embeddings': self.embeddings,
            'yields': self.yields,
            'data_shape': self.data.shape,
            'smiles_columns': self.smiles_columns,
            'embedding_dim': self.embedding_dim
        }
        
        with open(cache_path, 'wb') as f:
            pkl.dump(cache_data, f)
        
        if self.verbose:
            print(f"Computed and cached {len(self.embeddings)} embeddings")
            print(f"Embedding shape: {self.embeddings.shape}")
            print(f"Yields shape: {self.yields.shape}")
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the processed embeddings and yields.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (embeddings, yields)
        """
        return self.embeddings, self.yields
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for the embeddings."""
        feature_names = []
        for col in self.smiles_columns:
            for i in range(self.embedding_dim):
                feature_names.append(f"{col}_emb_{i}")
        return feature_names
    
    def get_data_info(self) -> Dict:
        """Get information about the dataset."""
        return {
            'n_reactions': len(self.data),
            'n_features': self.embeddings.shape[1],
            'embedding_dim': self.embedding_dim,
            'n_components': len(self.smiles_columns),
            'yield_stats': {
                'mean': np.mean(self.yields),
                'std': np.std(self.yields),
                'min': np.min(self.yields),
                'max': np.max(self.yields),
                'median': np.median(self.yields)
            }
        }
    
    def clear_cache(self):
        """Clear cached embeddings."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
            if self.verbose:
                print(f"Cleared cache: {cache_path}")


def split_data(
    embeddings: np.ndarray, 
    yields: np.ndarray, 
    test_size: float = 0.2, 
    val_size: float = 0.1, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets.
    
    Args:
        embeddings (np.ndarray): Feature embeddings
        yields (np.ndarray): Target yields
        test_size (float): Fraction for test set
        val_size (float): Fraction for validation set
        random_state (int): Random seed
        
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        embeddings, yields, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the reduced dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test 