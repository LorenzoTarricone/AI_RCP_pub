import os, sys
import numpy as np
import torch
from dgl import graph
import pickle as pkl
import warnings
from tqdm import tqdm
import re

#To assess running time and block printings
import time
from wurlitzer import pipes

# Import necessary libraries for ChemBERTa (if used)
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Warning: Transformers library not found. 'use_embedd=True' will not work.")
    AutoTokenizer, AutoModel = None, None

# Import RDKit (needed for rxnfp and parsing SMILES for ChemBERTa)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Warning: RDKit not found. 'use_rxnfp=True' or 'use_embedd=True' may not work.")
    Chem, AllChem = None, None


class GraphDataset():
    """A PyTorch dataset class for handling molecular reaction graphs and their associated data.
    
    This dataset class is designed to work with molecular reaction data, supporting both graph-based
    and embedding-based representations. It can handle various types of molecular embeddings including
    ChemBERTa and Morfeus embeddings.

    Args:
        rmol_graphs (list): List of reactant molecular graphs (DGLGraph objects)
        pmol_graphs (list): List of product molecular graphs (DGLGraph objects)
        labels (list): List of reaction condition labels
        smiles (list): List of reaction SMILES strings
        mol_emb (list, optional): List of molecular embeddings. Required for 'emb' and 'seq_emb' model types
        config (dict): Configuration dictionary containing model and data parameters
        split (str, optional): Dataset split type ('trn', 'val', or 'tst'). Defaults to 'tst'
        device (torch.device, optional): Device to use for computations. Defaults to None

    Example:
        >>> config = {
        ...     "rtype": "bh",
        ...     "use_rxnfp": False,
        ...     "emb_to_use": ["cb", "sec"],
        ...     "expand_data": True
        ... }
        >>> dataset = GraphDataset(rmol_graphs, pmol_graphs, labels, smiles, mol_emb, config)
    """

    def __init__(self, rmol_graphs, pmol_graphs, labels, smiles, mol_emb, config , split = 'tst', device=None):
        """Initialize the GraphDataset with molecular reaction data.
        
        Args:
            rmol_graphs (list): List of reactant molecular graphs
            pmol_graphs (list): List of product molecular graphs
            labels (list): List of reaction condition labels
            smiles (list): List of reaction SMILES strings
            mol_emb (list): List of molecular embeddings
            config (dict): Configuration dictionary
            split (str): Dataset split type ('trn', 'val', or 'tst')
            device (torch.device): Device to use for computations
        """
        assert split in ['trn', 'val', 'tst']

        if (config["emb_to_use"] != None) and (AutoTokenizer is None or AutoModel is None):
            raise ImportError("Transformers library is required when emb_to_use != None.")
        if (config["use_rxnfp"] or (config["emb_to_use"] != None)) and Chem is None:
                raise ImportError("RDKit library is required when use_rxnfp=True or emb_to_use != None.")
        
        self.category = config["rtype"]
        self.split = split
        self.use_rxnfp = config["use_rxnfp"]
        self.seed = config["random_seed"]
        self.expand_data = config["expand_data"]
        self.emb_to_use = config["emb_to_use"]
        self.clist = config["clist"]

        self.n_info = config.get("n_info", None)
        self.verbose = config.get("verbose", False)

        self.rmol_graphs = rmol_graphs
        self.pmol_graphs = pmol_graphs
        self.y = labels
        self.rsmi = smiles
        self.config = config

        if self.config["emb_to_use"] != None:


            if "cb" in self.emb_to_use:
                # ChemBERTa related attributes (initialized if use_embedd is True)
                self.tokenizer = None
                self.chemberta_model = None
                self.chemberta_embedding_dim = None

                # Store the target device for ChemBERTa computations
                self.device = device
                if self.device:
                    if self.verbose:
                        print("\n")
                        print(f"Attempting to use device: {self.device} for generating ChemBERTa embeddings.")

                self._initialize_chemberta()

            if "sec" in self.emb_to_use:
                assert mol_emb is not None

        self.mol_emb = mol_emb

        # Assert mol_emb is not None for both 'emb' and 'seq_emb' model types
        if config["model_type"] in ["emb", "seq_emb"] and "sec" in self.emb_to_use:
            assert mol_emb is not None, f"SEC derived embeddings are required for the '{config['model_type']}' model type"

        self.load()

    def _initialize_chemberta(self):
        """Initialize the ChemBERTa model and tokenizer for molecular embeddings.
        
        This method loads the ChemBERTa model and tokenizer, moves them to the specified device,
        and freezes the model parameters. It handles error cases gracefully and provides
        appropriate warnings if initialization fails.

        Raises:
            RuntimeError: If model/tokenizer loading fails and use_embedd is True
        """
        chemberta_model_name = "seyonec/ChemBERTa-zinc-base-v1"
        if self.verbose: print(f"Initializing ChemBERTa model ({chemberta_model_name}) in Dataset...")
        # Load tokenizer and model onto CPU for safety with DataLoader workers
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(chemberta_model_name)
            self.chemberta_model = AutoModel.from_pretrained(chemberta_model_name)
            self.chemberta_embedding_dim = self.chemberta_model.config.hidden_size

            # Freeze ChemBERTa parameters & set to eval mode
            for param in self.chemberta_model.parameters():
                param.requires_grad = False
            self.chemberta_model.eval()

            # --- Move model to the specified device ---
            if self.device and self.chemberta_model:
                try:
                    self.chemberta_model.to(self.device)
                    if self.verbose: print(f"ChemBERTa model moved to {self.device}.")
                except Exception as e:
                    warnings.warn(f"Failed to move ChemBERTa model to device {self.device}. Using CPU instead. Error: {e}")
                    self.device = torch.device("cpu") # Fallback to CPU
                    self.chemberta_model.to(self.device) # Ensure it's on CPU
            elif self.chemberta_model:
                    # If no device specified, ensure it's on CPU
                    self.device = torch.device("cpu")
                    self.chemberta_model.to(self.device)
                    if self.verbose: print("ChemBERTa model loaded on CPU.")
            # -----------------------------------------
            if self.verbose: print("ChemBERTa model loaded and frozen.")

        except Exception as e:
            warnings.warn(f"Failed to load ChemBERTa model '{chemberta_model_name}'. "
                          f"Setting use_embedd=False. Error: {e}")
            print(f"Error: {e}")
            self.emb_to_use.remove("cb") # Fallback if model loading fails
            self.tokenizer = None
            self.chemberta_model = None
            self.chemberta_embedding_dim = None


    def load(self, frac_val = 0.1):
        """Load and process the dataset.
        
        This method handles data processing, graph expansion, and embedding generation.
        The data splitting is assumed to have been done *before* this class is initialized.
        
        Args:
            frac_val (float, optional): This argument is no longer used but is kept for
                                       compatibility with any calling code. Defaults to 0.1.
        """
        clist = self.clist

        # ================================ FIX STARTS HERE =================================
        #
        # The entire `if self.split in ['trn', 'val']:` block that re-split the data
        # has been removed. The method now proceeds directly with the data that was
        # passed to the `__init__` method.
        #
        # ================================= FIX ENDS HERE ==================================

        self.n_classes = len(clist)
        # Handle cases where the dataset might be empty
        if len(self.rmol_graphs) == 0:
            warnings.warn(f"Dataset for split '{self.split}' is empty. Further processing will be skipped.")
            self.cnt_list = []
            self.n_reactions = 0
            self.n_conditions = 0
            # Set dummy values for dimensions to avoid errors, though they won't be used
            self.rmol_max_cnt = 0
            self.pmol_max_cnt = 0
            self.node_dim = 0
            self.edge_dim = 0
            return # Exit early
            
        self.rmol_max_cnt = len(self.rmol_graphs[0])
        self.pmol_max_cnt = len(self.pmol_graphs[0])
        self.node_dim = self.rmol_graphs[0][0].ndata['node_attr'].shape[1]
        self.edge_dim = self.rmol_graphs[0][0].edata['edge_attr'].shape[1]
        self.cnt_list = [len(a) for a in self.y] #Number of conditions sets for each reaction
        self.n_reactions = len(self.y)
        self.n_conditions = np.sum(self.cnt_list)
        
        if self.split == 'trn':
            self.rmol_graphs = sum([[self.rmol_graphs[i]] * self.cnt_list[i] for i in range(len(self.y))], [])
            self.pmol_graphs = sum([[self.pmol_graphs[i]] * self.cnt_list[i] for i in range(len(self.y))], [])
            self.rsmi_pre = self.rsmi
            self.rsmi = np.repeat(self.rsmi, self.cnt_list, 0).tolist()
            self.y = sum(self.y, [])

        
        assert len(self.rmol_graphs) == len(self.y), "Mismatch between graph count and label count after processing."
        assert len(self.rsmi) == len(self.y), "Mismatch between SMILES count and label count after processing."
        
        if self.use_rxnfp:
            self._get_rnx_fp()


        if self.config["emb_to_use"] != None:
          
            print(f"Generating embeddings...")

            if ("cb" in self.emb_to_use) and (not self.chemberta_model or not self.tokenizer):
                raise RuntimeError("use_embedd is True, but ChemBERTa model/tokenizer failed to load.")
            
            rmsi_to_use = self.rsmi if self.split != 'trn' else self.rsmi_pre
            
            print(f"Starting embeddings generation for {self.split} split")

            all_emb = []

            assert len(self.mol_emb) == len(rmsi_to_use), f"You have {len(self.mol_emb)} embeddigs and you are processing {len(rmsi_to_use)} reaction smiles"

            for i, rs in enumerate(tqdm(rmsi_to_use)):
                emb_lists = []
                try:
                    rsmiles_str, psmiles_str = rs.split('>>')
                    # Assuming SMILES are dot-separated for multiple molecules
                    rsmiles_list = [s for s in rsmiles_str.split('.') if s] # Filter empty strings
                    psmiles_list = [s for s in psmiles_str.split('.') if s] # Filter empty strings
                except ValueError:
                    warnings.warn(f"Could not parse reaction SMILES '{rs}'. Skipping embedding generation.")
                    # Create zero embeddings if parsing fails
                    emb_lists.append(torch.zeros(self.chemberta_embedding_dim))
                    emb_lists.append(torch.zeros(self.chemberta_embedding_dim))

                if "cb" in self.emb_to_use:
                    emb_lists.append(self._get_chemberta_embeddings(rsmiles_list))
                    emb_lists.append(self._get_chemberta_embeddings(psmiles_list))
                if "sec" in self.emb_to_use:
                    reaction_mol_embs = self.mol_emb[i]
                    emb_lists.append(reaction_mol_embs[0].to(self.config["device"]))
                    emb_lists.append(reaction_mol_embs[1].to(self.config["device"]))
                    emb_lists.append(reaction_mol_embs[2].to(self.config["device"]))

                
                combined_embedding = torch.cat(emb_lists, dim=1).float() # Ensure float
                all_emb.append(combined_embedding)

            self.all_emb = torch.cat(all_emb, dim=0)
            self.emb_dim = self.all_emb.shape[1]

            if self.split == 'trn':
                # Ensure self.all_emb is a tensor (it should be from the .T operation)
                if not isinstance(self.all_emb, torch.Tensor):
                    # This case would be unusual here but added for safety
                    print("Warning: self.all_emb was not a tensor before repeat_interleave. Converting.")
                    self.all_emb = torch.tensor(self.all_emb, dtype=torch.float32) # Adjust dtype if necessary

                # Ensure self.cnt_list is a 1D PyTorch tensor of integers (long)
                # self.cnt_list should have the same number of elements as there are rows in self.all_emb
                if not isinstance(self.cnt_list, torch.Tensor):
                    cnt_list_tensor = torch.tensor(self.cnt_list, dtype=torch.long, device=self.all_emb.device)
                else:
                    # If it's already a tensor, ensure it's the correct type and on the same device
                    cnt_list_tensor = self.cnt_list.to(dtype=torch.long, device=self.all_emb.device)
                
                # Check if lengths match if self.cnt_list is a tensor of counts
                if cnt_list_tensor.ndim > 0 and self.all_emb.shape[0] != cnt_list_tensor.shape[0]:
                    raise ValueError(f"Dimension mismatch for repeat_interleave: "
                                        f"self.all_emb has {self.all_emb.shape[0]} rows, "
                                        f"but self.cnt_list has {cnt_list_tensor.shape[0]} elements.")

                # Use torch.repeat_interleave instead of torch.repeat
                # Remove .tolist() to keep self.all_emb as a tensor
                self.all_emb = torch.repeat_interleave(self.all_emb, cnt_list_tensor, dim=0)

    def _get_rnx_fp(self):
        """Generate reaction fingerprints using RDKit.
        
        This method computes reaction fingerprints for each reaction in the dataset
        using RDKit's reaction fingerprinting capabilities.

        Note:
            Requires RDKit to be installed and use_rxnfp to be True in config.
        """
        del self.rmol_graphs
        del self.pmol_graphs
    
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        self.fp_dim = 512
        self.rxnfp = []

        # fpgen = AllChem.GetMorganGenerator(radius=2, fpSize = self.fp_dim, includeChirality=True)

        print(f"Generating RXNFP for {self.split} split")
        for rs in tqdm(self.rsmi):
            
            def try_sanitize(smiles):
                try:
                    Chem.SanitizeMol(Chem.MolFromSmiles(smiles, sanitize=False), sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=False)
                except Exception as e:
                    raise ValueError(f"Standard sanitization failed for SMILES: {smiles}. Error: {e}")
            
            rsmiles, psmiles = rs.split('>>')
            sm1_smiles, sm2_smiles = rsmiles.split('.')

            sm1_mol = Chem.MolFromSmiles(sm1_smiles)
            if sm1_mol is None:
                sm1_mol = try_sanitize(sm1_smiles)
            sm2_mol = Chem.MolFromSmiles(sm2_smiles)
            if sm2_mol is None:
                sm2_mol = try_sanitize(sm2_smiles)
            pmol = Chem.MolFromSmiles(psmiles)
            if pmol is None:
                pmol = try_sanitize(psmiles)
            
            sm1_fp = AllChem.GetMorganFingerprintAsBitVect(sm1_mol, radius=2, nBits=self.fp_dim, useFeatures=False, useChirality=True)
            sm2_fp = AllChem.GetMorganFingerprintAsBitVect(sm2_mol, radius=2, nBits=self.fp_dim, useFeatures=False, useChirality=True)
            pmol_fp = AllChem.GetMorganFingerprintAsBitVect(pmol, radius=2, nBits=self.fp_dim, useFeatures=False, useChirality=True)
            sm1_arr = np.zeros((self.fp_dim,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(sm1_fp, sm1_arr)
            sm2_arr = np.zeros((self.fp_dim,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(sm2_fp, sm2_arr)
            pmol_arr = np.zeros((self.fp_dim,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(pmol_fp, pmol_arr)
            rxnfp = np.concatenate([sm1_arr, sm2_arr, pmol_arr])

            self.rxnfp.append(rxnfp)
            
        self.rxnfp = np.array(self.rxnfp)

    def _get_chemberta_embeddings(self, smiles_list):
        """Generate ChemBERTa embeddings for a list of SMILES strings.
        
        Args:
            smiles_list (list): List of SMILES strings to generate embeddings for

        Returns:
            torch.Tensor: Tensor containing the generated embeddings

        Note:
            Requires transformers library and ChemBERTa model to be initialized.
        """
        # Determine the device to use (fallback to CPU if self.device is None)
        target_device = self.device if self.device else torch.device("cpu")

        if "cb" not in self.emb_to_use or not self.chemberta_model or not self.tokenizer:
            # Should not happen if checks in __init__ and __getitem__ are done
            return torch.zeros(self.chemberta_embedding_dim or 768, device=target_device) # Default size if unknown

        if not smiles_list or all(s == '' for s in smiles_list): # Handle empty or list of empty strings
            print("Found empty smile string")
            return torch.zeros(self.chemberta_embedding_dim, device = target_device)

        # Filter out any empty strings from the list before tokenizing
        valid_smiles = [s for s in smiles_list if s]
        if not valid_smiles:
                return torch.zeros(self.chemberta_embedding_dim, device=target_device)

        try:
            # Tokenize the valid SMILES strings
            inputs = self.tokenizer(valid_smiles, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            
            # Get model outputs (inference on CPU or specified device)
            with torch.no_grad():
                outputs = self.chemberta_model(**inputs)

            # Extract CLS token embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :] # Shape: [num_valid_smiles, embedding_dim]

            #normalize the embeddings to get vectors with norm 1
            cls_embeddings = cls_embeddings / torch.norm(cls_embeddings, dim=1, keepdim=True)

            # Aggregate embeddings with concatenation to pass from shape [num_valid_smiles, embedding_dim] to [embedding_dim * num_valid_smiles]
            aggregated_embedding = cls_embeddings.flatten().unsqueeze(0) # Shape: [1,embedding_dim * num_valid_smiles]

            return aggregated_embedding

        except Exception as e:
            warnings.warn(f"Error during ChemBERTa embedding generation for SMILES {valid_smiles}: {e}")
            return torch.zeros(self.chemberta_embedding_dim) # Return zeros on error


    def __getitem__(self, idx):
        """Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple: A tuple containing:
                - Reactant graphs (DGLGraph or list of DGLGraph)
                - Product graphs (DGLGraph or list of DGLGraph)
                - Labels (torch.Tensor or list)
                - SMILES string (str)
                - Molecular embeddings (torch.Tensor, optional)
                - Positive/negative sample flag (torch.Tensor, optional)
        """

        # --- Step 1: Get Label ---
        if self.split == 'trn':
            # In training, a label is a tuple: (condition_indices, pos_neg_flag)
            label_raw = self.y[idx][0]
            pos_neg_reac_val = self.y[idx][1]
            if pos_neg_reac_val is None:
                 raise ValueError("pos_neg_ flag is None")
            pos_neg_reac = torch.tensor([[pos_neg_reac_val]], dtype=torch.float32)
        else:
            # For val/tst, the pos_neg flag is not part of the label data.
            # It's created during inference. Here, we create a placeholder.
            
            label_raw = 0
            pos_neg_reac = torch.tensor([[float('nan')]], dtype=torch.float32)

        if self.split == 'trn':
            if self.expand_data:
                if self.n_info is None:
                        raise ValueError("n_info must be provided when expand_data is True.")
                # --- Label Expansion Logic (copied from original) ---
                label_full = np.zeros(self.n_classes, dtype=bool)
                label_full[label_raw] = 1 # label_raw contains indices for this sample

                (n_cats, n_sol_1, n_sol_2, n_add, n_base) = self.n_info

                if self.category.lower() in ['bh']: # Buchwald-Hartwig
                    off = 0
                    label_cat = label_full[off:off+n_cats]; off += n_cats
                    label_base = label_full[off:off+n_base]; off += n_base
                    label_solv_1 = label_full[off:off+n_sol_1]; off += n_sol_1
                    label_solv_2 = label_full[off:off+n_sol_2]; off += n_sol_2
                    label_water = label_full[off:off+1]; off += 1
                    label_add = label_full[off:off+n_add]; off += n_add
                elif self.category.lower() in ['sm']: # Suzuki-Miyaura
                    off = 0
                    label_solv_1 = label_full[off:off+n_sol_1]; off += n_sol_1
                    label_solv_2 = label_full[off:off+n_sol_2]; off += n_sol_2
                    label_water = label_full[off:off+1]; off += 1
                    label_add = label_full[off:off+n_add]; off += n_add
                    label_cat = label_full[off:off+n_cats]; off += n_cats
                    label_base = label_full[off:off+n_base]; off += n_base
                else:
                        raise ValueError(f"Unsupported category '{self.category}' for expand_data.")

                # Basic validation assertions (optional)
                # assert np.sum(label_cat) == 1, f"Catalyst count error: {np.sum(label_cat)}"
                # assert np.sum(label_base) == 1, f"Base count error: {np.sum(label_base)}"

                label_processed = (label_full, label_cat, label_solv_1, label_add, label_solv_2, label_base, label_water)
                # --- End Label Expansion ---
            else:
                # If not expanding, create the multi-hot vector directly
                label_processed = np.zeros(self.n_classes, dtype=bool)
                label_processed[label_raw] = 1
        else: # val or tst split
            # For val/tst, the raw label is a list of condition indices.
            label_processed = label_raw

    # --- Step 2: Handle Embeddings (if requested) ---
        if isinstance(self.emb_to_use, list) and len(self.emb_to_use) > 0:
            
            assert "cb" in self.emb_to_use or "sec" in self.emb_to_use, "Type of embedding to use not recognised"

            if not hasattr(self, 'rsmi') or idx >= self.all_emb.shape[0]:
                raise IndexError(f"Index {idx} out of bounds or rsmi not loaded for ChemBERTa.")
            
            combined_embedding = self.all_emb[idx]

            # Check if graphs should be returned
            if not self.use_rxnfp and self.rmol_graphs is not None: # Graphs available
                rg = self.rmol_graphs[idx]
                pg = self.pmol_graphs[idx]
                # Ensure float type for graph features (handle None padding)
                for g_list in [rg, pg]:
                    for g in g_list:
                        if g and hasattr(g, 'ndata'): # Check if it's a valid graph object
                            g.ndata['node_attr'] = g.ndata['node_attr'].float()
                            g.edata['edge_attr'] = g.edata['edge_attr'].float()
                return (*rg, *pg, label_processed, pos_neg_reac, combined_embedding)
            else: # Graphs not available (use_rxnfp=True during load or graphs deleted)
                # Return label and embedding
                # Convert label to tensor if it's a numpy array (e.g., non-expanded trn label)
                if isinstance(label_processed, np.ndarray):
                        label_tensor = torch.from_numpy(label_processed).float()
                        return (label_tensor, pos_neg_reac, combined_embedding)
                else:
                        # Return raw label (tuple or list) and embedding, handle in collate_fn
                        return (label_processed, pos_neg_reac, combined_embedding)

        # --- Step 3: Original Logic (if use_embedd is False) ---
        else:
            if self.use_rxnfp:
                # Return RXNFP and label tensor
                if not hasattr(self, 'rxnfp') or idx >= len(self.rxnfp):
                        raise IndexError(f"Index {idx} out of bounds for RXNFP.")
                rxnfp_tensor = torch.FloatTensor(self.rxnfp[idx])
                
                # For val/test, label_processed is zero. For training, it's a multi-hot vector.
                label_tensor = torch.FloatTensor(label_processed)

                return rxnfp_tensor, label_tensor, pos_neg_reac
            else:
                # Return graphs and label
                if self.rmol_graphs is None or idx >= len(self.rmol_graphs):
                        raise IndexError(f"Index {idx} out of bounds or graphs not loaded.")
                rg = self.rmol_graphs[idx]
                pg = self.pmol_graphs[idx]
                # Ensure float type for graph features (handle None padding)
                for g_list in [rg, pg]:
                    for g in g_list:
                        if g and hasattr(g, 'ndata'):
                            g.ndata['node_attr'] = g.ndata['node_attr'].float()
                            g.edata['edge_attr'] = g.edata['edge_attr'].float()
                # Return graphs unpacked and the processed label (numpy, tuple, or list)
                return (*rg, *pg, label_processed, pos_neg_reac)


    def __len__(self):
        """Get the total number of items in the dataset.
        
        Returns: 
            int: The total number of items in the dataset
        """
        return len(self.y)


def get_cardinalities_classes(config):

    # if category == "bh":
    #     if water:
    #         filepath = "./data/Buchwald-Hartwig_treshold_all_withWater_processed.npz"
    #         _, clist = np.load(filepath, allow_pickle = True)['data'][0]
    #     else:
    #         filepath = "./data/Buchwald-Hartwig_treshold_all_noWater_processed.npz"
    #         _, clist = np.load(filepath, allow_pickle = True)['data'][0]
    # elif category == 'sm':
    #     if water:
    #         filepath = "./data/Suzuki-Miyaura_treshold_all_withWater_processed.npz"
    #         _, clist = np.load(filepath, allow_pickle = True)['data'][0]
    #     else:
    #         filepath = "./data/Suzuki-Miyaura_treshold_all_noWater_processed.npz"
    #         _, clist = np.load(filepath, allow_pickle = True)['data'][0]
    # else:
    #     print(f".npz file not found for category {category} and water {water}")
    category = config["rtype"]
    clist = config["clist"]

    # Define the prefixes we are interested in
    if category == 'sm':
        target_prefixes = {'B', 'S', 'L', 'A', 'C'}
        # Regex to match a target prefix followed by one or more digits at the start
        # It captures the prefix (group 1) and the number (group 2)
        pattern = re.compile(r'^([BSLAC])(\d+)')

    else:
        # Define the prefixes we are interested in
        target_prefixes = {'B', 'S', 'A', 'C'}
        # Regex to match a target prefix followed by one or more digits at the start
        # It captures the prefix (group 1) and the number (group 2)
        pattern = re.compile(r'^([BSAC])(\d+)')
    
    # Initialize a dictionary to store the maximum number found for each prefix
    max_numbers = {prefix: 0 for prefix in target_prefixes}

    for item in clist:
        # Ensure item is a string before applying regex
        if not isinstance(item, str):
            continue

        match = pattern.match(item)
        if match:
            prefix = match.group(1)
            number_str = match.group(2)

            try:
                number = int(number_str)
                # Update the maximum if this number is larger
                if number > max_numbers[prefix]:
                    max_numbers[prefix] = number
            except ValueError:
                # This shouldn't happen due to regex \d+, but added for safety
                continue # Skip if conversion fails for some reason

    # if category == 'sm':
    #     n_solv_adds = max_numbers['S'] + max_numbers['L'] + max_numbers['A']
    # else:
    #     n_solv_adds = max_numbers['S'] + max_numbers['A']
    if category == 'sm':
        n_cats = max_numbers['C']
        n_base = max_numbers['B']
        n_sol_1 = max_numbers['S']
        n_add = max_numbers['A']
        n_sol_2 = max_numbers['L']
    else:
        n_cats = max_numbers['C']
        n_base = max_numbers['B']
        n_sol_1 = max_numbers['S']
        n_add = max_numbers['A']
        n_sol_2 = 0

        
    return (n_cats, n_sol_1, n_sol_2, n_add, n_base)