import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
RDLogger.DisableLog('rdApp.*') 

import torch
from dgl import graph

import warnings
warnings.filterwarnings("ignore")

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    
atom_list = ['Ag','Al','As','B','Bi','Br','C','Cl','Co','Cr','Cu','F','Ge','H','I','In','K','Li','Mg','Mo','N','Na','O','P','Pd','S','Sb','Se','Si','Sn','Te','Zn']
charge_list = [-1, 0, 1, 2]
degree_list = [1, 2, 3, 4, 5, 6, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 0]
valence_list = [1, 2, 3, 4, 5, 6, 7, 12, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]
bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

node_dim = len(atom_list) + len(charge_list) + len(degree_list) + len(hybridization_list) + len(hydrogen_list) + len(valence_list) + len(ringsize_list) - 1
edge_dim = len(bond_list) + 4


def mol_to_graph(mol, config):
    
    def _DA(mol):
        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list

    def _chirality(atom):
        return [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] if atom.HasProp('Chirality') else [0, 0]
        
    def _stereochemistry(bond):
        return [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] if bond.HasProp('Stereochemistry') else [0, 0]    

    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2
    
    D_list, A_list = _DA(mol)  
    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    
    node_attr = np.hstack([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10])

    mol_emb_dict = { "atom_emb": {"xtb": None, "sasa": None}, "mol_emb": {"xtb": None, "sasa": None} }

    from utils.utils_morfeus import get_SEC_embeddings
    SEC_atom_fea_xtb, SEC_mol_fea_xtb, SEC_atom_fea_sasa, SEC_mol_fea_sasa= get_SEC_embeddings(mol, config)

    if not config["addh"]:
        #If I just want to consider heavy atoms, I will need to cut 
        #the descriptors for the H atoms (that will always be in the end)
        n_heavy_atoms = node_attr.shape[0]
        SEC_atom_fea_xtb = SEC_atom_fea_xtb[:n_heavy_atoms,:]
        SEC_atom_fea_sasa = SEC_atom_fea_sasa[:n_heavy_atoms,:]

    mol_emb_dict["atom_emb"]["xtb"] = SEC_atom_fea_xtb
    mol_emb_dict["atom_emb"]["sasa"] = SEC_atom_fea_sasa
    mol_emb_dict["mol_emb"]["xtb"] = SEC_mol_fea_xtb
    mol_emb_dict["mol_emb"]["sasa"] = SEC_mol_fea_sasa
        

    if n_edge > 0:
        bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
        bond_fea2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
        bond_fea3 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)

        edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
        edge_attr = np.vstack([edge_attr, edge_attr])
        
        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])

    else:
        edge_attr = np.empty((0, edge_dim)).astype(bool)
        src = np.empty(0).astype(int)
        dst = np.empty(0).astype(int)

    g = graph((src, dst), num_nodes = n_node)
    g.ndata['node_attr'] = torch.from_numpy(node_attr).bool()
    g.edata['edge_attr'] = torch.from_numpy(edge_attr).bool()

    return g, mol_emb_dict

def dummy_graph():

    g = graph(([], []), num_nodes = 1)
    g.ndata['node_attr'] = torch.from_numpy(np.empty((1, node_dim))).bool()
    g.edata['edge_attr'] = torch.from_numpy(np.empty((0, edge_dim))).bool()

    SEC_mol_fea = None
    
    return g, SEC_mol_fea



def get_graph_data(data, keys, config):

    rmol_graphs = [[] for _ in range(2)]
    pmol_graphs = [[] for _ in range(1)]
    reaction_dict = {'y': [], 'rsmi': []}     

    #Initialize the dictionary to store the embeddings for each reaction

    print(f"INFO: Using the following atom embeddings - {config["atom_emb_list"]} ")
    embeddings_dict = {i: [] for i in range(len(keys))}

    for i, rsmi in enumerate(tqdm(keys)):
    
        [reactants_smi, products_smi] = rsmi.split('>>')
        ys = data[rsmi]

        #Initialize the dictionary to store the embeddings for each reaction
        embeddings_dict[i] = [None, None, None]

        # processing reactants
        reactants_smi_list = reactants_smi.split('.')
        for j, smi in enumerate(reactants_smi_list):
            if smi == '':
                raise ValueError(f"Reactant SMILES is empty for reaction {rsmi}")
            else:
                ############## FIND WAYS TO MAKE RDKit mol object ###############
                rmol = Chem.MolFromSmiles(smi, sanitize=False) #Do not sanitize during mol creation.
                if rmol is None:
                    raise ValueError(f"Couldn't create RDKit mol for SMILES: {smi}")
                    continue

                try:
                    Chem.SanitizeMol(rmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=False) #Try to sanitize
                except Exception as e:
                    print(f"Standard sanitization failed for SMILES: {smi}. Error: {e}")
                    print(f"Doing partial sanitization ignoring valence")
                    try:
                        rmol.UpdatePropertyCache(strict=False)
                        Chem.SanitizeMol(rmol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=False)
                    except Exception as e:
                        raise ValueError(f"Failed also partial sanitization with error {e}")
                        continue

                try:
                    # Chem.DetectBondStereoChemistry(rmol, -1) Gamini says that this is usually applied for 3D graph info that we don't have here
                    Chem.AssignStereochemistry(rmol, flagPossibleStereoCenters=True, force=True)
                    Chem.AssignAtomChiralTagsFromStructure(rmol, -1)
                except Exception as e:
                    raise ValueError(f"Stereochemistry assignment failed for SMILES: {smi}. Error: {e}")
                if rmol == None:
                    raise ValueError(f"Couldn't create RDKit mol for SMILES: {smi}")

                #################################################################
                rs = Chem.FindPotentialStereo(rmol)
                for element in rs:
                    if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': rmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                    elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': rmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))

                # --- Wrap the final step ---
                try:
                    if config["addh"]:
                        rmol = Chem.AddHs(rmol) 
                    else:
                        rmol = Chem.RemoveHs(rmol)

                    graph_obj, emb_info = mol_to_graph(rmol, config)
                    rmol_graphs[j].append(graph_obj)
                    embeddings_dict[i][j] = emb_info

                except Exception as e_graph:
                    raise ValueError(f"    ERROR during mol_to_graph for j={j}, smi={smi}. Error: {e_graph}")
                # --- End wrap ---
                
        
        smi = products_smi
        if smi == '':
            print("Warning: Appended dummy graph for product")
            raise ValueError(f"Product SMILES is empty for reaction {rsmi}")
        else: 
            ############## FIND WAYS TO MAKE RDKit mol object ###############
            pmol = Chem.MolFromSmiles(smi, sanitize=False) #Do not sanitize during mol creation.
            if pmol is None:
                raise ValueError(f"Couldn't create RDKit mol for SMILES: {smi}. Using dummy graph.")
            try:
                Chem.SanitizeMol(pmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=False) #Try to sanitize
            except Exception as e:
                print(f"Standard sanitization failed for SMILES: {smi}. Error: {e}")
                print(f"Doing partial sanitization ignoring valence")
                try:
                    pmol.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(pmol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=False)
                except Exception as e:
                    raise ValueError(f"Failed also partial sanitization with error {e}")
            try:
                # Chem.DetectBondStereoChemistry(pmol, -1) #See above or ask again gemini. If present raises error wrt C++ siganture
                Chem.AssignStereochemistry(pmol, flagPossibleStereoCenters=True, force=True)
                Chem.AssignAtomChiralTagsFromStructure(pmol, -1)
            except Exception as e:
                raise ValueError(f"Stereochemistry assignment failed for SMILES: {smi}. Error: {e}")
            if pmol == None:
                print("Couldn't create RDKit mol", smi)

            #################################################################
            ps = Chem.FindPotentialStereo(pmol)
            for element in ps:
                if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': pmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': pmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                    
            if config["addh"]:
                pmol = Chem.AddHs(pmol) 
            else:
                pmol = Chem.RemoveHs(pmol)
            
            graph_obj, emb_info = mol_to_graph(pmol, config)
            pmol_graphs[0].append(graph_obj)
            embeddings_dict[i][2] = emb_info # Product is the 3rd element
        
        reaction_dict['y'].append(ys)
        reaction_dict['rsmi'].append(rsmi)


    # --- Start Debug Prints ---
    print("\n")
    print("-" * 50)
    print(f"Processing Summary:")
    print(f"Total reactions processed: {len(keys)}")
    correct_length = len(keys)
    print(f"Expected length for all lists: {correct_length}")
    print("\nList Lengths:")
    all_lengths_match = True  # Initialize the variable
    for j in range(2):
        sublist_len = len(rmol_graphs[j])
        print(f"  rmol_graphs[{j}]: {sublist_len}")
        if sublist_len != correct_length:
            all_lengths_match = False
            print(f"  *** WARNING: Length mismatch for rmol_graphs[{j}] ***")
    
    sublist_len = len(pmol_graphs[0])
    print(f"  pmol_graphs[0]: {sublist_len}")
    if sublist_len != correct_length:
        print(f"  *** WARNING: Length mismatch for pmol_graphs[0] ***")
    
    print(f"\nReaction Dictionary:")
    print(f"  Labels (y): {len(reaction_dict['y'])}")
    print(f"  SMILES (rsmi): {len(reaction_dict['rsmi'])}")
    
    print(f"\nEmbeddings:")
    print(f"  Molecular embeddings: {len(embeddings_dict)}")
    
    print(f"\nAll lists have correct length: {all_lengths_match}")
    print("-" * 50)
    print("\n")
    # --- End Debug Prints ---

    rmol_graphs = list(map(list, zip(*rmol_graphs)))
    pmol_graphs = list(map(list, zip(*pmol_graphs))) 

    if config.get("save_embeddings", False):
        emb_dir = "./data_embeddings"
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
            print(f"Created directory: {emb_dir}")
        
        filename = f"{emb_dir}/embeddings_dict_{config['rtype']}_{config['data_type']}.pkl"
        with open(filename, 'wb') as f:
            pkl.dump(embeddings_dict, f)
        print(f"Saved molecular embeddings to: {filename}")

    if config.get("save_graphs", False):
        graph_dir = "./data_graphs"
        # Create the directory if it doesn't exist
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
            print(f"Created directory: {graph_dir}") # Optional: for confirmation

        # Construct the appropriate filename based on reaction type, data type
        filename = f"{graph_dir}/graph_data_{config['rtype']}_{config['data_type']}.pkl"
        
        with open(filename, 'wb') as f:
            pkl.dump([rmol_graphs, pmol_graphs, reaction_dict['y'], reaction_dict['rsmi']], f)
        print(f"Saved data to: {filename}") # Optional: for confirmation
    
    final_rmol_graphs, final_pmol_graphs, final_reaction_dict_y, final_reaction_dict_rsmi, final_embeddings_dict = load_graph_data(config)
    #Return data needed 
    return final_rmol_graphs, final_pmol_graphs, final_reaction_dict_y, final_reaction_dict_rsmi, final_embeddings_dict


def load_graph_data(config):
    graph_dir = "./data_graphs"
    filename = f"{graph_dir}/graph_data_{config['rtype']}_{config['data_type']}.pkl"
    with open(filename, 'rb') as f:
        rmol_graphs, pmol_graphs, reaction_dict_y, reaction_dict_rsmi = pkl.load(f)

    emb_dir = "./data_embeddings"
    filename = f"{emb_dir}/embeddings_dict_{config['rtype']}_{config['data_type']}.pkl"
    with open(filename, 'rb') as f:
        embeddings_dict = pkl.load(f)

    embeddings_mol = [[None, None, None] for _ in range(len(reaction_dict_y))]

    if config["model_type"] in ["emb", "seq_emb"]:

        #Normalize the values of sasa
        if "sasa" in config["atom_emb_list"]:
            print("Normalizing sasa values")
            atom_sasa_mean = torch.mean(torch.cat([embeddings_dict[i][j]["atom_emb"]["sasa"] for i in range(len(embeddings_dict)) for j in range(3)]))
            atom_sasa_std = torch.std(torch.cat([embeddings_dict[i][j]["atom_emb"]["sasa"] for i in range(len(embeddings_dict)) for j in range(3)]))
            mol_sasa_mean = torch.mean(torch.cat([embeddings_dict[i][j]["mol_emb"]["sasa"] for i in range(len(embeddings_dict)) for j in range(3)]))
            mol_sasa_std = torch.std(torch.cat([embeddings_dict[i][j]["mol_emb"]["sasa"] for i in range(len(embeddings_dict)) for j in range(3)]))
            for i in range(len(embeddings_dict)):
                for j in range(3):
                    embeddings_dict[i][j]["atom_emb"]["sasa"] = (embeddings_dict[i][j]["atom_emb"]["sasa"] - atom_sasa_mean) / (atom_sasa_std + 1e-6)
                    embeddings_dict[i][j]["mol_emb"]["sasa"] = (embeddings_dict[i][j]["mol_emb"]["sasa"] - mol_sasa_mean) / (mol_sasa_std + 1e-6)
        
        if (len(config["atom_emb_list"]) > 0) and ("sec" in config["emb_to_use"]):

            for i in range(len(rmol_graphs)):
                for j in range(len(rmol_graphs[i])):
                    if config["atom_emb_list"] == ["xtb"]:
                        rmol_graphs[i][j].ndata['node_attr'] = torch.cat([rmol_graphs[i][j].ndata['node_attr'].float(), embeddings_dict[i][j]["atom_emb"]["xtb"].float()], dim=1)
                    elif config["atom_emb_list"] == ["sasa"]:
                        rmol_graphs[i][j].ndata['node_attr'] = torch.cat([rmol_graphs[i][j].ndata['node_attr'].float(), embeddings_dict[i][j]["atom_emb"]["sasa"].float()], dim=1)
                    elif config["atom_emb_list"] == ["xtb", "sasa"] or config["atom_emb_list"] == ["sasa", "xtb"]:
                        rmol_graphs[i][j].ndata['node_attr'] = torch.cat([rmol_graphs[i][j].ndata['node_attr'].float(), embeddings_dict[i][j]["atom_emb"]["xtb"].float(), embeddings_dict[i][j]["atom_emb"]["sasa"].float()], dim=1)

            for i in range(len(pmol_graphs)):
                for j in range(len(pmol_graphs[i])):
                    if config["atom_emb_list"] == ["xtb"]:
                        pmol_graphs[i][j].ndata['node_attr'] = torch.cat([pmol_graphs[i][j].ndata['node_attr'].float(), embeddings_dict[i][2]["atom_emb"]["xtb"].float()], dim=1)
                    elif config["atom_emb_list"] == ["sasa"]:
                        pmol_graphs[i][j].ndata['node_attr'] = torch.cat([pmol_graphs[i][j].ndata['node_attr'].float(), embeddings_dict[i][2]["atom_emb"]["sasa"].float()], dim=1)
                    elif config["atom_emb_list"] == ["xtb", "sasa"] or config["atom_emb_list"] == ["sasa", "xtb"]:
                        pmol_graphs[i][j].ndata['node_attr'] = torch.cat([pmol_graphs[i][j].ndata['node_attr'].float(), embeddings_dict[i][2]["atom_emb"]["xtb"].float(), embeddings_dict[i][2]["atom_emb"]["sasa"].float()], dim=1)

            for i in range(len(embeddings_dict)):
                for j in range(3):
                    if embeddings_dict[i][j] is not None:
                        if config["atom_emb_list"] == ["xtb"]:
                            embeddings_mol[i][j] = embeddings_dict[i][j]["mol_emb"]["xtb"]
                        elif config["atom_emb_list"] == ["sasa"]:
                            embeddings_mol[i][j] = embeddings_dict[i][j]["mol_emb"]["sasa"]
                        elif config["atom_emb_list"] == ["xtb", "sasa"] or config["atom_emb_list"] == ["sasa", "xtb"]:
                            embeddings_mol[i][j] = torch.cat([embeddings_dict[i][j]["mol_emb"]["xtb"], embeddings_dict[i][j]["mol_emb"]["sasa"]], dim=1)

    return rmol_graphs, pmol_graphs, reaction_dict_y, reaction_dict_rsmi, embeddings_mol