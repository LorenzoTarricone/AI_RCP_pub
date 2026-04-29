import os
import sys

# Ensure the conda environment's bin directory (where xtb lives) is on PATH.
# Derives the bin path from sys.executable so it works regardless of where the
# environment was created — replaces an earlier hardcoded HPC-specific path.
_env_bin = os.path.dirname(sys.executable)
if _env_bin and _env_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_env_bin}{os.pathsep}{os.environ.get('PATH', '')}"

import time
from wurlitzer import pipes #To block prints of the energy minimization engine

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import torch

from utils.utils_auto3d import optimize_ase_atoms_geometry

def _initialize_morfeus(config):
    # morfeus is now installed as the `morfeus-ml` conda package (see
    # environment.yaml); the previous logic that bolted a local `morfeus/`
    # submodule onto sys.path is no longer needed. Kept as a no-op so that
    # call sites in this file remain valid.
    return




def get_SEC_embeddings(mol, config):

    

    verbose_sec = config.get("verbose_sec", False)

    if config["morfeus_init"]:
        #Initialize morfeus importing the functions needed just the first time
        _initialize_morfeus(config)
        config["morfeus_init"] = False
    
    from morfeus.conformer import ConformerEnsemble
    
    smi = Chem.MolToSmiles(mol)

    if verbose_sec:
        print(f"\nProcessing SMILES: {smi}")

    elements = None
    coordinates = None
    num_atoms_for_morfeus = 0
    molecule_processed_successfully = False

    if config["ASE_opt"]:
        try:
            import ase.io
            from ase.io import write
            from ase.optimize import LBFGS
            from aimnet.calculators import AIMNet2ASE
            from ase import Atoms


            def ase_to_morfeus_ensamble(ase_conformers):
                elements = ase_conformers[0].get_atomic_numbers()
                coordinates = [atoms.get_positions() for atoms in ase_conformers]
                morfeus_ensemble = ConformerEnsemble(elements, coordinates)
                mol_copy = Chem.Mol(mol)
                mol_copy = Chem.AddHs(mol_copy)
                morfeus_ensemble.mol = mol_copy
                morfeus_ensemble.update_mol()

                return morfeus_ensemble
            
            def morfeus_to_ase_conformers(morfeus_ensemble):
                ase_conformers = [Atoms(symbols=conf.elements, positions=conf.coordinates) for conf in morfeus_ensemble]
                return ase_conformers
            
            def write_conformers_to_sdf(morfeus_ensemble, sdf_filename):
                morfeus_ensemble.update_mol()
                mol = morfeus_ensemble.mol
                conformers = list(mol.GetConformers())
                writer = Chem.SDWriter(sdf_filename)
                for conf in conformers:
                    # Create a copy of the molecule with only this conformer
                    mol_copy = Chem.Mol(mol)
                    mol_copy.RemoveAllConformers()
                    mol_copy.AddConformer(conf, assignId=True)
                    writer.write(mol_copy)
                writer.close()

            def ase_conformers_from_sdf(sdf_file)-> tuple[list[Atoms], list[float]]:
                """
                Parses the content of an SDF file to extract molecular structures and their
                optimized energies.

                Args:
                    sdf_file: A string containing the entire content of an SDF file.

                Returns:
                    A tuple containing two lists:
                    - The first list holds ase.Atoms objects for each molecule.
                    - The second list holds the corresponding optimized energies as floats.
                """
                atoms_list = []
                energies = []

                # Split the file content into individual molecule blocks
                molecule_blocks = sdf_file.strip().split('$$$$')

                for block in molecule_blocks:
                    if not block.strip():
                        continue

                    lines = block.strip().split('\n')
                    
                    try:
                        # The fourth line contains the number of atoms
                        num_atoms = int(lines[3].strip().split()[0])
                        
                        # Atom information starts from the fifth line
                        atom_lines = lines[4:4 + num_atoms]
                        
                        symbols = []
                        positions = []
                        for line in atom_lines:
                            parts = line.split()
                            positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            symbols.append(parts[3])
                        
                        # Create an ASE Atoms object
                        atoms = Atoms(symbols=symbols, positions=positions)
                        atoms_list.append(atoms)
                        
                        # Find and parse the total energy
                        energy = None
                        for i, line in enumerate(lines):
                            if '>  <E_tot>' in line:
                                energy = float(lines[i + 1].strip())
                                break
                        
                        if energy is not None:
                            energies.append(energy)
                        else:
                            energies.append(float('nan')) # Append NaN if energy is not found

                    except (IndexError, ValueError) as e:
                        print(f"Skipping a malformed block due to an error: {e}")
                        # Ensure lists remain synchronized in case of an error
                        if 'atoms' in locals() and (not atoms_list or atoms is not atoms_list[-1]):
                            atoms_list.append(None) 
                        energies.append(float('nan'))

                return atoms_list, energies

            # ------------------------ Standard ASE optimization ------------------------
            
            if verbose_sec:
                print(f"Attempting conformer generation for {smi}...")
            time_0 = time.time()

            # 1. Generate conformers (with MMFF94 optimization)
            ce = ConformerEnsemble.from_rdkit(mol, optimize="MMFF94")
            if len(ce) == 0:
                if verbose:
                    print(f"No conformers generated for {smi}.")
                return None  # or handle as appropriate
            if verbose_sec:
                print(f"Generated {len(ce)} conformers.")
            
            # 1.1 Prune conformers based on RMSD
            ce.prune_rmsd()
            ce.sort()
            if verbose_sec:
                print(f"Kept {len(ce)} conformers after RMSD pruning.")

            # 2. Convert conformers to ASE Atoms objects
            ase_conformers = morfeus_to_ase_conformers(ce)

            # 3. Compute AIMNet2 single-point energies for all conformers
            if verbose_sec:
                print(f"Computing AIMNet2 single-point energies for {len(ase_conformers)} conformers...")


            # Set the calculator for each atoms object
            for atoms in ase_conformers:
                atoms.calc = AIMNet2ASE()

            # This will compute energies for all conformers in a single batch call internally
            aimnet2_energies = np.array([atoms.get_potential_energy() for atoms in ase_conformers])
            #make them in kcal/mol
            aimnet2_energies = aimnet2_energies * 23.060548
            time_1 = time.time()
            if verbose_sec:
                print(f"AIMNet2 single-point energies took {time_1 - time_0:.2f} seconds.")

            # 4. Prune conformers based on AIMNet2 energies
            min_energy = np.min(aimnet2_energies)
            energy_threshold = config.get("aimnet2_energy_threshold", 7.0)  # kcal/mol
            keep_indices = np.where(aimnet2_energies <= min_energy + energy_threshold)[0]
            ase_conformers = [ase_conformers[i] for i in keep_indices]
            if verbose_sec:
                print(f"Kept {len(ase_conformers)} conformers within {energy_threshold} kcal/mol of the minimum AIMNet2 energy.")

            
            # 4.1 Convert ase conformers to morfeus ensemble
            morfeus_ensemble = ase_to_morfeus_ensamble(ase_conformers)
            morfeus_ensemble.sort()

            # 4.2 Keep only the number of conformers specified in the config
            max_num_conformers = min(config["max_num_conformers"], len(morfeus_ensemble))
            morfeus_ensemble = morfeus_ensemble[:max_num_conformers]
            if verbose_sec:
                print(f"Kept {len(morfeus_ensemble)} conformers after max_num_conformers pruning.")

            # 4.3 Convert morfeus ensemble to ase conformers
            ase_conformers = morfeus_to_ase_conformers(morfeus_ensemble)

            # 5. Geometric optimization 
            # ------- with Auto3D (batched) -------
            if config["Geometry_opt_type"] == "Auto3D":
                if verbose_sec:
                    print(f"Geometric optimization of {len(ase_conformers)} conformers...")
                optimized_ase_conformers, opt_energies = optimize_ase_atoms_geometry(ase_conformers, model_name="AIMNET", gpu_idx=0, config=config)
                time_2 = time.time()
                if verbose_sec:
                    print(f"Geometry optimization with Auto3D took {time_2 - time_1:.2f} seconds.")

            elif config["Geometry_opt_type"] == "Auto3D_OG":
                #Use the original Auto3D code to optimize the conformers
                #write sdf file with the conformers
                from Auto3D.ASE.geometry import opt_geometry
                sdf_file = "input.sdf"
                try:
                    write_conformers_to_sdf(morfeus_ensemble, sdf_file)
                except Exception as e:
                    print(f"Error writing SDF file: {e}")

                optimized = opt_geometry(sdf_file, model_name="AIMNET", opt_tol=0.002)
                optimized_ase_conformers, opt_energies = ase_conformers_from_sdf(optimized)
                time_2 = time.time()
                if verbose_sec:
                    print(f"Geometry optimization with Auto3D_OG took {time_2 - time_1:.2f} seconds.")
                    
            
            elif config["Geometry_opt_type"] == "ASE":
            # ------- with ASE (unbatched) -------
                optimized_ase_conformers = []
                if verbose_sec:
                    print(f"Optimizing {len(ase_conformers)} conformers with AIMNet2...")
                for atoms in ase_conformers:
                    atoms.calc = AIMNet2ASE()
                    opt = LBFGS(atoms, logfile=None)
                    opt.run(fmax=0.01)
                    optimized_ase_conformers.append(atoms)
                opt_energies = np.array([atoms.get_potential_energy() for atoms in optimized_ase_conformers])
                time_2 = time.time()
                if verbose_sec:
                    print(f"Geometry optimization with ASE took {time_2 - time_1:.2f} seconds.")

            else:
                print(f"Error: Invalid geometry optimization type: {config['Geometry_opt_type']}")
                return np.array([]), np.array([])

            # ------------------------------------

            # 6. Select the lowest-energy conformer after optimization
            best_idx = np.argmin(opt_energies)
            best_atoms = optimized_ase_conformers[best_idx]
            elements = best_atoms.get_chemical_symbols()
            coordinates = best_atoms.get_positions()
            num_atoms_for_morfeus = len(elements)
            molecule_processed_successfully = True
            if verbose_sec:
                print(f"Selected lowest-energy conformer after AIMNet2 optimization for {smi}.")
                    
        except Exception as e_ce:
            print(f"ConformerEnsemble method failed for {smi}: {e_ce}. Falling back to RDKit method.")

        if not molecule_processed_successfully:

            # 2. Add Hydrogens
            try:
                mol = Chem.AddHs(mol)
            except Exception as e_addhs:
                print(f"Error adding hydrogens for {smi}: {e_addhs}. Skipping.")
                return np.array([]), np.array([])

            num_atoms_for_morfeus = mol.GetNumAtoms()
            if num_atoms_for_morfeus == 0:
                print(f"Error: Molecule {smi} has 0 atoms after processing. Skipping.")
                return np.array([]), np.array([])

            # 3. Generate 3D Conformers
            params = AllChem.ETKDGv3()
            params.randomSeed = config["random_seed"]
            # Ensure num_conformers is at least 1 if EmbedMultipleConfs is used.
            # However, EmbedMultipleConfs can return 0 conformers.
            num_conformers=50
            actual_num_conformers_to_generate = max(1, num_conformers)
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=actual_num_conformers_to_generate, params=params)

            if not conf_ids: # No conformers generated by EmbedMultipleConfs
                if verbose_sec:
                    print(f"Warning: EmbedMultipleConfs did not return any conformer IDs for {smi}. Trying EmbedMolecule as a fallback.")
                # Ensure params.randomSeed is set for EmbedMolecule as well
                if AllChem.EmbedMolecule(mol, params=params) == -1: # Returns -1 on failure
                    print(f"Error: Could not generate any conformers for {smi} even with fallback EmbedMolecule. Skipping.")
                    return np.array([]), np.array([])
                conf_ids = [0] # EmbedMolecule creates one conformer with ID 0
                if verbose_sec:
                    print(f"Generated 1 initial conformer for {smi} using fallback EmbedMolecule.")
            else:
                if verbose_sec:
                    print(f"Generated {len(conf_ids)} initial conformers for {smi}.")

            # 4. Optimize Conformers (MMFF94s or UFF) and get energies
            results_data = []
            optimization_method_used = None

            # Try MMFF94s first
            try:
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
                if mmff_props is not None:
                    if verbose_sec: print(f"Optimizing conformers with MMFF94s for {smi}...")
                    # MMFFOptimizeMoleculeConfs returns a list of (notConverged, energy) tuples
                    # The index in the list corresponds to the conformer index (which is its ID if they are 0,1,2...)
                    mmff_opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant="MMFF94s", maxIters=500)

                    if mmff_opt_results and len(mmff_opt_results) == mol.GetNumConformers():
                        for i, res_tuple in enumerate(mmff_opt_results):
                            not_converged_flag, energy_val = res_tuple
                            conf = mol.GetConformer(i) # Get conformer by index
                            results_data.append({
                                'id': conf.GetId(), # Store the actual ID
                                'energy': energy_val,
                                'converged': not not_converged_flag # True if not_converged_flag is 0
                            })
                        if any(r['energy'] is not None and r['energy'] != float('inf') for r in results_data):
                            optimization_method_used = "MMFF94s"
                    else:
                        if verbose_sec: print(f"MMFFOptimizeMoleculeConfs returned unexpected results for {smi}. Will try UFF.")
                else:
                    if verbose_sec:
                        print(f"MMFF94s properties not available for {smi} Will try UFF.")
            except Exception as e_mmff_opt:
                print(f"Exception during MMFF94s setup or optimization for {smi}: {e_mmff_opt}. Will try UFF for {smi}.")

            # If MMFF94s was not used or failed to produce valid results, try UFF
            if not optimization_method_used:
                if not AllChem.UFFHasAllMoleculeParams(mol):
                    if verbose_sec:
                        print(f"Warning: Molecule '{smi}' may not have all UFF parameters. Optimization might be unreliable or fail.")
                    # Continue to try optimization anyway, or could skip here

                uff_results_temp = []
                for conf_id in mol.GetConformers(): # Iterate through actual conformer objects
                    actual_conf_id = conf_id.GetId()
                    try:
                        uff_ff = AllChem.UFFGetMoleculeForceField(mol, confId=actual_conf_id)
                        if uff_ff is None:
                            print(f"Warning: Could not initialize UFF for conformer ID {actual_conf_id} of {smi}. Adding with high energy.")
                            uff_results_temp.append({'id': actual_conf_id, 'energy': float('inf'), 'converged': False})
                            continue
                        
                        uff_ff.Initialize() # Good practice
                        not_converged_flag = uff_ff.Minimize(maxIts=500) # 0 if converged, 1 if not
                        energy = uff_ff.CalcEnergy()
                        uff_results_temp.append({'id': actual_conf_id, 'energy': energy, 'converged': not not_converged_flag})
                        if verbose_sec:
                            print(f"  Conformer {actual_conf_id}: UFF Energy = {energy:.2f} kcal/mol (Converged: {not not_converged_flag})")
                    except Exception as e_uff_min:
                        print(f"  Error during UFF minimization for conformer {actual_conf_id} of {smi}: {e_uff_min}. Adding with high energy.")
                        uff_results_temp.append({'id': actual_conf_id, 'energy': float('inf'), 'converged': False})
                
                if any(r['energy'] is not None and r['energy'] != float('inf') for r in uff_results_temp):
                    results_data = uff_results_temp # UFF successful, use its results
                    optimization_method_used = "UFF"
                else:
                    if verbose_sec: print(f"UFF optimization also failed to produce valid results for {smi}.")


            if not optimization_method_used or not results_data:
                print(f"Error: Could not optimize conformers for {smi} with any method. Skipping.")
                return np.array([]), np.array([])

            # Filter out conformers that didn't yield a valid energy or failed optimization
            valid_energy_results = [res for res in results_data if res['energy'] is not None and res['energy'] != float('inf')]
            if not valid_energy_results:
                print(f"Error: No conformers yielded valid energies after {optimization_method_used} optimization for {smi}. Skipping.")
                return np.array([]), np.array([])

            # Prioritize converged conformers
            converged_results = [res for res in valid_energy_results if res['converged']]
            if verbose_sec:
                print(f"\n{optimization_method_used} TOP Optimization Results for {smi}:")
                for res in results_data[:5]: # Log all attempted results for diagnostics
                    print(f"  Conformer ID {res['id']}: Energy = {res.get('energy', float('nan')):.2f} kcal/mol (Converged: {res.get('converged', False)})")

            results_to_consider_for_min = converged_results
            if not converged_results:
                if verbose_sec:
                    print(f"Warning: No conformers fully converged with {optimization_method_used} for {smi}. Using lowest energy from all (possibly non-converged) attempts.")
                results_to_consider_for_min = valid_energy_results # Fallback to non-converged if no converged ones

            if not results_to_consider_for_min:
                print(f"Error: No suitable conformers found after {optimization_method_used} optimization for {smi}. Skipping.")
                return np.array([]), np.array([])

            # 5. Identify Lowest Energy Conformer from the chosen method
            lowest_energy_result = min(results_to_consider_for_min, key=lambda x: x['energy'])
            best_conf_id = lowest_energy_result['id']
            lowest_energy = lowest_energy_result['energy']

            if verbose_sec:
                print(f"\nLowest energy {optimization_method_used} conformer ID: {best_conf_id} with Energy: {lowest_energy:.2f} kcal/mol")

            best_conformer_obj = mol.GetConformer(best_conf_id) # Get by actual ID

            # 6. Get elements and coordinates from the best conformer
            elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
            coordinates = best_conformer_obj.GetPositions()

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    else: #Do enrgy minimization with RDKit
        # 2. Add Hydrogens
        try:
            mol = Chem.AddHs(mol)
        except Exception as e_addhs:
            print(f"Error adding hydrogens for {smi}: {e_addhs}. Skipping.")
            return np.array([]), np.array([])

        num_atoms_for_morfeus = mol.GetNumAtoms()
        if num_atoms_for_morfeus == 0:
            print(f"Error: Molecule {smi} has 0 atoms after processing. Skipping.")
            return np.array([]), np.array([])

        # 3. Generate 3D Conformers
        params = AllChem.ETKDGv3()
        params.randomSeed = config["random_seed"]
        # Ensure num_conformers is at least 1 if EmbedMultipleConfs is used.
        # However, EmbedMultipleConfs can return 0 conformers.
        num_conformers=50
        actual_num_conformers_to_generate = max(1, num_conformers)
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=actual_num_conformers_to_generate, params=params)

        if not conf_ids: # No conformers generated by EmbedMultipleConfs
            if verbose:
                print(f"Warning: EmbedMultipleConfs did not return any conformer IDs for {smi}. Trying EmbedMolecule as a fallback.")
            # Ensure params.randomSeed is set for EmbedMolecule as well
            if AllChem.EmbedMolecule(mol, params=params) == -1: # Returns -1 on failure
                print(f"Error: Could not generate any conformers for {smi} even with fallback EmbedMolecule. Skipping.")
                return np.array([]), np.array([])
            conf_ids = [0] # EmbedMolecule creates one conformer with ID 0
            if verbose:
                print(f"Generated 1 initial conformer for {smi} using fallback EmbedMolecule.")
        else:
            if verbose:
                print(f"Generated {len(conf_ids)} initial conformers for {smi}.")

        # 4. Optimize Conformers (MMFF94s or UFF) and get energies
        results_data = []
        optimization_method_used = None

        # Try MMFF94s first
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if mmff_props is not None:
                if verbose: print(f"Optimizing conformers with MMFF94s for {smi}...")
                # MMFFOptimizeMoleculeConfs returns a list of (notConverged, energy) tuples
                # The index in the list corresponds to the conformer index (which is its ID if they are 0,1,2...)
                mmff_opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant="MMFF94s", maxIters=500)

                if mmff_opt_results and len(mmff_opt_results) == mol.GetNumConformers():
                    for i, res_tuple in enumerate(mmff_opt_results):
                        not_converged_flag, energy_val = res_tuple
                        conf = mol.GetConformer(i) # Get conformer by index
                        results_data.append({
                            'id': conf.GetId(), # Store the actual ID
                            'energy': energy_val,
                            'converged': not not_converged_flag # True if not_converged_flag is 0
                        })
                    if any(r['energy'] is not None and r['energy'] != float('inf') for r in results_data):
                        optimization_method_used = "MMFF94s"
                else:
                    if verbose: print(f"MMFFOptimizeMoleculeConfs returned unexpected results for {smi}. Will try UFF.")
            else:
                print(f"MMFF94s properties not available for {smi} Will try UFF.")
        except Exception as e_mmff_opt:
            print(f"Exception during MMFF94s setup or optimization for {smi}: {e_mmff_opt}. Will try UFF for {smi}.")

        # If MMFF94s was not used or failed to produce valid results, try UFF
        if not optimization_method_used:
            if not AllChem.UFFHasAllMoleculeParams(mol):
                print(f"Warning: Molecule '{smi}' may not have all UFF parameters. Optimization might be unreliable or fail.")
                # Continue to try optimization anyway, or could skip here

            uff_results_temp = []
            for conf_id in mol.GetConformers(): # Iterate through actual conformer objects
                actual_conf_id = conf_id.GetId()
                try:
                    uff_ff = AllChem.UFFGetMoleculeForceField(mol, confId=actual_conf_id)
                    if uff_ff is None:
                        print(f"Warning: Could not initialize UFF for conformer ID {actual_conf_id} of {smi}. Adding with high energy.")
                        uff_results_temp.append({'id': actual_conf_id, 'energy': float('inf'), 'converged': False})
                        continue
                    
                    uff_ff.Initialize() # Good practice
                    not_converged_flag = uff_ff.Minimize(maxIts=500) # 0 if converged, 1 if not
                    energy = uff_ff.CalcEnergy()
                    uff_results_temp.append({'id': actual_conf_id, 'energy': energy, 'converged': not not_converged_flag})
                    if verbose:
                        print(f"  Conformer {actual_conf_id}: UFF Energy = {energy:.2f} kcal/mol (Converged: {not not_converged_flag})")
                except Exception as e_uff_min:
                    print(f"  Error during UFF minimization for conformer {actual_conf_id} of {smi}: {e_uff_min}. Adding with high energy.")
                    uff_results_temp.append({'id': actual_conf_id, 'energy': float('inf'), 'converged': False})
            
            if any(r['energy'] is not None and r['energy'] != float('inf') for r in uff_results_temp):
                results_data = uff_results_temp # UFF successful, use its results
                optimization_method_used = "UFF"
            else:
                if verbose: print(f"UFF optimization also failed to produce valid results for {smi}.")


        if not optimization_method_used or not results_data:
            print(f"Error: Could not optimize conformers for {smi} with any method. Skipping.")
            return np.array([]), np.array([])

        # Filter out conformers that didn't yield a valid energy or failed optimization
        valid_energy_results = [res for res in results_data if res['energy'] is not None and res['energy'] != float('inf')]
        if not valid_energy_results:
            print(f"Error: No conformers yielded valid energies after {optimization_method_used} optimization for {smi}. Skipping.")
            return np.array([]), np.array([])

        # Prioritize converged conformers
        converged_results = [res for res in valid_energy_results if res['converged']]
        # if verbose:
        #     print(f"\n{optimization_method_used} TOP Optimization Results for {smi}:")
        #     for res in results_data[:5]: # Log all attempted results for diagnostics
        #         print(f"  Conformer ID {res['id']}: Energy = {res.get('energy', float('nan')):.2f} kcal/mol (Converged: {res.get('converged', False)})")

        results_to_consider_for_min = converged_results
        if not converged_results:
            if verbose:
                print(f"Warning: No conformers fully converged with {optimization_method_used} for {smi}. Using lowest energy from all (possibly non-converged) attempts.")
            results_to_consider_for_min = valid_energy_results # Fallback to non-converged if no converged ones

        if not results_to_consider_for_min:
            print(f"Error: No suitable conformers found after {optimization_method_used} optimization for {smi}. Skipping.")
            return np.array([]), np.array([])

        # 5. Identify Lowest Energy Conformer from the chosen method
        lowest_energy_result = min(results_to_consider_for_min, key=lambda x: x['energy'])
        best_conf_id = lowest_energy_result['id']
        lowest_energy = lowest_energy_result['energy']

        if verbose:
            print(f"\nLowest energy {optimization_method_used} conformer ID: {best_conf_id} with Energy: {lowest_energy:.2f} kcal/mol")

        best_conformer_obj = mol.GetConformer(best_conf_id) # Get by actual ID

        # 6. Get elements and coordinates from the best conformer
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = best_conformer_obj.GetPositions()

    atom_embeddings_xtb = []
    atom_embeddings_sasa = []
    mol_embeddings_xtb = []
    mol_embeddings_sasa = []
    
    try:
        if 'xtb' in config["atom_emb_list"]:
            from morfeus.xtb import XTB
            if num_atoms_for_morfeus < 2 and 'sterimol' in config["atom_emb_list"]: # bond_order requires at least 2 atoms for default indices
                print(f"Warning: XTB bond_order requires at least 2 atoms for default indices (1,2) for {smi}. Skipping XTB bond order.")
            xtb_calculator = XTB(elements, coordinates)

            #Molecular properties
            mol_embeddings_xtb.append(xtb_calculator.get_ip(corrected=True))
            mol_embeddings_xtb.append(xtb_calculator.get_ea())
            mol_embeddings_xtb.append(xtb_calculator.get_homo())
            mol_embeddings_xtb.extend(xtb_calculator.get_dipole())
            mol_embeddings_xtb.append(xtb_calculator.get_global_descriptor("electrophilicity", corrected=True))
            mol_embeddings_xtb.append(xtb_calculator.get_global_descriptor("nucleophilicity", corrected=True))

            #TODO Atom mapping here for atom and edges electornic descriptors
            #Atomic properties
            fuki_nuc_dict = xtb_calculator.get_fukui("nucleophilicity")
            fuki_ele_dict = xtb_calculator.get_fukui("electrophilicity")
            charges_dict = xtb_calculator.get_charges()

            #if the atoms are in the correct order this should be fine
            fuki_nuc_array = np.array([fuki_nuc_dict[i] for i in sorted(fuki_nuc_dict.keys())])
            fuki_ele_array = np.array([fuki_ele_dict[i] for i in sorted(fuki_ele_dict.keys())])
            charges_array = np.array([charges_dict[i] for i in sorted(charges_dict.keys())])

            atom_embeddings_xtb.append(fuki_nuc_array)
            atom_embeddings_xtb.append(fuki_ele_array)
            atom_embeddings_xtb.append(charges_array)

            # #Edge properties
            # if num_atoms_for_morfeus >= 2 : # Check if default bond_order(1,2) is valid
            #         edge_embeddings.append(xtb_calculator.get_bond_order(1, 2)) # atom indices are 1-based for morfeus
            # else: # Handle single atom molecules or provide NaN/placeholder
            #     edge_embeddings.append(float('nan')) # Or some other placeholder for bond order
            

        #TODO: atom mapping and then understnd what are the residues we want to include and if we want to inclide this
        #      as a molecular level information jsut for the important spots or for all the edges? 
        # if 'sterimol' in config["atom_emb_list"]:
        #     from morfeus.sterimol import Sterimol
        #     if num_atoms_for_morfeus < 2:
        #             print(f"Warning: Sterimol requires at least 2 atoms for default indices (1,2) for {smi}. Appending NaNs for Sterimol.")
        #             edge_embeddings.extend([float('nan')] * 4) # L, B5, B1, bond_length
        #     else:
        #         # Atom indices for morfeus.sterimol are 1-based.
        #         # Ensure atom1_idx and atom2_idx are valid for the molecule.
        #         # Defaulting to 1 and 2. For robustness, one might need to define these more carefully.
        #         atom1_idx, atom2_idx = 1, 2
        #         if num_atoms_for_morfeus < max(atom1_idx, atom2_idx):
        #             print(f"Warning: Not enough atoms ({num_atoms_for_morfeus}) for Sterimol indices {atom1_idx}, {atom2_idx} in {smi}. Appending NaNs.")
        #             edge_embeddings.extend([float('nan')] * 4)
        #         else:
        #             sterimol_calculator = Sterimol(elements, coordinates, atom1_idx, atom2_idx)
        #             edge_embeddings.append(sterimol_calculator.L_value)
        #             edge_embeddings.append(sterimol_calculator.B_5_value)
        #             edge_embeddings.append(sterimol_calculator.B_1_value)
        #             edge_embeddings.append(sterimol_calculator.bond_length)

        if 'sasa' in config["atom_emb_list"]:
            from morfeus.sasa import SASA
            sasa = SASA(elements, coordinates)
            #Molecular properties
            mol_embeddings_sasa.append(sasa.area)
            mol_embeddings_sasa.append(sasa.volume)

            #Atomic properties
            sasa_array = np.array([sasa.atom_areas[atom_n] for atom_n in range(1,num_atoms_for_morfeus+1)], dtype=np.float64)
            atom_embeddings_sasa.append(sasa_array)
    
    except Exception as e_morfeus:
        print(f"Error during features calclation calculation for {smi}: {e_morfeus}. Skipping.")
        return np.array([]), np.array([]), np.array([])

    return torch.tensor(np.vstack(atom_embeddings_xtb, dtype=np.float64)).T, torch.tensor(np.vstack(mol_embeddings_xtb, dtype=np.float64)).T, torch.tensor(np.vstack(atom_embeddings_sasa, dtype=np.float64)).T, torch.tensor(np.vstack(mol_embeddings_sasa, dtype=np.float64)).T