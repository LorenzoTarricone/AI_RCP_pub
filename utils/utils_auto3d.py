import torch
import os
import ase
from ase.build import molecule
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
import Auto3D
# Attempt to import optional dependencies
try:
    import torchani
    # Assuming ANI2xt is a local or installable module
    # from .ANI2xt_no_rep import ANI2xt 
    class ANI2xt(torch.nn.Module): # Placeholder if real import fails
        def __init__(self, device):
            super().__init__()
            print("Warning: Using a placeholder for ANI2xt. Please ensure the actual module is available.")
            self.model = torch.nn.Identity()
        def __call__(self, species, coords):
            return torch.zeros(len(species), device=coords.device)
except ImportError:
    print("Warning: torchani is not installed. ANI models will not be available.")
    torchani = None
    ANI2xt = None


# --- Conversion & Padding Helpers for ASE/Numpy ---

def mols2lists(mols: List[ase.Atoms], model_name: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Converts a list of ase.Atoms objects to lists of coordinates, atomic numbers, and charges.
    This version is designed to work with ASE Atoms objects and produce numpy arrays.
    
    Args:
        mols (List[ase.Atoms]): A list of ASE Atoms objects.
        model_name (str): The name of the model being used (e.g., "ANI2xt").

    Returns:
        A tuple containing:
        - A list of coordinate arrays (numpy.ndarray).
        - A list of atomic number arrays (numpy.ndarray).
        - A list of charges (int).
    """
    coords = [mol.get_positions().astype(np.float32) for mol in mols]
    charges = [int(mol.get_initial_charges().sum()) for mol in mols]
    
    if model_name == "ANI2xt":
        # Mapping from atomic number to ANI2xt species index
        ani2xt_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}
        numbers = [np.array([ani2xt_index[z] for z in mol.get_atomic_numbers()], dtype=np.int64) for mol in mols]
    else:
        numbers = [mol.get_atomic_numbers().astype(np.int64) for mol in mols]
        
    return coords, numbers, charges


def padding_coords(coords: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """
    Pads a list of coordinate arrays into a single numpy array.
    
    Args:
        coords (List[np.ndarray]): List of coordinate arrays (N, 3).
    
    Returns:
        A single padded numpy array (B, max_N, 3).
    """
    max_len = max(len(c) for c in coords)
    B = len(coords)
    padded_coords = np.full((B, max_len, 3), pad_value, dtype=np.float32)
    for i, c in enumerate(coords):
        padded_coords[i, :len(c), :] = c
    return padded_coords


def padding_species(species: List[np.ndarray], pad_value: int = -1) -> np.ndarray:
    """
    Pads a list of atomic number arrays into a single numpy array.

    Args:
        species (List[np.ndarray]): List of atomic number arrays (N,).
    
    Returns:
        A single padded numpy array (B, max_N).
    """
    max_len = max(len(s) for s in species)
    B = len(species)
    padded_species = np.full((B, max_len), pad_value, dtype=np.int64)
    for i, s in enumerate(species):
        padded_species[i, :len(s)] = s
    return padded_species


# --- Core Optimization Logic (from Auto3D) ---

@torch.jit.script
class FIRE():
    """a general optimization program"""
    def __init__(self, coord):
        self.dt_max = 0.1
        self.Nmin = 5
        self.maxstep = 0.1
        self.finc = 1.5
        self.fdec = 0.7
        self.astart = 0.1
        self.fa = 0.99
        self.v = torch.zeros_like(coord)
        self.Nsteps = torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        self.dt = torch.full(coord.shape[:1], 0.1, device=coord.device)
        self.a = torch.full(coord.shape[:1], 0.1, device=coord.device)

    def __call__(self, coord, forces):
        vf = (forces * self.v).flatten(-2, -1).sum(-1)
        w_vf = vf > 0.0
        if w_vf.all():
            a = self.a.unsqueeze(-1).unsqueeze(-1)
            v = self.v
            f = forces
            self.v = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(
                -1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(
                -1)
            self.Nsteps += 1
        elif w_vf.any():
            a = self.a[w_vf].unsqueeze(-1).unsqueeze(-1)
            v = self.v[w_vf]
            f = forces[w_vf]
            self.v[w_vf] = (1.0 - a) * v + a * v.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(
                -1).unsqueeze(-1) * f / f.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(
                -1)

            w_N = self.Nsteps > self.Nmin
            w_vfN = w_vf & w_N
            self.dt[w_vfN] = (self.dt[w_vfN] * self.finc).clamp(max=self.dt_max)
            self.a[w_vfN] *= self.fa
            self.Nsteps[w_vfN] += 1

        w_vf = ~w_vf
        if w_vf.all():
            self.v[:] = 0.0
            self.a[:] = torch.tensor(self.astart, device=self.a.device)
            self.dt[:] *= self.fdec
            self.Nsteps[:] = 0
        elif w_vf.any():
            self.v[w_vf] = torch.tensor(0.0, device=self.v.device)
            self.a[w_vf] = torch.tensor(self.astart, device=self.a.device)
            self.dt[w_vf] *= self.fdec
            self.Nsteps[w_vf] = torch.tensor(0, device=self.v.device)

        dt = self.dt.unsqueeze(-1).unsqueeze(-1)
        self.v += dt * forces
        dr = dt * self.v
        normdr = dr.flatten(-2, -1).norm(p=2, dim=-1).unsqueeze(-1).unsqueeze(-1)
        dr *= (self.maxstep / normdr).clamp(max=1.0)
        return coord + dr

    def clean(self, mask: torch.Tensor) -> bool:
        self.v = self.v[mask]
        self.Nsteps = self.Nsteps[mask]
        self.dt = self.dt[mask]
        self.a = self.a[mask]
        return True


class EnForce_ANI(torch.nn.Module):
    """Wrapper for models to compute energy and forces."""
    def __init__(self, ani, name, batchsize_atoms=1024 * 16):
        super().__init__()
        self.add_module('ani', ani)
        self.name = name
        self.batchsize_atoms = batchsize_atoms
        self.hartree2ev = 27.211386024359793

    def forward(self, coord, numbers, charges):
        if self.name == "AIMNET":
            d = self.ani(dict(coord=coord, numbers=numbers, charge=charges))
            e = d['energy'].to(torch.double)
            f = d['forces']
        elif self.name == "ANI2xt":
            e = self.ani(numbers, coord)
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g if g is not None else torch.zeros_like(coord)
        elif self.name == "ANI2x":
            e = self.ani((numbers, coord)).energies * self.hartree2ev
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g if g is not None else torch.zeros_like(coord)
        else: # user-provided NNP
            e = self.ani(numbers, coord, charges)
            g = torch.autograd.grad([e.sum()], [coord])[0]
            f = -g if g is not None else torch.zeros_like(coord)
        return e, f

    def forward_batched(self, coord, numbers, charges):
        B, N = coord.shape[:2]
        if N == 0: # Handle empty molecules
            return torch.empty(0, device=coord.device), torch.empty(0, 0, 3, device=coord.device)
        
        e, f = [], []
        idx = torch.arange(B, device=coord.device)
        batch_size = self.batchsize_atoms // N if N > 0 else B
        if batch_size == 0: batch_size = B

        for batch in idx.split(batch_size):
            _e, _f = self(coord[batch], numbers[batch], charges[batch])
            e.append(_e)
            f.append(_f)
        return torch.cat(e, dim=0), torch.cat(f, dim=0)


def print_stats(state, patience, config=None):
    """Print the optimization status"""
    if config and not config.get('verbose_sec', False):
        return
        
    num_total = state['numbers'].size(0)
    num_converged_dropped = torch.sum(state['converged_mask']).item()
    oscillating_count = state['oscilating_count'] >= patience
    num_dropped = torch.sum(oscillating_count).item()
    num_converged = num_converged_dropped - num_dropped
    num_active = num_total - num_converged_dropped
    print(f"Total: {num_total} | Converged: {num_converged} | Dropped: {num_dropped} | Active: {num_active}", flush=True)


def n_steps(state, n, opttol, patience, config=None):
    """Performs n optimization steps."""
    coord = state['coord']
    optimizer = FIRE(coord)
    state["oscilating_count"] = torch.zeros(len(coord), 1, dtype=torch.float, device=coord.device)
    smallest_fmax0 = torch.full((len(coord), 1), 999.0, dtype=torch.float, device=coord.device)

    # Conditionally show progress bar based on verbose_sec setting
    if config and config.get('verbose_sec', False):
        iterator = tqdm(range(1, n + 1), desc="Optimizing")
    else:
        iterator = range(1, n + 1)

    for istep in iterator:
        not_converged = ~state['converged_mask']
        if not not_converged.any():
            if config and config.get('verbose_sec', False):
                print(f"\nOptimization finished early at step {istep-1}.")
            break

        current_coord = state['coord'][not_converged]
        current_numbers = state['numbers'][not_converged]
        current_charges = state['charges'][not_converged]
        
        current_coord.requires_grad_(True)
        e, f = state['nn'].forward_batched(current_coord, current_numbers, current_charges)
        current_coord.requires_grad_(False)

        new_coord = optimizer(current_coord, f)
        fmax = f.norm(dim=-1).max(dim=-1)[0]
        
        # Update state for active molecules
        state['coord'][not_converged] = new_coord
        state['energy'][not_converged] = e.detach()
        state['fmax'][not_converged] = fmax
        
        # Convergence and oscillation check
        newly_converged = fmax <= opttol
        oscilating_count = state["oscilating_count"][not_converged]
        smallest_fmax = smallest_fmax0[not_converged]

        fmax_reduced = (fmax.reshape(-1, 1) < smallest_fmax)
        oscilating_count[fmax_reduced] = 0
        oscilating_count[~fmax_reduced] += 1
        
        smallest_fmax[fmax_reduced] = fmax.reshape(-1, 1)[fmax_reduced]
        smallest_fmax0[not_converged] = smallest_fmax
        state["oscilating_count"][not_converged] = oscilating_count
        
        not_oscillating = (oscilating_count < patience).flatten()
        
        # Update overall converged mask
        temp_mask = torch.clone(state['converged_mask'][not_converged])
        temp_mask[newly_converged & not_oscillating] = True
        temp_mask[~not_oscillating] = True # Mark oscillating molecules as "converged" to drop them
        state['converged_mask'][not_converged] = temp_mask
        
        optimizer.clean(mask=~temp_mask)
        
        if (istep % (n // 10 if n > 10 else 1)) == 0:
            print_stats(state, patience, config)
            
    else: # This else belongs to the for loop, executes if loop finishes without break
        if config and config.get('verbose_sec', False):
            print("\nReached maximum optimization steps.")
    print_stats(state, patience, config)


def ensemble_opt(net, coord, numbers, charges, param, device):
    """Main entry point for batch optimization."""
    coord_tensor = torch.tensor(coord, dtype=torch.float32, device=device)
    numbers_tensor = torch.tensor(numbers, dtype=torch.int64, device=device)
    charges_tensor = torch.tensor(charges, dtype=torch.long, device=device)
    
    state = {
        'ids': torch.arange(coord_tensor.shape[0], device=device),
        'coord': coord_tensor,
        'numbers': numbers_tensor,
        'charges': charges_tensor,
        'converged_mask': torch.zeros(coord_tensor.shape[0], dtype=torch.bool, device=device),
        'fmax': torch.full(coord_tensor.shape[:1], 999.0, device=device),
        'energy': torch.full(coord_tensor.shape[:1], 999.0, dtype=torch.double, device=device),
        'nn': net,
    }
    
    n_steps(state, param['opt_steps'], param['opttol'], param['patience'], param)

    return {k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v 
            for k, v in state.items() if k != 'nn'}


class Optimizing:
    """
    A class to optimize the geometry of molecules using a neural network potential.
    This version is adapted to work with a list of ase.Atoms objects.
    """
    def __init__(self, model_name: str, device: str, config: Dict):
        self.model_name = model_name
        self.device = device
        self.config = config
        
        # Construct the path to the 'models' directory
        model_dir =  os.path.join(Auto3D.__path__[0], "models")

        os.makedirs(model_dir, exist_ok=True)
        if config.get('verbose_sec', False):
            print(f"Initializing optimizer with model: {self.model_name} on device: {self.device}")

        # --- Model Loading Logic ---
        if self.model_name == "AIMNET":
            model_path = os.path.join(model_dir, "aimnet2_wb97m_ens_f.jpt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"AIMNET model not found at {model_path}")
            self.model = torch.jit.load(model_path, map_location=device)
        elif self.model_name == "ANI2xt":
            if ANI2xt is None:
                raise ImportError("ANI2xt model requested but not available.")
            self.model = ANI2xt(device)
        elif self.model_name == "ANI2x":
            if torchani is None:
                raise ImportError("torchani not installed, cannot use ANI2x model.")
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        elif os.path.exists(self.model_name):
            if config.get('verbose_sec', False):
                print(f"Loading model from path: {self.model_name}")
            self.model = torch.jit.load(self.model_name, map_location=device)
        else:
            raise ValueError(f"Model '{self.model_name}' not recognized.")

    def run(self, ase_atoms_list: List[ase.Atoms]) -> List[ase.Atoms]:
        if not all(isinstance(item, ase.Atoms) for item in ase_atoms_list):
            raise TypeError("ase_atoms_list must be a list of ase.Atoms objects.")
        
        original_atoms_list = [atoms.copy() for atoms in ase_atoms_list]
        if self.config.get('verbose_sec', False):
            print(f"Preparing for optimization... (Max steps: {self.config.get('opt_steps', 'N/A')})")
            print(f"Total conformers to optimize: {len(original_atoms_list)}")

        coords, numbers, charges = mols2lists(original_atoms_list, self.model_name)
        coord_padded = padding_coords(coords)
        numbers_padded = padding_species(numbers)
        
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        enforce_model = EnForce_ANI(self.model, self.model_name, self.config.get("batchsize_atoms", 5120))
        
        with torch.jit.optimized_execution(False):
            optdict = ensemble_opt(enforce_model, coord_padded, numbers_padded, charges, self.config, self.device)

        # Process results
        energies = optdict['energy']
        fmax_values = optdict['fmax']
        optimized_coords = np.array(optdict['coord'])
        convergence_mask = np.array(fmax_values) <= self.config.get('opttol', 0.001)

        optimized_atoms_list = []
        for i, original_atoms in enumerate(original_atoms_list):
            num_atoms = len(original_atoms)
            result_atoms = original_atoms.copy()
            opt_coords_i = optimized_coords[i, :num_atoms, :]
            result_atoms.set_positions(opt_coords_i)
            
            result_atoms.info['E_tot'] = energies[i]
            result_atoms.info['fmax'] = fmax_values[i]
            result_atoms.info['Converged'] = bool(convergence_mask[i])
            result_atoms.info['ID'] = original_atoms.info.get('ID', f'conf_{i}')
            optimized_atoms_list.append(result_atoms)

        if self.config.get('verbose_sec', False):
            print("Optimization complete.")
        return optimized_atoms_list


def optimize_ase_atoms_geometry(atoms_list: list[ase.Atoms], model_name="AIMNET", gpu_idx=0, opt_tol=0.003, opt_steps=1000, patience=100, config=None):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")

    opt_config = {"opt_steps": opt_steps, "opttol": opt_tol, "patience": patience, "batchsize_atoms": 16384}
    
    # Merge config if provided
    if config:
        opt_config.update(config)
    
    try:
        opt_engine = Optimizing(model_name, device, opt_config)
        optimized_molecules = opt_engine.run(atoms_list)
        energies = [mol.info['E_tot'] for mol in optimized_molecules]
        return optimized_molecules, energies
    except (FileNotFoundError, ValueError, ImportError) as e:
        if config and config.get('verbose_sec', False):
            print(f"Error: {e}")
        return atoms_list, [np.nan] * len(atoms_list)

