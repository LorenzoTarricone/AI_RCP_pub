# Constraint-aware plate design: compares the ILP formulation against the naive
# frequency-ranking heuristic under the positive-coverage constraint, and reports each 96-well
# plate as the four disjoint categories (positive / negative / uncertain / unknown). This is the
# script behind Table 3 of the paper; it writes constrained_ilp_comparison_results_new.csv.
#
# It does not train anything: pass a generative model with --gen_model_path together with the
# config it was trained under. Any model trained from this repository works; see the README.
#
# Must be set before any imports to avoid macOS segfault caused by
# OpenMP thread contention between numpy and PyTorch model loading.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import logging
import pandas as pd
import json
import random
from collections import defaultdict, Counter
import pulp
import itertools
import wandb

# Imports from project
from utils.create_graphs import get_graph_data, load_graph_data
from utils.collate_functions import collate_reaction_graphs, collate_graphs_and_embeddings
from utils.dataset import GraphDataset, get_cardinalities_classes
# VAE models
from models.model_VAE_seq_emb import VAE_seq_emb as VAE_seq_emb_model
from models.model_VAE_seq_emb import Trainer as Trainer_seq_emb

def flatten_wandb_config(config):
    """Recursively flattens a wandb config dictionary."""
    flat_config = {}
    for key, value in config.items():
        if key.startswith('_'):
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
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    if not verbose:
        RDLogger.DisableLog('rdApp.warning')
        RDLogger.DisableLog('rdApp.error')

    return logging.getLogger(__name__)

def get_model_and_trainer(model_type, config):
    if model_type == 'seq_emb':
        config["use_rxnfp"] = False
        config["collate_fn"] = collate_graphs_and_embeddings
        config["expand_data"] = True
        return VAE_seq_emb_model, Trainer_seq_emb, collate_graphs_and_embeddings
    else:
        raise ValueError(f"This script is simplified to only support 'seq_emb', but got '{model_type}'")

def generate_conditions_from_model(trainer, tst_loader, config, n_conditions):
    """Generate conditions using the provided generative model."""
    if config["data_type"] == "positive":
        tst_y_preds_pos, _ = trainer.inference(tst_loader, n_sampling=n_conditions, temperature=config.get("temperature", 1.0))
        tst_y_preds_neg = []
    else:
        tst_y_preds_pos, tst_y_preds_neg = trainer.inference(tst_loader, n_sampling=n_conditions, temperature=config.get("temperature", 1.0))

    smiles_list = tst_loader.dataset.rsmi
    generated_conditions = defaultdict(lambda: {'positive': [], 'negative': []})

    for i, smiles in enumerate(smiles_list):
        if tst_y_preds_pos:
            # Convert to tuples for hashability
            generated_conditions[smiles]['positive'] = [tuple(sorted(cond)) for cond in tst_y_preds_pos[i]]
        if tst_y_preds_neg:
            generated_conditions[smiles]['negative'] = [tuple(sorted(cond)) for cond in tst_y_preds_neg[i]]

    return generated_conditions

# ---------------------------------------------------------------------------
# New functions for plate-level pooling and 4-category scoring
# ---------------------------------------------------------------------------

def pool_conditions_to_plate_level(conditions, class_map):
    """Project raw condition tuples to (catalyst, (solvent, base)) plate-well level.

    Discards non-plate components (water, additives) and deduplicates so that
    each unique (catalyst, solvent-base pair) counts only once.
    """
    pooled = set()
    for cond in conditions:
        cat = next(c for c in cond if class_map.get(c) == 'C')
        sol = next(s for s in cond if class_map.get(s) == 'S')
        bas = next(b for b in cond if class_map.get(b) == 'B')
        pooled.add((cat, (sol, bas)))
    return pooled

def categorize_pooled_conditions(pooled_positive, pooled_negative):
    """Split pooled conditions into pure positive, pure negative, and uncertain.

    Uncertain = conditions appearing in both positive and negative sets.
    """
    uncertain = pooled_positive & pooled_negative
    pure_positive = pooled_positive - uncertain
    pure_negative = pooled_negative - uncertain
    return pure_positive, pure_negative, uncertain

def score_plate_4category(plate_catalyst_indices, plate_sb_pair_indices,
                          pure_positive, pure_negative, uncertain,
                          logger, plate_name):
    """Classify each of the 96 plate wells into 4 categories.

    Returns (positive_count, negative_count, uncertain_count, unknown_count)
    where the four values sum to 96.
    """
    pos, neg, unc, unk = 0, 0, 0, 0
    for cat in plate_catalyst_indices:
        for sb in plate_sb_pair_indices:
            pair = (cat, sb)
            if pair in uncertain:
                unc += 1
            elif pair in pure_positive:
                pos += 1
            elif pair in pure_negative:
                neg += 1
            else:
                unk += 1

    total = pos + neg + unc + unk
    assert total == 96, f"Expected 96 wells, got {total}"

    logger.info(f"\n--- {plate_name} Plate (4-category) ---")
    logger.info(f"  Positive:  {pos}  ({100*pos/96:.1f}%)")
    logger.info(f"  Negative:  {neg}  ({100*neg/96:.1f}%)")
    logger.info(f"  Uncertain: {unc}  ({100*unc/96:.1f}%)")
    logger.info(f"  Unknown:   {unk}  ({100*unk/96:.1f}%)")

    return pos, neg, unc, unk

# ---------------------------------------------------------------------------
# Modified naive approach — operates on pooled positive conditions
# ---------------------------------------------------------------------------

def naive_plate_design_pooled(pooled_positive, config, logger):
    """Construct a naive plate design from pooled positive (cat, (sol, base)) pairs.

    Each unique pair contributes one count to its catalyst and one to its S-B pair.
    Top-12 catalysts and top-8 S-B pairs are selected by frequency.
    """
    reagents_df = pd.read_csv(f"./reagents_dfs/{config['rtype']}_treshold_all_all_reagent_df.csv")
    class_map = {i: cls for i, cls in enumerate(list(reagents_df["reagent_type"]))}

    catalyst_counts = Counter()
    sb_pair_counts = Counter()

    for cat, sb in pooled_positive:
        catalyst_counts[cat] += 1
        sb_pair_counts[sb] += 1

    # Get top 12 catalysts, fill if necessary
    top_12_catalysts_indices = [item[0] for item in catalyst_counts.most_common(12)]
    if len(top_12_catalysts_indices) < 12:
        logger.warning(f"Found only {len(top_12_catalysts_indices)} catalysts in pooled positive. Filling randomly.")
        all_catalyst_indices = [i for i, cls in class_map.items() if cls == 'C']
        remaining = list(set(all_catalyst_indices) - set(top_12_catalysts_indices))
        needed = 12 - len(top_12_catalysts_indices)
        if remaining:
            top_12_catalysts_indices.extend(remaining[:needed])

    # Get top 8 S-B pairs, fill if necessary
    top_8_sb_pairs_indices = [item[0] for item in sb_pair_counts.most_common(8)]
    if len(top_8_sb_pairs_indices) < 8:
        logger.warning(f"Found only {len(top_8_sb_pairs_indices)} SB pairs in pooled positive. Filling randomly.")
        all_solvent_indices = [i for i, cls in class_map.items() if cls == 'S']
        all_base_indices = [i for i, cls in class_map.items() if cls == 'B']
        all_sb_pairs = list(itertools.product(all_solvent_indices, all_base_indices))
        existing = set(top_8_sb_pairs_indices)
        remaining = [p for p in all_sb_pairs if p not in existing]
        needed = 8 - len(top_8_sb_pairs_indices)
        if remaining:
            top_8_sb_pairs_indices.extend(remaining[:needed])

    return top_12_catalysts_indices, top_8_sb_pairs_indices

# ---------------------------------------------------------------------------
# Modified ILP — operates on pooled pure-positive / pure-negative pairs
# ---------------------------------------------------------------------------

def construct_plate_bh_ilp_pooled(pure_positive_pairs, pure_negative_pairs,
                                  config, logger,
                                  negative_penalty_weight=1.0,
                                  minimum_pos_covered=None):
    """Construct a plate design via ILP operating on plate-level (cat, (sol, base)) pairs.

    Uncertain conditions (in both sets) must already be excluded before calling.
    """
    reagents_df = pd.read_csv(f"./reagents_dfs/{config['rtype']}_treshold_all_all_reagent_df.csv")
    class_map = {i: cls for i, cls in enumerate(list(reagents_df["reagent_type"]))}

    catalyst_indices = [i for i, cls in class_map.items() if cls == 'C']
    base_indices = [i for i, cls in class_map.items() if cls == 'B']
    solvent_indices = [i for i, cls in class_map.items() if cls == 'S']
    all_sb_pairs = list(itertools.product(solvent_indices, base_indices))

    pure_positive_list = list(pure_positive_pairs)
    pure_negative_list = list(pure_negative_pairs)

    # --- ILP Problem Setup ---
    prob = pulp.LpProblem("Maximize_Coverage_Pooled", pulp.LpMaximize)

    x_vars = pulp.LpVariable.dicts("Catalyst", catalyst_indices, cat='Binary')
    p_vars = pulp.LpVariable.dicts("SB_Pair", all_sb_pairs, cat='Binary')

    y_vars = pulp.LpVariable.dicts("Desirable", range(len(pure_positive_list)), cat='Binary')
    z_vars = pulp.LpVariable.dicts("Undesirable", range(len(pure_negative_list)), cat='Binary')

    # --- Objective ---
    prob += pulp.lpSum(y_vars) - negative_penalty_weight * pulp.lpSum(z_vars), "Total_Score"

    # --- Budget Constraints ---
    prob += pulp.lpSum(x_vars) == 12, "Catalyst_Budget"
    prob += pulp.lpSum(p_vars) == 8, "SB_Pair_Budget"

    # --- Linking Constraints for positive pairs ---
    for i, (cat, (sol, bas)) in enumerate(pure_positive_list):
        prob += y_vars[i] <= x_vars[cat]
        prob += y_vars[i] <= p_vars[(sol, bas)]
        prob += y_vars[i] >= x_vars[cat] + p_vars[(sol, bas)] - 1

    # --- Linking Constraints for negative pairs ---
    for i, (cat, (sol, bas)) in enumerate(pure_negative_list):
        prob += z_vars[i] <= x_vars[cat]
        prob += z_vars[i] <= p_vars[(sol, bas)]
        prob += z_vars[i] >= x_vars[cat] + p_vars[(sol, bas)] - 1

    # --- Minimum positive coverage constraint ---
    if minimum_pos_covered is not None:
        prob += pulp.lpSum(y_vars) >= minimum_pos_covered, "Minimum_Positive_Coverage"
        logger.info(f"Added constraint: minimum positive coverage = {minimum_pos_covered}")

    # --- Solve ---
    logger.info("Solving the ILP problem...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    logger.info("Solver finished.")

    if prob.status == pulp.LpStatusOptimal:
        selected_catalysts = [i for i in catalyst_indices if x_vars[i].varValue > 0.9]
        selected_sb_pairs = [p for p in all_sb_pairs if p_vars[p].varValue > 0.9]
        return selected_catalysts, selected_sb_pairs
    else:
        logger.error(f"ILP solver failed with status: {pulp.LpStatus[prob.status]}")
        logger.error(f"Problem has {len(pure_positive_list)} pure-positive and {len(pure_negative_list)} pure-negative pairs")
        if minimum_pos_covered is not None:
            logger.error(f"Minimum positive coverage constraint: {minimum_pos_covered}")
        if len(catalyst_indices) < 12:
            logger.error(f"Not enough catalysts available! Need 12, have {len(catalyst_indices)}")
        if len(all_sb_pairs) < 8:
            logger.error(f"Not enough SB pairs available! Need 8, have {len(all_sb_pairs)}")
        return None, None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 (revised): Compare plate design algorithms with plate-level 4-category coverage."
    )
    parser.add_argument("--gen_model_path", type=str, required=True, help="Path to generative model (.pt file).")
    parser.add_argument("--gen_config_path", type=str, required=True, help="Path to generative model config (YAML file).")
    parser.add_argument("--n_conditions", type=int, default=500, help="Number of conditions to generate.")
    parser.add_argument("--reaction_smiles_list", type=str, nargs='+', help="List of reaction SMILES to design plates for.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--negative_penalty_weight", type=float, default=1.0, help="Weight to penalize negative conditions in the ILP objective.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="experiment_3_new_plate_design", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (optional).")

    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    logger.info(f"Args: {args}")

    with open(args.gen_config_path, 'r') as f:
        gen_config = yaml.safe_load(f)
    gen_config = flatten_wandb_config(gen_config)

    # Initialize WandB if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "gen_model_path": args.gen_model_path,
                "gen_config_path": args.gen_config_path,
                "n_conditions": args.n_conditions,
                "negative_penalty_weight": args.negative_penalty_weight,
                "model_type": gen_config.get("model_type"),
                "rtype": gen_config.get("rtype") if "bh" in gen_config.get("filepath", "") else "unknown",
            }
        )
        logger.info(f"WandB initialized: {wandb.run.url}")

    logger.info("Starting Experiment 3 (revised — plate-level 4-category coverage)...")
    if gen_config.get("device") == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif gen_config.get("device") == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    gen_config['device'] = device

    file_path = gen_config.get('filepath')
    if not file_path:
        logger.error("'filepath' not in generative model config.")
        return

    if "bh" in file_path: gen_config["rtype"] = 'bh'
    if "all_all" in file_path: gen_config["data_type"] = "all"

    loaded_npz = np.load(file_path, allow_pickle=True)
    _, clist = loaded_npz['data']
    gen_config["clist"] = clist

    all_rmol_graphs, all_pmol_graphs, all_reaction_labels, all_reaction_smiles, all_embeddings_mol = load_graph_data(gen_config)
    smiles_to_idx = {smi: i for i, smi in enumerate(all_reaction_smiles)}

    reagent_df_path = f"reagents_dfs/{gen_config['rtype']}_treshold_all_all_reagent_df.csv"
    reagent_df = pd.read_csv(reagent_df_path)
    class_map = {i: cls for i, cls in enumerate(list(reagent_df["reagent_type"]))}

    if not reagent_df['reagent'].is_unique:
        duplicates = reagent_df[reagent_df['reagent'].duplicated(keep=False)]
        logger.error(f"FATAL: Duplicate reagent names found in {reagent_df_path}.")
        logger.error("Duplicate entries:\n" + duplicates.to_string())
        raise ValueError("Duplicate reagent names found. Aborting experiment.")

    reaction_smiles_to_process = args.reaction_smiles_list
    if not reaction_smiles_to_process:
        logger.info("No reaction SMILES provided. Selecting 10 random SMILES.")
        reaction_smiles_to_process = random.sample(all_reaction_smiles, 10)

    # Force exactly 10 reactions
    if len(reaction_smiles_to_process) != 10:
        logger.info(f"Adjusting to exactly 10 reactions (was {len(reaction_smiles_to_process)}).")
        if len(reaction_smiles_to_process) > 10:
            reaction_smiles_to_process = reaction_smiles_to_process[:10]
        else:
            additional_needed = 10 - len(reaction_smiles_to_process)
            remaining_smiles = [smi for smi in all_reaction_smiles if smi not in reaction_smiles_to_process]
            reaction_smiles_to_process.extend(random.sample(remaining_smiles, additional_needed))

    negative_penalty_weights = [0.5, 1.0, 2.0]

    # -----------------------------------------------------------------------
    # Part 1: Multi-weight comparison
    # -----------------------------------------------------------------------
    results = []

    for single_reaction_smiles in reaction_smiles_to_process:
        logger.info(f"\n{'='*80}\nProcessing Reaction: {single_reaction_smiles}\n{'='*80}")
        reaction_idx = smiles_to_idx.get(single_reaction_smiles)
        if reaction_idx is None:
            logger.error(f"SMILES {single_reaction_smiles} not found. Skipping.")
            continue

        single_rmol_graph = [all_rmol_graphs[reaction_idx]]
        single_pmol_graph = [all_pmol_graphs[reaction_idx]]
        single_label = [all_reaction_labels[reaction_idx]]
        single_smiles = [all_reaction_smiles[reaction_idx]]
        single_mol_emb = [all_embeddings_mol[reaction_idx]]

        Model, Trainer, collate_fn = get_model_and_trainer(gen_config['model_type'], gen_config)
        gen_config['collate_fn'] = collate_fn

        temp_data = GraphDataset(single_rmol_graph, single_pmol_graph, single_label, single_smiles, single_mol_emb, gen_config, split='tst', device=device)
        gen_config["n_classes"] = temp_data.n_classes
        gen_config["rmol_max_cnt"] = temp_data.rmol_max_cnt
        gen_config["pmol_max_cnt"] = temp_data.pmol_max_cnt
        gen_config["n_info"] = get_cardinalities_classes(gen_config)

        single_loader = DataLoader(dataset=temp_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
        gen_net = Model(gen_config["rtype"], temp_data.node_dim, temp_data.edge_dim, gen_config["n_classes"], gen_config["n_info"], temp_data.emb_dim)
        gen_net.load_state_dict(torch.load(args.gen_model_path, map_location=device))
        gen_net.to(device)
        gen_net.eval()

        gen_config['model_path'] = "dummy"
        gen_trainer = Trainer(gen_net, device, gen_config)
        generated_conditions = generate_conditions_from_model(gen_trainer, single_loader, gen_config, args.n_conditions)

        positive_conditions_raw = generated_conditions[single_reaction_smiles]['positive']
        negative_conditions_raw = generated_conditions[single_reaction_smiles]['negative']

        # Deduplicate raw conditions
        positive_conditions = list(set(positive_conditions_raw))
        negative_conditions = list(set(negative_conditions_raw))

        # Pool to plate level
        pooled_positive = pool_conditions_to_plate_level(positive_conditions, class_map)
        pooled_negative = pool_conditions_to_plate_level(negative_conditions, class_map)
        pure_positive, pure_negative, uncertain = categorize_pooled_conditions(pooled_positive, pooled_negative)

        logger.info(f"Raw: {len(positive_conditions_raw)} pos, {len(negative_conditions_raw)} neg")
        logger.info(f"Unique raw: {len(positive_conditions)} pos, {len(negative_conditions)} neg")
        logger.info(f"Pooled (plate-level): {len(pooled_positive)} pos, {len(pooled_negative)} neg")
        logger.info(f"Categories: {len(pure_positive)} pure-pos, {len(pure_negative)} pure-neg, {len(uncertain)} uncertain")

        if args.wandb:
            wandb.log({
                "reaction_idx": reaction_idx,
                "conditions/positive_raw": len(positive_conditions_raw),
                "conditions/negative_raw": len(negative_conditions_raw),
                "conditions/positive_unique": len(positive_conditions),
                "conditions/negative_unique": len(negative_conditions),
                "conditions/pooled_positive": len(pooled_positive),
                "conditions/pooled_negative": len(pooled_negative),
                "conditions/pure_positive": len(pure_positive),
                "conditions/pure_negative": len(pure_negative),
                "conditions/uncertain": len(uncertain),
            })

        # Naive plate design (uses pooled_positive, including uncertain pairs)
        logger.info("\n--- Naive Plate Design (Frequency-based, pooled) ---")
        naive_cat_indices, naive_sb_indices = naive_plate_design_pooled(pooled_positive, gen_config, logger)
        if not naive_cat_indices or not naive_sb_indices:
            logger.error("Failed to generate naive plate design. Skipping this reaction.")
            continue

        naive_pos, naive_neg, naive_unc, naive_unk = score_plate_4category(
            naive_cat_indices, naive_sb_indices,
            pure_positive, pure_negative, uncertain,
            logger, "Naive"
        )

        # ILP at different penalty weights
        for neg_weight in negative_penalty_weights:
            logger.info(f"\n--- ILP Plate Design (weight={neg_weight}) ---")
            ilp_cats, ilp_sbs = construct_plate_bh_ilp_pooled(
                pure_positive, pure_negative, gen_config, logger, neg_weight
            )

            if ilp_cats is not None and ilp_sbs is not None:
                ilp_pos, ilp_neg, ilp_unc, ilp_unk = score_plate_4category(
                    ilp_cats, ilp_sbs,
                    pure_positive, pure_negative, uncertain,
                    logger, f"ILP (w={neg_weight})"
                )

                results.append({
                    'rxn_idx': reaction_idx,
                    'neg_weight': neg_weight,
                    'naive_pos': naive_pos,
                    'naive_neg': naive_neg,
                    'naive_uncertain': naive_unc,
                    'naive_unknown': naive_unk,
                    'ilp_pos': ilp_pos,
                    'ilp_neg': ilp_neg,
                    'ilp_uncertain': ilp_unc,
                    'ilp_unknown': ilp_unk,
                })

                if args.wandb:
                    wandb.log({
                        "reaction_idx": reaction_idx,
                        "neg_weight": neg_weight,
                        f"naive/pos_w{neg_weight}": naive_pos,
                        f"naive/neg_w{neg_weight}": naive_neg,
                        f"naive/uncertain_w{neg_weight}": naive_unc,
                        f"naive/unknown_w{neg_weight}": naive_unk,
                        f"ilp/pos_w{neg_weight}": ilp_pos,
                        f"ilp/neg_w{neg_weight}": ilp_neg,
                        f"ilp/uncertain_w{neg_weight}": ilp_unc,
                        f"ilp/unknown_w{neg_weight}": ilp_unk,
                    })
            else:
                logger.error(f"ILP failed for weight {neg_weight}.")
                results.append({
                    'rxn_idx': reaction_idx,
                    'neg_weight': neg_weight,
                    'naive_pos': naive_pos,
                    'naive_neg': naive_neg,
                    'naive_uncertain': naive_unc,
                    'naive_unknown': naive_unk,
                    'ilp_pos': None,
                    'ilp_neg': None,
                    'ilp_uncertain': None,
                    'ilp_unknown': None,
                })

    # Save multi-weight results
    results_df = pd.DataFrame(results)
    output_file = "plate_design_comparison_results_new.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")

    # -----------------------------------------------------------------------
    # Part 2: Constrained ILP comparison (for paper table)
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*80}\nConstrained ILP test: match naive positive coverage\n{'='*80}")

    constrained_results = []
    fixed_penalty_weight = 1.0

    for single_reaction_smiles in reaction_smiles_to_process:
        logger.info(f"\n--- Constrained ILP for: {single_reaction_smiles} ---")
        reaction_idx = smiles_to_idx.get(single_reaction_smiles)
        if reaction_idx is None:
            logger.error(f"SMILES {single_reaction_smiles} not found. Skipping.")
            continue

        # Re-generate conditions
        single_rmol_graph = [all_rmol_graphs[reaction_idx]]
        single_pmol_graph = [all_pmol_graphs[reaction_idx]]
        single_label = [all_reaction_labels[reaction_idx]]
        single_smiles = [all_reaction_smiles[reaction_idx]]
        single_mol_emb = [all_embeddings_mol[reaction_idx]]

        Model, Trainer, collate_fn = get_model_and_trainer(gen_config['model_type'], gen_config)
        gen_config['collate_fn'] = collate_fn

        temp_data = GraphDataset(single_rmol_graph, single_pmol_graph, single_label, single_smiles, single_mol_emb, gen_config, split='tst', device=device)
        gen_config["n_classes"] = temp_data.n_classes
        gen_config["rmol_max_cnt"] = temp_data.rmol_max_cnt
        gen_config["pmol_max_cnt"] = temp_data.pmol_max_cnt
        gen_config["n_info"] = get_cardinalities_classes(gen_config)

        single_loader = DataLoader(dataset=temp_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
        gen_net = Model(gen_config["rtype"], temp_data.node_dim, temp_data.edge_dim, gen_config["n_classes"], gen_config["n_info"], temp_data.emb_dim)
        gen_net.load_state_dict(torch.load(args.gen_model_path, map_location=device))
        gen_net.to(device)
        gen_net.eval()

        gen_config['model_path'] = "dummy"
        gen_trainer = Trainer(gen_net, device, gen_config)
        generated_conditions = generate_conditions_from_model(gen_trainer, single_loader, gen_config, args.n_conditions)

        positive_conditions_raw = generated_conditions[single_reaction_smiles]['positive']
        negative_conditions_raw = generated_conditions[single_reaction_smiles]['negative']

        positive_conditions = list(set(positive_conditions_raw))
        negative_conditions = list(set(negative_conditions_raw))

        # Pool and categorize
        pooled_positive = pool_conditions_to_plate_level(positive_conditions, class_map)
        pooled_negative = pool_conditions_to_plate_level(negative_conditions, class_map)
        pure_positive, pure_negative, uncertain = categorize_pooled_conditions(pooled_positive, pooled_negative)

        logger.info(f"Pooled: {len(pooled_positive)} pos, {len(pooled_negative)} neg")
        logger.info(f"Categories: {len(pure_positive)} pure-pos, {len(pure_negative)} pure-neg, {len(uncertain)} uncertain")

        # Naive plate design
        naive_cat_indices, naive_sb_indices = naive_plate_design_pooled(pooled_positive, gen_config, logger)
        if not naive_cat_indices or not naive_sb_indices:
            logger.error("Naive design failed. Skipping.")
            continue

        naive_pos, naive_neg, naive_unc, naive_unk = score_plate_4category(
            naive_cat_indices, naive_sb_indices,
            pure_positive, pure_negative, uncertain,
            logger, "Naive"
        )

        # Constrained ILP: must match or exceed naive positive coverage
        logger.info(f"Constraining ILP to >= {naive_pos} positive wells")
        constrained_cats, constrained_sbs = construct_plate_bh_ilp_pooled(
            pure_positive, pure_negative, gen_config, logger,
            fixed_penalty_weight, minimum_pos_covered=naive_pos
        )

        if constrained_cats is not None and constrained_sbs is not None:
            ilp_pos, ilp_neg, ilp_unc, ilp_unk = score_plate_4category(
                constrained_cats, constrained_sbs,
                pure_positive, pure_negative, uncertain,
                logger, f"Constrained ILP (min_pos={naive_pos})"
            )

            constrained_results.append({
                'rxn_idx': reaction_idx,
                'naive_pos': naive_pos,
                'naive_neg': naive_neg,
                'naive_uncertain': naive_unc,
                'naive_unknown': naive_unk,
                'ilp_pos': ilp_pos,
                'ilp_neg': ilp_neg,
                'ilp_uncertain': ilp_unc,
                'ilp_unknown': ilp_unk,
                'pos_diff': ilp_pos - naive_pos,
                'neg_reduction': naive_neg - ilp_neg,
            })

            if args.wandb:
                wandb.log({
                    "constrained/reaction_idx": reaction_idx,
                    "constrained/naive_pos": naive_pos,
                    "constrained/naive_neg": naive_neg,
                    "constrained/naive_uncertain": naive_unc,
                    "constrained/naive_unknown": naive_unk,
                    "constrained/ilp_pos": ilp_pos,
                    "constrained/ilp_neg": ilp_neg,
                    "constrained/ilp_uncertain": ilp_unc,
                    "constrained/ilp_unknown": ilp_unk,
                    "constrained/pos_diff": ilp_pos - naive_pos,
                    "constrained/neg_reduction": naive_neg - ilp_neg,
                })

            logger.info(f"Constrained ILP: pos={ilp_pos} (target>={naive_pos}), neg={ilp_neg} (naive={naive_neg}, reduction={naive_neg - ilp_neg})")
        else:
            logger.error(f"Constrained ILP failed for reaction {reaction_idx}.")
            constrained_results.append({
                'rxn_idx': reaction_idx,
                'naive_pos': naive_pos,
                'naive_neg': naive_neg,
                'naive_uncertain': naive_unc,
                'naive_unknown': naive_unk,
                'ilp_pos': None,
                'ilp_neg': None,
                'ilp_uncertain': None,
                'ilp_unknown': None,
                'pos_diff': None,
                'neg_reduction': None,
            })

    # Save constrained results
    constrained_df = pd.DataFrame(constrained_results)
    constrained_file = "constrained_ilp_comparison_results_new.csv"
    constrained_df.to_csv(constrained_file, index=False)
    logger.info(f"\nConstrained ILP results saved to {constrained_file}")

    # Summary table and statistics
    if constrained_results:
        successful = [r for r in constrained_results if r['ilp_pos'] is not None]
        if successful:
            avg_neg_reduction = np.mean([r['neg_reduction'] for r in successful])
            avg_pos_diff = np.mean([r['pos_diff'] for r in successful])

            # Print formatted summary table
            header = (
                f"{'Rxn':>5} | {'Naive':>5} {'Naive':>5} {'Naive':>5} {'Naive':>5} | "
                f"{'ILP':>5} {'ILP':>5} {'ILP':>5} {'ILP':>5} | {'Pos':>5} {'Neg':>5}"
            )
            subheader = (
                f"{'Idx':>5} | {'Pos':>5} {'Neg':>5} {'Unc':>5} {'Unk':>5} | "
                f"{'Pos':>5} {'Neg':>5} {'Unc':>5} {'Unk':>5} | {'Diff':>5} {'Reduc':>5}"
            )
            sep = "-" * len(header)
            table_lines = [
                "\n" + sep,
                "Constrained ILP vs Naive — 4-Category Plate Coverage (each row sums to 96)",
                sep,
                header,
                subheader,
                sep,
            ]
            for i, r in enumerate(successful, 1):
                table_lines.append(
                    f"{i:>5} | {r['naive_pos']:>5} {r['naive_neg']:>5} {r['naive_uncertain']:>5} {r['naive_unknown']:>5} | "
                    f"{r['ilp_pos']:>5} {r['ilp_neg']:>5} {r['ilp_uncertain']:>5} {r['ilp_unknown']:>5} | "
                    f"{r['pos_diff']:>5} {r['neg_reduction']:>5}"
                )
            table_lines.append(sep)
            table_lines.append(
                f"{'Avg':>5} | {' ':>5} {' ':>5} {' ':>5} {' ':>5} | "
                f"{' ':>5} {' ':>5} {' ':>5} {' ':>5} | "
                f"{avg_pos_diff:>5.1f} {avg_neg_reduction:>5.1f}"
            )
            table_lines.append(sep)
            logger.info("\n".join(table_lines))

            if args.wandb:
                wandb.log({
                    "summary/avg_neg_reduction": avg_neg_reduction,
                    "summary/avg_pos_diff": avg_pos_diff,
                    "summary/num_reactions": len(reaction_smiles_to_process),
                    "summary/num_successful": len(successful),
                })
                wandb.log({"results_table": wandb.Table(dataframe=results_df)})
                wandb.log({"constrained_results_table": wandb.Table(dataframe=constrained_df)})

    if args.wandb:
        wandb.finish()

    logger.info("\nExperiment 3 (revised) finished.")

if __name__ == "__main__":
    main()
