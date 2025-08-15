# This file contains miscellaneous functions that are used throughout the project
import numpy as np
from typing import List, Tuple
from collections import Counter

def create_pos_neg_count_matrices(reaction_labels: List[List], n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert list of reaction labels to separate binary multi-label matrices for positive and negative samples.
    
    Args:
        reaction_labels: List where each element is a list of (condition_idx, pos_neg_flag) tuples
        n_classes: Total number of condition classes
        
    Returns:
        Tuple of (pos_count_matrix, neg_count_matrix) each of shape (n_reactions, n_classes)
    """
    n_reactions = len(reaction_labels)
    pos_count_matrix_1 = np.zeros((n_reactions, n_classes), dtype=int)
    pos_count_matrix_0 = np.zeros((n_reactions, n_classes), dtype=int)
    neg_count_matrix_1 = np.zeros((n_reactions, n_classes), dtype=int)
    neg_count_matrix_0 = np.zeros((n_reactions, n_classes), dtype=int)
    
    for i, reaction_conditions in enumerate(reaction_labels):
        for condition_idx, pos_neg_flag, _ in reaction_conditions:
            not_condition_idx = [j for j in range(n_classes) if j not in condition_idx]
            if pos_neg_flag == 1:  # Positive conditions
                #add 1 to the index where the condition_idx is 1
                pos_count_matrix_1[i, condition_idx] += 1
                pos_count_matrix_0[i, not_condition_idx] += 1
            elif pos_neg_flag == 0:  # Negative conditions
                neg_count_matrix_1[i, condition_idx] += 1
                neg_count_matrix_0[i, not_condition_idx] += 1

                
    return pos_count_matrix_1, pos_count_matrix_0, neg_count_matrix_1, neg_count_matrix_0


def compute_random_baseline(tst_y_pos, n_classes, T=1, random_seed=None):
    """
    Compute random baseline metrics for multi-label prediction, matching the model's metric definitions.
    For each test sample, generate T random predictions (each a set of unique classes),
    and aggregate as in the model code (see Trainer classes).
    Args:
        tst_y_pos: List of lists of true positive class indices for each test sample
        n_classes: Total number of classes
        T: Number of random samples per test example (sample size)
        random_seed: Optional random seed for reproducibility
    Returns:
        accuracy: Fraction of samples where at least one true label is in any of the T random predictions
        macro_recall: Mean recall per sample (fraction of true labels recovered in any of the T predictions)
        micro_recall: Total true positives recovered / total true labels
    """
    # Filter out empty lists from tst_y_pos
    tst_y_pos = [y for y in tst_y_pos if y]
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = len(tst_y_pos)
    # For each test sample, generate T random predictions (each a list of unique classes)
    random_preds = []
    for _ in range(n_samples):
        preds_for_sample = [np.where(np.random.rand(n_classes) < 0.5)[0].tolist() for _ in range(T)]
        random_preds.append(preds_for_sample)
    # Now, for each sample, flatten the T predictions into a single list (as in model code)
    # This matches the structure: filtered_tst_y_preds_pos[i][:T] in the model
    # But in the model, each prediction is a set, so we concatenate all T predictions
    # and treat them as the pool of predictions for that sample
    accuracy = np.mean([
        np.max([(c in random_preds[i]) for c in tst_y_pos[i]]) if tst_y_pos[i] else 0.0
        for i in range(n_samples)
    ])
    macro_recall = np.mean([
        np.mean([(c in random_preds[i]) for c in tst_y_pos[i]]) if tst_y_pos[i] else 0.0
        for i in range(n_samples)
    ])
    total_true = np.sum([len(a) for a in tst_y_pos])
    micro_recall = (
        np.sum([
            np.sum([(c in random_preds[i]) for c in tst_y_pos[i]])
            for i in range(n_samples)
        ]) / total_true if total_true > 0 else 0.0
    )
    return accuracy, macro_recall, micro_recall


def compute_structured_random_baseline(tst_y_pos, n_classes, T=1, random_seed=None, rtype=None, n_info=None):
    """
    Compute a structured random baseline for multi-label prediction, matching the model's metric definitions.
    For each test sample, generate T random predictions according to the reaction type (bh or sm):
      - For bh: one random catalyst, one random base, one random solvent, water if required, and with 50% probability a random additive.
      - For sm: one random solvent 1, with 50% chance a random solvent 2, water (randomly present or not), with 50% chance one additive, then one catalyst and one base.
    Args:
        tst_y_pos: List of lists of true positive class indices for each test sample
        n_classes: Total number of classes
        T: Number of random samples per test example (sample size)
        random_seed: Optional random seed for reproducibility
        rtype: Reaction type ('bh' or 'sm')
        n_info: Tuple of (n_cats, n_sol_1, n_sol_2, n_add, n_base)
    Returns:
        accuracy, macro_recall, micro_recall
    """
    # Filter out empty lists from tst_y_pos
    tst_y_pos = [y for y in tst_y_pos if y]
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = len(tst_y_pos)
    if n_info is None or rtype not in ('bh', 'sm'):
        raise ValueError("rtype and n_info must be provided and valid.")
    n_cats, n_sol_1, n_sol_2, n_add, n_base = n_info
    # For each test sample, generate T structured random predictions
    random_preds = []
    for _ in range(n_samples):
        preds_for_sample = []
        for _ in range(T):
            pred = []
            if rtype == 'bh':
                # Catalyst
                cat_idx = np.random.randint(0, n_cats)
                pred.append(cat_idx)
                # Base
                base_idx = n_cats + np.random.randint(0, n_base)
                pred.append(base_idx)
                # Solvent 1
                solv1_idx = n_cats + n_base + np.random.randint(0, n_sol_1)
                pred.append(solv1_idx)
                # Water (50% change)
                if np.random.rand() < 0.5:
                    water_idx = n_cats + n_base + n_sol_1
                    pred.append(water_idx)
                # Additive (50% chance)
                if n_add > 0 and np.random.rand() < 0.5:
                    add_idx = n_cats + n_base + n_sol_1 + 1 + np.random.randint(0, n_add)
                    pred.append(add_idx)
            elif rtype == 'sm':
                # Solvent 1
                solv1_idx = np.random.randint(0, n_sol_1)
                pred.append(solv1_idx)
                # Solvent 2 (50% chance)
                if n_sol_2 > 0 and np.random.rand() < 0.5:
                    solv2_idx = n_sol_1 + np.random.randint(0, n_sol_2)
                    pred.append(solv2_idx)
                # Water (randomly present or not)
                water_idx = n_sol_1 + n_sol_2
                if np.random.rand() < 0.5:
                    pred.append(water_idx)
                # Additive (50% chance)
                if n_add > 0 and np.random.rand() < 0.5:
                    add_idx = n_sol_1 + n_sol_2 + 1 + np.random.randint(0, n_add)
                    pred.append(add_idx)
                # Catalyst
                cat_idx = n_sol_1 + n_sol_2 + 1 + n_add + np.random.randint(0, n_cats)
                pred.append(cat_idx)
                # Base
                base_idx = n_sol_1 + n_sol_2 + 1 + n_add + n_cats + np.random.randint(0, n_base)
                pred.append(base_idx)
            preds_for_sample.append(pred)
        random_preds.append(preds_for_sample)
    # Flatten T predictions for each sample
    accuracy = np.mean([
        np.max([(c in random_preds[i]) for c in tst_y_pos[i]]) if tst_y_pos[i] else 0.0
        for i in range(n_samples)
    ])
    macro_recall = np.mean([
        np.mean([(c in random_preds[i]) for c in tst_y_pos[i]]) if tst_y_pos[i] else 0.0
        for i in range(n_samples)
    ])
    total_true = np.sum([len(a) for a in tst_y_pos])
    micro_recall = (
        np.sum([
            np.sum([(c in random_preds[i]) for c in tst_y_pos[i]])
            for i in range(n_samples)
        ]) / total_true if total_true > 0 else 0.0
    )
    return accuracy, macro_recall, micro_recall


def compute_frequency_chain_baseline(
    tst_y,
    y_train,
    T=1,
    rtype=None,
    n_info=None,
    sample_neg=False,
    random_seed=None
):
    """
    Frequency-chain baseline: samples T sets of conditions for each test sample using empirical conditional distributions
    learned from y_train, in the canonical order for the given rtype. Supports both positive and negative samples.

    Args:
        tst_y: List of lists of true class indices for each test sample (positive or negative, depending on sample_neg)
        y_train: List of tuples (list_of_condition_indices, pos_neg_flag, <other>) for each training sample
        n_classes: Total number of classes
        T: Number of samples per test example
        rtype: Reaction type ('bh' or 'sm')
        n_info: Tuple of (n_cats, n_sol_1, n_sol_2, n_add, n_base)
        clist: List of class names (optional, for debugging)
        sample_neg: If True, use negative samples (pos_neg_flag == 0), else positive (pos_neg_flag == 1)
        random_seed: Optional random seed for reproducibility
    Returns:
        accuracy, macro_recall, micro_recall (as in other baselines)
    """
    # Filter out empty lists from tst_y
    tst_y = [y for y in tst_y if y]
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = len(tst_y)
    # --- Determine condition order and index ranges ---
    if rtype == 'bh':
        n_cats, n_sol_1, n_sol_2, n_add, n_base = n_info
        order = [
            ('cat', 0, n_cats),
            ('base', n_cats, n_cats + n_base),
            ('solv1', n_cats + n_base, n_cats + n_base + n_sol_1),
            ('water', n_cats + n_base + n_sol_1, n_cats + n_base + n_sol_1 + 1),
            ('add', n_cats + n_base + n_sol_1 + 1, n_cats + n_base + n_sol_1 + 1 + n_add)
        ]
    elif rtype == 'sm':
        n_cats, n_sol_1, n_sol_2, n_add, n_base = n_info
        order = [
            ('solv1', 0, n_sol_1),
            ('solv2', n_sol_1, n_sol_1 + n_sol_2),
            ('water', n_sol_1 + n_sol_2, n_sol_1 + n_sol_2 + 1),
            ('add', n_sol_1 + n_sol_2 + 1, n_sol_1 + n_sol_2 + 1 + n_add),
            ('cat', n_sol_1 + n_sol_2 + 1 + n_add, n_sol_1 + n_sol_2 + 1 + n_add + n_cats),
            ('base', n_sol_1 + n_sol_2 + 1 + n_add + n_cats, n_sol_1 + n_sol_2 + 1 + n_add + n_cats + n_base)
        ]
    else:
        raise ValueError(f"Unknown rtype: {rtype}")

    # --- Build conditional frequency tables ---
    cond_freqs = [{} for _ in order]
    for sample in y_train:
        for experiment in sample:
            # experiment: (list_of_condition_indices, pos_neg_flag, <other>)
            cond_idx_list, pos_neg_flag, *_ = experiment
            if (not sample_neg and pos_neg_flag == 1) or (sample_neg and pos_neg_flag == 0):
                indices = list(cond_idx_list)
            else:
                continue
            # For each step, extract the index for that condition type (if present)
            step_vals = []
            for name, start, end in order:
                found = [i for i in indices if start <= i < end]
                if found:
                    step_vals.append(found[0])
                else:
                    #to account for the case where the condition is not present in the training label
                    step_vals.append(f'{name}_None')
            prev = tuple()
            for j, val in enumerate(step_vals):
                if val is not None:
                    cond_freqs[j].setdefault(prev, Counter())
                    cond_freqs[j][prev][val] += 1
                prev = prev + (val,)
    # --- Convert counts to probabilities ---
    cond_probs = [{} for _ in order]
    for j, table in enumerate(cond_freqs):
        for prev, counter in table.items():
            total = sum(counter.values())
            if total > 0:
                cond_probs[j][prev] = {k: v / total for k, v in counter.items()}
            else:
                cond_probs[j][prev] = {}
    # --- For each test sample, sample T times from the chain ---
    preds = []
    for _ in range(n_samples):
        preds_for_sample = []
        for _ in range(T):
            prev = tuple()
            pred = []
            for j, (name, start, end) in enumerate(order):
                prob_dict = cond_probs[j].get(prev, None)
                if prob_dict is None or len(prob_dict) == 0:
                    choices = list(range(start, end))
                    val = np.random.choice(choices)
                else:
                    choices, probs = zip(*prob_dict.items())
                    probs = np.array(probs)
                    probs = probs / probs.sum()  # Normalize
                    val = np.random.choice(choices, p=probs)
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        pass
                pred.append(val)
                prev = prev + (val,)
            #remove the strings in the preds to take out none values
            pred = [int(p) for p in pred if isinstance(p, (int, np.integer))]
            preds_for_sample.append(pred)
        preds.append(preds_for_sample)
    # --- Evaluate metrics (same as other baselines) ---
    accuracy = np.mean([
        np.max([(c in preds[i]) for c in tst_y[i]]) if tst_y[i] else 0.0
        for i in range(n_samples)
    ])
    macro_recall = np.mean([
        np.mean([(c in preds[i]) for c in tst_y[i]]) if tst_y[i] else 0.0
        for i in range(n_samples)
    ])
    total_true = np.sum([len(a) for a in tst_y])
    micro_recall = (
        np.sum([
            np.sum([(c in preds[i]) for c in tst_y[i]])
            for i in range(n_samples)
        ]) / total_true if total_true > 0 else 0.0
    )
    return accuracy, macro_recall, micro_recall


def compute_most_frequent_baseline(tst_y, y_train, n_info, rtype, T=1):
    """
    Baseline: Predicts the T most frequent conditions from the training set for all test samples.
    Args:
        tst_y: List of lists of true class indices for each test sample
        y_train: List of tuples (list_of_condition_indices, pos_neg_flag, <other>) for each training sample
        n_info: Tuple of (n_cats, n_sol_1, n_sol_2, n_add, n_base)
        rtype: Reaction type ('bh' or 'sm')
        T: Number of most frequent conditions to predict
    Returns:
        accuracy, macro_recall, micro_recall (as in other baselines)
    """
    # Filter out empty lists from tst_y
    tst_y = [y for y in tst_y if y]
    
    n_cats, n_sol_1, n_sol_2, n_add, n_base = n_info
    if rtype == 'bh':
        order = [
            ('cat', 0, n_cats),
            ('base', n_cats, n_cats + n_base),
            ('solv1', n_cats + n_base, n_cats + n_base + n_sol_1),
            ('water', n_cats + n_base + n_sol_1, n_cats + n_base + n_sol_1 + 1),
            ('add', n_cats + n_base + n_sol_1 + 1, n_cats + n_base + n_sol_1 + 1 + n_add)
        ]
    elif rtype == 'sm':
        order = [
            ('solv1', 0, n_sol_1),
            ('solv2', n_sol_1, n_sol_1 + n_sol_2),
            ('water', n_sol_1 + n_sol_2, n_sol_1 + n_sol_2 + 1),
            ('add', n_sol_1 + n_sol_2 + 1, n_sol_1 + n_sol_2 + 1 + n_add),
            ('cat', n_sol_1 + n_sol_2 + 1 + n_add, n_sol_1 + n_sol_2 + 1 + n_add + n_cats),
            ('base', n_sol_1 + n_sol_2 + 1 + n_add + n_cats, n_sol_1 + n_sol_2 + 1 + n_add + n_cats + n_base)
        ]
        
    condition_counter = Counter()
    for sample in y_train:
        for experiment in sample:
            cond_idx_list, pos_neg_flag, *_ = experiment
            if pos_neg_flag == 1:
                condition_counter.update(cond_idx_list)

    # Get the T most frequent conditions for each type
    most_common_conditions_by_type = {}
    for name, start, end in order:
        if name in ['add', 'solv2']:  # Skip these as they are often absent
            most_common_conditions_by_type[name] = [None] * T # Placeholder
            continue
        
        counter_subset = Counter({k: v for k, v in condition_counter.items() if start <= k < end})
        
        # Get the top T items, but handle cases with fewer than T unique items
        top_items = [item[0] for item in counter_subset.most_common(T)]
        if len(top_items) < T:
            # If not enough unique items, cycle through the ones we have
            cycled_items = [top_items[i % len(top_items)] for i in range(T)] if top_items else [None] * T
            most_common_conditions_by_type[name] = cycled_items
        else:
            most_common_conditions_by_type[name] = top_items

    # Construct T most frequent predictions
    most_frequent_preds = []
    for i in range(T):
        pred = []
        for name, _, _ in order:
            condition = most_common_conditions_by_type[name][i]
            if condition is not None:
                pred.append(condition)
        most_frequent_preds.append(pred)

    # Predict the T most frequent conditions for all test samples
    preds = [most_frequent_preds for _ in tst_y]
    n_samples = len(tst_y)
    
    accuracy = np.mean([
        np.max([(c in preds[i]) for c in tst_y[i]]) if tst_y[i] else 0.0
        for i in range(n_samples)
    ])
    macro_recall = np.mean([
        np.mean([(c in preds[i]) for c in tst_y[i]]) if tst_y[i] else 0.0
        for i in range(n_samples)
    ])
    total_true = np.sum([len(a) for a in tst_y])
    micro_recall = (
        np.sum([
            np.sum([(c in preds[i]) for c in tst_y[i]])
            for i in range(n_samples)
        ]) / total_true if total_true > 0 else 0.0
    )
    return accuracy, macro_recall, micro_recall