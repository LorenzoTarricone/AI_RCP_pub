#!/usr/bin/env python3
"""
Stratified sampling utilities for multi-label reaction condition prediction.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging
from typing import List, Tuple, Optional

def create_multilabel_matrix(reaction_labels: List[List], n_classes: int) -> np.ndarray:
    """
    Convert list of reaction labels to binary multi-label matrix.
    
    Args:
        reaction_labels: List where each element is a list of (condition_idx, pos_neg_flag) tuples
        n_classes: Total number of condition classes
        
    Returns:
        Binary matrix of shape (n_reactions, n_classes)
    """
    n_reactions = len(reaction_labels)
    multilabel_matrix = np.zeros((n_reactions, n_classes), dtype=int)

    
    for i, reaction_conditions in enumerate(reaction_labels):
        for condition_idx, pos_neg_flag, _ in reaction_conditions:
            if pos_neg_flag == 1:  # Only consider positive conditions for stratification
                multilabel_matrix[i, condition_idx] = 1
                
    return multilabel_matrix

def iterative_stratified_split(reaction_labels: List[List], n_classes: int, 
                              test_size: float, n_folds: int, 
                              random_state: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Perform stratified train/test split and k-fold CV using iterative stratification.
    
    Args:
        reaction_labels: List of reaction labels 
        n_classes: Total number of condition classes
        test_size: Fraction for test set
        n_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (test_indices, list_of_fold_splits)
    """
    try:
        from skmultilearn.model_selection import IterativeStratification
    except ImportError:
        raise ImportError("Please install scikit-multilearn: pip install scikit-multilearn")
    
    logger = logging.getLogger(__name__)
    
    # Convert to multi-label matrix
    multilabel_matrix = create_multilabel_matrix(reaction_labels, n_classes)
    n_reactions = len(reaction_labels)
    
    logger.info(f"Multi-label matrix shape: {multilabel_matrix.shape}")
    logger.info(f"Label density: {multilabel_matrix.mean():.3f}")
    
    # Calculate split sizes
    indices = np.arange(n_reactions)
    train_val_size = int(n_reactions * (1 - test_size))
    test_size_actual = n_reactions - train_val_size
    
    # First split: separate test set
    try:
        if random_state is not None:
            stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[(1-test_size), test_size], random_state=np.random.seed(random_state))
        else:
            stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[(1-test_size), test_size])
        
        splits = list(stratifier.split(indices.reshape(-1, 1), multilabel_matrix))
        if len(splits) < 2:
            raise ValueError("Insufficient splits generated")
        
        # Use the split that gives us closest to desired test size
        split_idx = 0 if len(splits[0][1]) <= test_size_actual else 1
        train_val_indices, test_indices = splits[split_idx]
        
    except Exception as e:
        logger.warning(f"Iterative stratification failed for test split: {e}. Falling back to random")
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(indices)
        train_val_indices = shuffled_indices[:train_val_size]
        test_indices = shuffled_indices[train_val_size:]
    
    logger.info(f"Test set size: {len(test_indices)} (target: {test_size_actual})")
    
    # Second split: k-fold CV on remaining data
    train_val_labels = multilabel_matrix[train_val_indices]
    if random_state is not None:
        cv_stratifier = IterativeStratification(n_splits=n_folds, order=1, random_state=np.random.seed(random_state))
    else:
        cv_stratifier = IterativeStratification(n_splits=n_folds, order=1)
    
    fold_splits = []
    try:
        for train_fold_idx, val_fold_idx in cv_stratifier.split(train_val_indices.reshape(-1, 1), train_val_labels):
            # Convert back to original indices
            train_indices = train_val_indices[train_fold_idx]
            val_indices = train_val_indices[val_fold_idx]
            fold_splits.append((train_indices, val_indices))
    except Exception as e:
        logger.warning(f"Iterative stratification failed for CV: {e}. Using simple k-fold.")
        fold_splits = simple_kfold_split(train_val_indices, n_folds, random_state)
    
    return test_indices, fold_splits

def stratify_by_condition_count(reaction_labels: List[List], test_size: float, 
                               n_folds: int, random_state: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Stratify by number of positive conditions per reaction (simpler approach).
    
    Args:
        reaction_labels: List of reaction labels
        test_size: Fraction for test set
        n_folds: Number of CV folds  
        random_state: Random seed
        
    Returns:
        Tuple of (test_indices, list_of_fold_splits)
    """
    logger = logging.getLogger(__name__)
    
    # Count positive conditions per reaction
    condition_counts = []
    for reaction_conditions in reaction_labels:
        pos_count = sum(1 for _, pos_neg_flag in reaction_conditions if pos_neg_flag == 1)
        condition_counts.append(pos_count)
    
    condition_counts = np.array(condition_counts)
    unique_counts, count_frequencies = np.unique(condition_counts, return_counts=True)
    
    logger.info(f"Condition count distribution: {dict(zip(unique_counts, count_frequencies))}")
    
    # Check if we have enough samples for stratification
    min_samples_per_class = 2 * n_folds  # Need at least 2 samples per class per fold
    rare_classes = unique_counts[count_frequencies < min_samples_per_class]
    
    if len(rare_classes) > 0:
        logger.warning(f"Some condition counts have too few samples for stratification: {rare_classes}")
        logger.info("Falling back to simple random split")
        indices = np.arange(len(reaction_labels))
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(indices)
        test_size_actual = int(len(indices) * test_size)
        test_indices = shuffled_indices[:test_size_actual]
        remaining_indices = shuffled_indices[test_size_actual:]
        fold_splits = simple_kfold_split(remaining_indices, n_folds, random_state)
        return test_indices, fold_splits
    
    # Use StratifiedKFold to split by condition counts
    indices = np.arange(len(reaction_labels))
    
    # First split: test set
    try:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, stratify=condition_counts, 
            random_state=random_state
        )
    except ValueError as e:
        logger.warning(f"Stratified test split failed: {e}. Using random split.")
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(indices)
        test_size_actual = int(len(indices) * test_size)
        test_indices = shuffled_indices[:test_size_actual]
        train_val_indices = shuffled_indices[test_size_actual:]
        fold_splits = simple_kfold_split(train_val_indices, n_folds, random_state)
        return test_indices, fold_splits
    
    # Second split: k-fold CV
    train_val_counts = condition_counts[train_val_indices]
    
    try:
        if random_state is not None:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
        
        fold_splits = []
        for train_fold_idx, val_fold_idx in skf.split(train_val_indices, train_val_counts):
            train_indices = train_val_indices[train_fold_idx]
            val_indices = train_val_indices[val_fold_idx]
            fold_splits.append((train_indices, val_indices))
    except ValueError as e:
        logger.warning(f"Stratified CV split failed: {e}. Using simple k-fold.")
        fold_splits = simple_kfold_split(train_val_indices, n_folds, random_state)
        
    return test_indices, fold_splits

def stratify_by_frequent_conditions(reaction_labels: List[List], n_classes: int, 
                                   test_size: float, n_folds: int, 
                                   top_k: int = 5, random_state: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Stratify by presence of most frequent conditions.
    
    Args:
        reaction_labels: List of reaction labels
        n_classes: Total number of condition classes
        test_size: Fraction for test set
        n_folds: Number of CV folds
        top_k: Number of most frequent conditions to consider
        random_state: Random seed
        
    Returns:
        Tuple of (test_indices, list_of_fold_splits)
    """
    logger = logging.getLogger(__name__)
    
    # Count condition frequencies
    condition_frequencies = np.zeros(n_classes)
    for reaction_conditions in reaction_labels:
        for condition_idx, pos_neg_flag in reaction_conditions:
            if pos_neg_flag == 1:
                condition_frequencies[condition_idx] += 1
    
    # Get top-k most frequent conditions
    top_conditions = np.argsort(condition_frequencies)[-top_k:]
    logger.info(f"Top {top_k} conditions: {top_conditions} with frequencies: {condition_frequencies[top_conditions]}")
    
    # Create stratification labels based on presence of top conditions
    stratify_labels = []
    for reaction_conditions in reaction_labels:
        pos_conditions = {idx for idx, flag in reaction_conditions if flag == 1}
        # Create binary string indicating presence of each top condition
        label_str = ''.join('1' if cond in pos_conditions else '0' for cond in top_conditions)
        stratify_labels.append(label_str)
    
    stratify_labels = np.array(stratify_labels)
    unique_labels, label_counts = np.unique(stratify_labels, return_counts=True)
    logger.info(f"Stratification pattern distribution: {len(unique_labels)} unique patterns")
    
    # Merge rare patterns into "other" category
    min_samples = max(2, len(reaction_labels) // (20 * n_folds))  # At least 2 samples per class
    rare_patterns = unique_labels[label_counts < min_samples]
    
    stratify_labels_merged = stratify_labels.copy()
    for pattern in rare_patterns:
        stratify_labels_merged[stratify_labels_merged == pattern] = 'other'
    
    logger.info(f"After merging rare patterns: {len(np.unique(stratify_labels_merged))} patterns")
    
    # Use StratifiedKFold
    indices = np.arange(len(reaction_labels))
    
    try:
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, stratify=stratify_labels_merged, 
            random_state=random_state
        )
    except ValueError as e:
        logger.warning(f"Stratified test split failed: {e}. Using random split.")
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(indices)
        test_size_actual = int(len(indices) * test_size)
        test_indices = shuffled_indices[:test_size_actual]
        train_val_indices = shuffled_indices[test_size_actual:]
        fold_splits = simple_kfold_split(train_val_indices, n_folds, random_state)
        return test_indices, fold_splits
    
    train_val_labels = stratify_labels_merged[train_val_indices]
    
    try:
        if random_state is not None:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
        
        fold_splits = []
        for train_fold_idx, val_fold_idx in skf.split(train_val_indices, train_val_labels):
            train_indices = train_val_indices[train_fold_idx]
            val_indices = train_val_indices[val_fold_idx]
            fold_splits.append((train_indices, val_indices))
    except ValueError as e:
        logger.warning(f"Stratified CV split failed: {e}. Using simple k-fold.")
        fold_splits = simple_kfold_split(train_val_indices, n_folds, random_state)
        
    return test_indices, fold_splits

def simple_kfold_split(indices: np.ndarray, n_folds: int, 
                      random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Fallback to simple k-fold split."""
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(indices)
    
    fold_splits = []
    fold_size = len(indices) // n_folds
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(indices)
        
        val_indices = shuffled_indices[val_start:val_end]
        train_indices = np.concatenate([shuffled_indices[:val_start], shuffled_indices[val_end:]])
        
        fold_splits.append((train_indices, val_indices))
    
    return fold_splits

def analyze_stratification_quality(reaction_labels: List[List], n_classes: int, 
                                  test_indices: np.ndarray, fold_splits: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Analyze and log the quality of stratification."""
    logger = logging.getLogger(__name__)
    
    multilabel_matrix = create_multilabel_matrix(reaction_labels, n_classes)
    
    # Overall statistics
    overall_label_freq = multilabel_matrix.mean(axis=0)
    logger.info(f"Overall label frequencies (top 10): {np.sort(overall_label_freq)[-10:]}")
    
    # Test set analysis
    test_label_freq = multilabel_matrix[test_indices].mean(axis=0)
    logger.info(f"Test set label frequency deviation: {np.abs(test_label_freq - overall_label_freq).mean():.4f}")
    
    # CV folds analysis
    fold_deviations = []
    for i, (train_idx, val_idx) in enumerate(fold_splits):
        val_label_freq = multilabel_matrix[val_idx].mean(axis=0)
        deviation = np.abs(val_label_freq - overall_label_freq).mean()
        fold_deviations.append(deviation)
        logger.info(f"Fold {i+1} validation set deviation: {deviation:.4f}")
    
    logger.info(f"Average CV fold deviation: {np.mean(fold_deviations):.4f} ± {np.std(fold_deviations):.4f}") 

def stratified_train_val_split(reaction_labels: List[List], n_classes: int,
                               val_size: float, stratification_method: str = "iterative",
                               top_k: int = 5, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single stratified train/validation split using the same methodology as the existing functions
    but without requiring k-fold cross-validation.
    
    Args:
        reaction_labels: List of reaction labels (multilabel format)
        n_classes: Total number of condition classes
        val_size: Fraction for validation set (e.g., 0.1 for 10%)
        stratification_method: Method to use ("iterative", "condition_count", "frequent_conditions", or "random")
        top_k: Number of most frequent conditions to consider (for frequent_conditions method)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices)
    """
    logger = logging.getLogger(__name__)
    
    num_reactions = len(reaction_labels)
    indices = np.arange(num_reactions)
    
    if stratification_method == "random":
        # Simple random split
        if random_state is not None:
            np.random.seed(random_state)
        val_indices = np.random.choice(indices, size=int(val_size * num_reactions), replace=False)
        train_indices = np.setdiff1d(indices, val_indices)
        logger.info(f"Used random split: {len(train_indices)} train, {len(val_indices)} val")
        
    elif stratification_method == "iterative":
        try:
            from skmultilearn.model_selection import IterativeStratification
            
            multilabel_matrix = create_multilabel_matrix(reaction_labels, n_classes)
            
            val_size_actual = int(num_reactions * val_size)

            if random_state is not None:
                stratifier = IterativeStratification(
                    n_splits=2,
                    order=1,
                    sample_distribution_per_fold=[1.0 - val_size, val_size],
                    random_state=np.random.seed(random_state)
                )
            else:
                stratifier = IterativeStratification(
                    n_splits=2,
                    order=1,
                    sample_distribution_per_fold=[1.0 - val_size, val_size]
                )

            splits = list(stratifier.split(indices.reshape(-1, 1), multilabel_matrix))
            if len(splits) < 2:
                raise ValueError("Insufficient splits generated by IterativeStratification")
            
            # Use the split that gives us closest to desired test size
            split_idx = 0 if len(splits[0][1]) <= val_size_actual else 1
            train_indices, val_indices = splits[split_idx]

            logger.info(f"Used iterative stratification: {len(train_indices)} train, {len(val_indices)} val")
            
        except ImportError:
            logger.warning("skmultilearn package not available. Falling back to random split.")
            return stratified_train_val_split(reaction_labels, n_classes, val_size, "random", top_k, random_state)
        except Exception as e:
            logger.warning(f"Iterative stratification failed: {e}. Falling back to random split.")
            return stratified_train_val_split(reaction_labels, n_classes, val_size, "random", top_k, random_state)
            
    elif stratification_method == "condition_count":
        # Stratify by number of conditions per reaction
        condition_counts = np.array([sum(1 for _, flag in labels if flag == 1) for labels in reaction_labels])
        
        try:
            train_indices, val_indices = train_test_split(
                indices, test_size=val_size, stratify=condition_counts, random_state=random_state
            )
            logger.info(f"Used condition count stratification: {len(train_indices)} train, {len(val_indices)} val")
            
        except Exception as e:
            logger.warning(f"Condition count stratification failed: {e}. Using random split.")
            return stratified_train_val_split(reaction_labels, n_classes, val_size, "random", top_k, random_state)
            
    elif stratification_method == "frequent_conditions":
        # Stratify by presence of most frequent conditions
        
        # Count condition frequencies
        condition_frequencies = np.zeros(n_classes)
        for labels in reaction_labels:
            for label in labels:
                if isinstance(label, (list, tuple)):
                    condition_frequencies[label[0]] += 1
                else:
                    condition_frequencies[label] += 1
        
        # Find top-k most frequent conditions
        top_conditions = np.argsort(condition_frequencies)[-top_k:]
        
        # Create stratification key based on presence of top conditions
        stratify_labels = []
        for labels in reaction_labels:
            label_set = set()
            for label in labels:
                if isinstance(label, (list, tuple)):
                    label_set.add(label[0])
                else:
                    label_set.add(label)
            
            # Create binary signature for top conditions
            signature = tuple(1 if cond in label_set else 0 for cond in top_conditions)
            stratify_labels.append(signature)
        
        try:
            # Convert to string for stratification (sklearn needs hashable labels)
            stratify_keys = [str(sig) for sig in stratify_labels]
            
            train_indices, val_indices = train_test_split(
                indices, test_size=val_size, stratify=stratify_keys, random_state=random_state
            )
            
            logger.info(f"Used frequent conditions stratification (top-{top_k}): {len(train_indices)} train, {len(val_indices)} val")
            
        except Exception as e:
            logger.warning(f"Frequent conditions stratification failed: {e}. Using condition_count method.")
            return stratified_train_val_split(reaction_labels, n_classes, val_size, "condition_count", top_k, random_state)
    
    else:
        raise ValueError(f"Unknown stratification_method: {stratification_method}")
    
    # Verify the split
    if len(train_indices) + len(val_indices) != num_reactions:
        raise RuntimeError(f"Data loss detected: {len(train_indices)} + {len(val_indices)} != {num_reactions}")
    
    if len(set(train_indices) & set(val_indices)) > 0:
        raise RuntimeError("Overlapping indices detected between train and validation sets")
    
    return train_indices, val_indices
