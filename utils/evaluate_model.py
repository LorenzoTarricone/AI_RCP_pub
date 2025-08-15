#!/usr/bin/env python3
"""
Model evaluation utilities for reaction condition prediction.
Handles evaluation of different model types with metrics calculation and diversity assessment.
"""

import numpy as np
import logging

def evaluate_simple_models(trainer, tst_loader, tst_y_pos, tst_y_neg, config, fold, wandb_is_active, logger, temperature=1.0):
    """
    Evaluate simple models (rxnfp, baselineofbaseline) that only produce single predictions.
    
    Args:
        trainer: Model trainer instance
        tst_loader: Test data loader
        tst_y_pos: Positive test labels
        tst_y_neg: Negative test labels (can be None)
        config: Configuration dictionary
        fold: Current fold number
        wandb_is_active: Whether W&B is active
        logger: Logger instance
        temperature: Temperature parameter for scaling logits (default: 1.0)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    tst_y_preds_pos, tst_y_preds_neg = trainer.inference(tst_loader, temperature=temperature)
    T = 1  # Only T=1 for these models

    # Filter out empty lists for initial best metrics calculation
    filtered_tst_y_pos = [y for y in tst_y_pos if y]
    best_macro_recall_pos = np.mean([1/len(y) for y in filtered_tst_y_pos]) if filtered_tst_y_pos else 0.0
    best_micro_recall_pos = len(filtered_tst_y_pos) / np.sum([len(a) for a in filtered_tst_y_pos]) if filtered_tst_y_pos else 0.0
    
    if config["data_type"] == "all":
        filtered_tst_y_neg = [y for y in tst_y_neg if y]
        best_macro_recall_neg = np.mean([1/len(y) for y in filtered_tst_y_neg]) if filtered_tst_y_neg else 0.0
        best_micro_recall_neg = len(filtered_tst_y_neg) / np.sum([len(a) for a in filtered_tst_y_neg]) if filtered_tst_y_neg else 0.0

    # Process positive predictions
    pos_pairs = zip(tst_y_pos, tst_y_preds_pos)
    filtered_pos_pairs = [(y_true, y_pred) for y_true, y_pred in pos_pairs if y_true]

    if filtered_pos_pairs:
        filtered_tst_y_pos, filtered_tst_y_preds_pos = map(list, zip(*filtered_pos_pairs))
        accuracy_pos = np.mean([np.max([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))])
        macro_recall_pos = np.mean([np.mean([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))])
        micro_recall_pos = np.sum([np.sum([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))]) / np.sum([len(a) for a in filtered_tst_y_pos])
        
        # Hardcode diversity for simple models (single prediction per sample)
        diversity_pos = 1.0
    else:
        accuracy_pos = 0.0
        macro_recall_pos = 0.0
        micro_recall_pos = 0.0
        diversity_pos = 1.0
        logger.warning("Warning: No positive samples in test set for metrics calculation")

    # Initialize results dictionary
    results = {
        'positive': {
            'accuracy': accuracy_pos,
            'macro_recall': macro_recall_pos,
            'micro_recall': micro_recall_pos,
            'diversity': diversity_pos
        },
        'negative': None
    }

    # Process negative predictions for "all" data
    if tst_y_neg and tst_y_preds_neg is not None:
        neg_pairs = zip(tst_y_neg, tst_y_preds_neg)
        filtered_neg_pairs = [(y_true, y_pred) for y_true, y_pred in neg_pairs if y_true]

        if filtered_neg_pairs:
            filtered_tst_y_neg, filtered_tst_y_preds_neg = map(list, zip(*filtered_neg_pairs))
            accuracy_neg = np.mean([np.max([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))])
            macro_recall_neg = np.mean([np.mean([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))])
            micro_recall_neg = np.sum([np.sum([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))]) / np.sum([len(a) for a in filtered_tst_y_neg])

            # Hardcode diversity for simple models (single prediction per sample)
            diversity_neg = 1.0

            results['negative'] = {
                'accuracy': accuracy_neg,
                'macro_recall': macro_recall_neg,
                'micro_recall': micro_recall_neg,
                'diversity': diversity_neg
            }

            logger.info(f'--- TEST (-) T={T} accuracy/macro-recall/micro-recall/diversity: {accuracy_neg:.4f}/{macro_recall_neg:.4f}/{micro_recall_neg:.4f}/{diversity_neg:.2f}')
        else:
            logger.warning("Warning: No negative samples in test set for metrics calculation")

    logger.info(f'--- TEST T={T} accuracy/pos_macro-recall/pos_micro-recall/diversity: {accuracy_pos:.4f}/{macro_recall_pos:.4f}/{micro_recall_pos:.4f}/{diversity_pos:.2f}')

    return results


def evaluate_vae_models(trainer, tst_loader, tst_y_pos, tst_y_neg, config, fold, wandb_is_active, logger, temperature=1.0):
    """
    Evaluate VAE models (baseline, seq, emb, seq_emb) that can produce multiple predictions.
    
    Args:
        trainer: Model trainer instance
        tst_loader: Test data loader
        tst_y_pos: Positive test labels
        tst_y_neg: Negative test labels (can be None)
        config: Configuration dictionary
        fold: Current fold number
        wandb_is_active: Whether W&B is active
        logger: Logger instance
        temperature: Temperature parameter for scaling logits (default: 1.0)
        
    Returns:
        dict: Dictionary containing evaluation metrics for all T values
    """
    if config["data_type"] == "positive":
        tst_y_preds_pos, _ = trainer.inference(tst_loader, n_sampling=1000, temperature=temperature)
        tst_y_preds_neg = None
    else:  # "all" data
        tst_y_preds_pos, tst_y_preds_neg = trainer.inference(tst_loader, n_sampling=1000, temperature=temperature)

    # Get T values from config, with fallback to default values
    T_values = config.get("T_values", [1, 10, 50, 100, 500, 1000])

    # Process positive predictions
    pos_pairs = zip(tst_y_pos, tst_y_preds_pos)
    filtered_pos_pairs = [(y_true, y_pred) for y_true, y_pred in pos_pairs if y_true]

    if filtered_pos_pairs:
        filtered_tst_y_pos, filtered_tst_y_preds_pos = map(list, zip(*filtered_pos_pairs))
    else:
        filtered_tst_y_pos, filtered_tst_y_preds_pos = [], []
        logger.warning("Warning: No positive samples in test set for metrics calculation")

    results = {'T_values': T_values, 'positive': {}, 'negative': {}}

    for T in T_values:
        if filtered_tst_y_pos:
            accuracy = np.mean([np.max([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))])
            macro_recall = np.mean([np.mean([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))])
            micro_recall = np.sum([np.sum([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))]) / np.sum([len(a) for a in filtered_tst_y_pos])

            # Calculate diversity for positive predictions
            if filtered_tst_y_preds_pos:
                # Truncate predictions to T and calculate diversity
                truncated_preds_pos = [pred[:T] for pred in filtered_tst_y_preds_pos]
                diversity_pos = np.mean(np.array([len(set(tuple(pred) for pred in i)) for i in truncated_preds_pos]))
            else:
                diversity_pos = 0.0
        else:
            accuracy = 0.0
            macro_recall = 0.0
            micro_recall = 0.0
            diversity_pos = 0.0

        results['positive'][T] = {
            'accuracy': accuracy,
            'macro_recall': macro_recall,
            'micro_recall': micro_recall,
            'diversity': diversity_pos
        }

        logger.info(f'--- TEST T={T} accuracy/pos_macro-recall/pos_micro-recall/diversity: {accuracy:.4f}/{macro_recall:.4f}/{micro_recall:.4f}/{diversity_pos:.2f}')

        # Process negative predictions only for "all" data
        if config["data_type"] == "all" and tst_y_neg and tst_y_preds_neg:
            neg_pairs = zip(tst_y_neg, tst_y_preds_neg)
            filtered_neg_pairs = [(y_true, y_pred) for y_true, y_pred in neg_pairs if y_true]

            if filtered_neg_pairs:
                filtered_tst_y_neg, filtered_tst_y_preds_neg = map(list, zip(*filtered_neg_pairs))
            else:
                filtered_tst_y_neg, filtered_tst_y_preds_neg = [], []

            accuracy_neg = np.mean([np.max([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))])
            macro_recall_neg = np.mean([np.mean([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))])
            micro_recall_neg = np.sum([np.sum([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))]) / np.sum([len(a) for a in filtered_tst_y_neg])

            # Calculate diversity for negative predictions
            if filtered_tst_y_preds_neg:
                # Truncate predictions to T and calculate diversity
                truncated_preds_neg = [pred[:T] for pred in filtered_tst_y_preds_neg]
                diversity_neg = np.mean(np.array([len(set(tuple(pred) for pred in i)) for i in truncated_preds_neg]))
            else:
                diversity_neg = 0.0

            # Calculate inter-diversity (Jaccard distance) between pos and neg prediction sets
            inter_diversity = 0.0
            if filtered_tst_y_preds_pos and filtered_tst_y_preds_neg:
                truncated_pos = [pred[:T] for pred in filtered_tst_y_preds_pos]
                truncated_neg = [pred[:T] for pred in filtered_tst_y_preds_neg]
                jaccard_distances = []
                for pos_preds, neg_preds in zip(truncated_pos, truncated_neg):
                    unique_pos = set(tuple(p) for p in pos_preds)
                    unique_neg = set(tuple(p) for p in neg_preds)
                    union = len(unique_pos.union(unique_neg))
                    inter = len(unique_pos.intersection(unique_neg))
                    jaccard_distances.append(1.0 - (inter / union) if union > 0 else 0.0)
                inter_diversity = float(np.mean(jaccard_distances)) if jaccard_distances else 0.0

            results['negative'][T] = {
                'accuracy': accuracy_neg,
                'macro_recall': macro_recall_neg,
                'micro_recall': micro_recall_neg,
                'diversity': diversity_neg,
                'inter_diversity': inter_diversity
            }

            logger.info(f'--- TEST (-) T={T} accuracy/neg_macro-recall/neg_micro-recall/diversity: {accuracy_neg:.4f}/{macro_recall_neg:.4f}/{micro_recall_neg:.4f}/{diversity_neg:.2f}')
            logger.info(f'--- TEST T={T} inter-diversity: {inter_diversity:.2f}')
    return results


def update_fold_metrics(fold_metrics, results, config, fold):
    """
    Update the fold_metrics dictionary with evaluation results.
    
    Args:
        fold_metrics: Dictionary to store metrics across folds
        results: Evaluation results from evaluate_simple_models or evaluate_vae_models
        config: Configuration dictionary
        fold: Current fold number
        
    Returns:
        dict: Updated fold_metrics dictionary
    """
    if config["model_type"] in ['rxnfp', 'baselineofbaseline']:
        # Simple models - only T=1
        T = 1
        if config["data_type"] == "positive":
            fold_metrics["accuracy_pos"][T].append(results['positive']['accuracy'])
            fold_metrics["macro_recall_pos"][T].append(results['positive']['macro_recall'])
            fold_metrics["micro_recall_pos"][T].append(results['positive']['micro_recall'])
            fold_metrics["diversity_pos"][T].append(results['positive']['diversity'])
        else:  # "all" data
            fold_metrics["accuracy_pos"][T].append(results['positive']['accuracy'])
            fold_metrics["macro_recall_pos"][T].append(results['positive']['macro_recall'])
            fold_metrics["micro_recall_pos"][T].append(results['positive']['micro_recall'])
            fold_metrics["diversity_pos"][T].append(results['positive']['diversity'])
            
            if results['negative']:
                fold_metrics["accuracy_neg"][T].append(results['negative']['accuracy'])
                fold_metrics["macro_recall_neg"][T].append(results['negative']['macro_recall'])
                fold_metrics["micro_recall_neg"][T].append(results['negative']['micro_recall'])
                fold_metrics["diversity_neg"][T].append(results['negative']['diversity'])


    elif config["model_type"] in ['baseline', 'seq', 'emb', 'seq_emb']:
        # VAE models - multiple T values
        for T in results['T_values']:
            if config["data_type"] == "positive":
                fold_metrics["accuracy_pos"][T].append(results['positive'][T]['accuracy'])
                fold_metrics["macro_recall_pos"][T].append(results['positive'][T]['macro_recall'])
                fold_metrics["micro_recall_pos"][T].append(results['positive'][T]['micro_recall'])
                fold_metrics["diversity_pos"][T].append(results['positive'][T]['diversity'])
            else:  # "all" data
                fold_metrics["accuracy_pos"][T].append(results['positive'][T]['accuracy'])
                fold_metrics["macro_recall_pos"][T].append(results['positive'][T]['macro_recall'])
                fold_metrics["micro_recall_pos"][T].append(results['positive'][T]['micro_recall'])
                fold_metrics["diversity_pos"][T].append(results['positive'][T]['diversity'])
                
                if T in results['negative']:
                    fold_metrics["accuracy_neg"][T].append(results['negative'][T]['accuracy'])
                    fold_metrics["macro_recall_neg"][T].append(results['negative'][T]['macro_recall'])
                    fold_metrics["micro_recall_neg"][T].append(results['negative'][T]['micro_recall'])
                    fold_metrics["diversity_neg"][T].append(results['negative'][T]['diversity'])
                    fold_metrics["avg_inter_diversity"][T].append(results['negative'][T]['inter_diversity'])
    return fold_metrics


def update_val_fold_metrics(val_fold_metrics, results, config, fold):
    """
    Update the val_fold_metrics dictionary with validation evaluation results.
    Similar to update_fold_metrics but for validation data used in hyperparameter optimization.
    
    Args:
        val_fold_metrics: Dictionary to store validation metrics across folds
        results: Evaluation results from evaluate_simple_models or evaluate_vae_models
        config: Configuration dictionary
        fold: Current fold number
        
    Returns:
        dict: Updated val_fold_metrics dictionary
    """
    if config["model_type"] in ['rxnfp', 'baselineofbaseline']:
        # Simple models - only T=1
        T = 1
        if config["data_type"] == "positive":
            val_fold_metrics["val_accuracy_pos"][T].append(results['positive']['accuracy'])
            val_fold_metrics["val_macro_recall_pos"][T].append(results['positive']['macro_recall'])
            val_fold_metrics["val_micro_recall_pos"][T].append(results['positive']['micro_recall'])
            val_fold_metrics["val_diversity_pos"][T].append(results['positive']['diversity'])
        else:  # "all" data
            val_fold_metrics["val_accuracy_pos"][T].append(results['positive']['accuracy'])
            val_fold_metrics["val_macro_recall_pos"][T].append(results['positive']['macro_recall'])
            val_fold_metrics["val_micro_recall_pos"][T].append(results['positive']['micro_recall'])
            val_fold_metrics["val_diversity_pos"][T].append(results['positive']['diversity'])
            
            if results['negative']:
                val_fold_metrics["val_accuracy_neg"][T].append(results['negative']['accuracy'])
                val_fold_metrics["val_macro_recall_neg"][T].append(results['negative']['macro_recall'])
                val_fold_metrics["val_micro_recall_neg"][T].append(results['negative']['micro_recall'])
                val_fold_metrics["val_diversity_neg"][T].append(results['negative']['diversity'])

    
    elif config["model_type"] in ['baseline', 'seq', 'emb', 'seq_emb']:
        # VAE models - multiple T values
        for T in results['T_values']:
            if config["data_type"] == "positive":
                val_fold_metrics["val_accuracy_pos"][T].append(results['positive'][T]['accuracy'])
                val_fold_metrics["val_macro_recall_pos"][T].append(results['positive'][T]['macro_recall'])
                val_fold_metrics["val_micro_recall_pos"][T].append(results['positive'][T]['micro_recall'])
                val_fold_metrics["val_diversity_pos"][T].append(results['positive'][T]['diversity'])
            else:  # "all" data
                val_fold_metrics["val_accuracy_pos"][T].append(results['positive'][T]['accuracy'])
                val_fold_metrics["val_macro_recall_pos"][T].append(results['positive'][T]['macro_recall'])
                val_fold_metrics["val_micro_recall_pos"][T].append(results['positive'][T]['micro_recall'])
                val_fold_metrics["val_diversity_pos"][T].append(results['positive'][T]['diversity'])
                
                if T in results['negative']:
                    val_fold_metrics["val_accuracy_neg"][T].append(results['negative'][T]['accuracy'])
                    val_fold_metrics["val_macro_recall_neg"][T].append(results['negative'][T]['macro_recall'])
                    val_fold_metrics["val_micro_recall_neg"][T].append(results['negative'][T]['micro_recall'])
                    val_fold_metrics["val_diversity_neg"][T].append(results['negative'][T]['diversity'])
                    val_fold_metrics["val_avg_inter_diversity"][T].append(results['negative'][T]['inter_diversity'])
    return val_fold_metrics


def evaluate_model(trainer, tst_loader, tst_y_pos, tst_y_neg, config, fold, wandb_is_active, logger, temperature=1.0):
    """
    Main evaluation function that routes to appropriate evaluation method based on model type.
    
    Args:
        trainer: Model trainer instance
        tst_loader: Test data loader
        tst_y_pos: Positive test labels
        tst_y_neg: Negative test labels (can be None)
        config: Configuration dictionary
        fold: Current fold number
        wandb_is_active: Whether W&B is active
        logger: Logger instance
        temperature: Temperature parameter for scaling logits (default: 1.0)
        
    Returns:
        dict: Dictionary containing evaluation results
    """
    if config["model_type"] in ['rxnfp', 'baselineofbaseline']:
        return evaluate_simple_models(trainer, tst_loader, tst_y_pos, tst_y_neg, config, fold, wandb_is_active, logger, temperature)
    elif config["model_type"] in ['baseline', 'seq', 'emb', 'seq_emb']:
        return evaluate_vae_models(trainer, tst_loader, tst_y_pos, tst_y_neg, config, fold, wandb_is_active, logger, temperature)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

def evaluate_model_and_get_preds(trainer, tst_loader, tst_y_pos, tst_y_neg, config, logger):
    """
    Evaluate VAE models and return metrics and predictions.
    Adapted from `evaluate_vae_models` in `utils/evaluate_model.py`.
    """
    n_sampling = config.get('n_sampling_test', 1000)
    temperature = config.get('temperature', 1.0)

    if config["data_type"] == "positive":
        tst_y_preds_pos, _ = trainer.inference(tst_loader, n_sampling=n_sampling, temperature=temperature)
        tst_y_preds_neg = None
    else:
        tst_y_preds_pos, tst_y_preds_neg = trainer.inference(tst_loader, n_sampling=n_sampling, temperature=temperature)

    T_values = config.get("T_values", [1, 10, 50, 100, 500, 1000])
    
    pos_pairs = zip(tst_y_pos, tst_y_preds_pos)
    filtered_pos_pairs = [(y_true, y_pred) for y_true, y_pred in pos_pairs if y_true]

    if filtered_pos_pairs:
        filtered_tst_y_pos, filtered_tst_y_preds_pos = map(list, zip(*filtered_pos_pairs))
    else:
        filtered_tst_y_pos, filtered_tst_y_preds_pos = [], []
        logger.warning("Warning: No positive samples in test set for metrics calculation")

    results = {'T_values': T_values, 'positive': {}, 'negative': {}}

    for T in T_values:
        if filtered_tst_y_pos:
            accuracy = np.mean([np.max([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))])
            macro_recall = np.mean([np.mean([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))])
            micro_recall = np.sum([np.sum([(c in filtered_tst_y_preds_pos[i][:T]) for c in filtered_tst_y_pos[i]]) for i in range(len(filtered_tst_y_pos))]) / np.sum([len(a) for a in filtered_tst_y_pos])
            diversity_pos = np.mean(np.array([len(set(tuple(pred) for pred in i[:T])) for i in filtered_tst_y_preds_pos])) if filtered_tst_y_preds_pos else 0.0
        else:
            accuracy, macro_recall, micro_recall, diversity_pos = 0.0, 0.0, 0.0, 0.0

        results['positive'][T] = {'accuracy': accuracy, 'macro_recall': macro_recall, 'micro_recall': micro_recall, 'diversity': diversity_pos}
        logger.info(f'--- TEST T={T} accuracy/pos_macro-recall/pos_micro-recall/diversity: {accuracy:.4f}/{macro_recall:.4f}/{micro_recall:.4f}/{diversity_pos:.2f}')

        if config["data_type"] == "all" and tst_y_neg and tst_y_preds_neg:
            neg_pairs = zip(tst_y_neg, tst_y_preds_neg)
            filtered_neg_pairs = [(y_true, y_pred) for y_true, y_pred in neg_pairs if y_true]

            if filtered_neg_pairs:
                filtered_tst_y_neg, filtered_tst_y_preds_neg = map(list, zip(*filtered_neg_pairs))
                accuracy_neg = np.mean([np.max([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))])
                macro_recall_neg = np.mean([np.mean([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))])
                micro_recall_neg = np.sum([np.sum([(c in filtered_tst_y_preds_neg[i][:T]) for c in filtered_tst_y_neg[i]]) for i in range(len(filtered_tst_y_neg))]) / np.sum([len(a) for a in filtered_tst_y_neg])
                diversity_neg = np.mean(np.array([len(set(tuple(pred) for pred in i[:T])) for i in filtered_tst_y_preds_neg])) if filtered_tst_y_preds_neg else 0.0
                # Compute inter-diversity (Jaccard distance) between pos and neg prediction sets
                inter_diversity = 0.0
                if filtered_tst_y_preds_pos and filtered_tst_y_preds_neg:
                    truncated_pos = [pred[:T] for pred in filtered_tst_y_preds_pos]
                    truncated_neg = [pred[:T] for pred in filtered_tst_y_preds_neg]
                    jaccard_distances = []
                    for pos_preds, neg_preds in zip(truncated_pos, truncated_neg):
                        unique_pos = set(tuple(p) for p in pos_preds)
                        unique_neg = set(tuple(p) for p in neg_preds)
                        union = len(unique_pos.union(unique_neg))
                        inter = len(unique_pos.intersection(unique_neg))
                        jaccard_distances.append(1.0 - (inter / union) if union > 0 else 0.0)
                    inter_diversity = float(np.mean(jaccard_distances)) if jaccard_distances else 0.0

                results['negative'][T] = {
                    'accuracy': accuracy_neg,
                    'macro_recall': macro_recall_neg,
                    'micro_recall': micro_recall_neg,
                    'diversity': diversity_neg,
                    'inter_diversity': inter_diversity
                }
                logger.info(f'--- TEST (-) T={T} accuracy/neg_macro-recall/neg_micro-recall/diversity: {accuracy_neg:.4f}/{macro_recall_neg:.4f}/{micro_recall_neg:.4f}/{diversity_neg:.2f}')

    results['positive_predictions'] = tst_y_preds_pos
    results['negative_predictions'] = tst_y_preds_neg
    return results
