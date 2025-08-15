import numpy as np
import pandas as pd
import pickle as pkl
import os
from typing import Tuple, Dict, List, Optional
import warnings
import json
from datetime import datetime
from tqdm import tqdm

# XGBoost and scikit-learn imports
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import joblib
from xgboost.callback import EarlyStopping

# Optional imports
try:
    import wandb
except ImportError:
    wandb = None
    print("Warning: WandB not available, logging will be disabled")



try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: Matplotlib not available, plotting will be disabled")


class XGBoostYieldRegressor:
    """
    XGBoost model for reaction yield prediction with uncertainty estimation.
    
    Implements:
    1. Point prediction using standard XGBoost regression
    2. Uncertainty estimation using quantile regression (confidence intervals)
    3. Ensemble methods for additional uncertainty quantification
    
    Args:
        quantiles (List[float]): Quantiles to predict for uncertainty estimation
        n_ensemble (int): Number of models in ensemble for uncertainty
        scale_features (bool): Whether to standardize input features
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        n_ensemble: int = 5,
        scale_features: bool = True,
        random_state: int = 42,
        verbose: bool = True,
        config: Optional[Dict] = None
    ):
        self.quantiles = sorted(quantiles)
        self.n_ensemble = n_ensemble
        self.scale_features = scale_features
        self.random_state = random_state
        self.verbose = verbose
        self.config = config
        
        # Model storage
        self.models = {}  # Dictionary to store different model types
        self.scaler = StandardScaler() if scale_features else None
        self.is_fitted = False
        
        # Training history
        self.training_history = {}
        
        # Feature importance
        self.feature_importance = None
        self.feature_names = None
        
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = False):
        """Prepare data by scaling if requested."""
        if self.scale_features:
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
            
        return X_scaled, y
    
    def _train_quantile_regressors(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        sample_weight: Optional[np.ndarray] = None,
        use_early_stopping: bool = True,
        **xgb_params
    ):
        """Train XGBoost quantile regressors for uncertainty estimation."""
        
        quantile_models = {}
        
        with tqdm(self.quantiles, desc="Training Quantile Regressors") as pbar:
            for quantile in pbar:
                pbar.set_description(f"Training Quantile q={quantile:.1f}")
                
                # Setup callbacks for early stopping
                callbacks = [EarlyStopping(rounds=50, save_best=True)] if use_early_stopping and X_val is not None else None

                # Only set essential parameters; all others come from xgb_params
                params = {
                    'objective': 'reg:quantileerror',
                    'quantile_alpha': quantile,
                    'random_state': self.random_state + int(quantile * 100),
                    'n_jobs': -1,
                    'eval_metric': 'rmse',
                    'callbacks': callbacks
                }
                params.update(xgb_params)

                print("Parameters set for quantile estimator: ", params)

                model = xgb.XGBRegressor(**{k: v for k, v in params.items() if v is not None})
                
                # Prepare validation data
                eval_set = [(X_train, y_train)]
                sample_weight_eval_set = [sample_weight] if sample_weight is not None else None
                if X_val is not None and y_val is not None:
                    eval_set.append((X_val, y_val))
                    if sample_weight_eval_set is not None:
                        sample_weight_eval_set.append(None) # Do not weight validation data
                
                # Train
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weight,
                    eval_set=eval_set,
                    sample_weight_eval_set=sample_weight_eval_set,
                    verbose=False
                )

                if self.verbose and use_early_stopping:
                    print(f"Quantile estimator for q={quantile:.1f} trained with {model.best_iteration} trees")
                
                quantile_models[quantile] = model
        
        self.models['quantile'] = quantile_models
        if self.verbose:
            print(f"Trained {len(quantile_models)} quantile regressors")
    
    def _train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        sample_weight: Optional[np.ndarray] = None,
        use_early_stopping: bool = True,
        **xgb_params
    ):
        """Train an ensemble of strong models using the best hyperparameters."""

        ensemble_models = []
        
        # Use tqdm for a progress bar during training
        for i in tqdm(range(self.n_ensemble), desc="Training Ensemble Models"):
            # 1. Start with the best parameters from the hyperparameter search
            params = xgb_params.copy()
            
            # 2. Set a unique random state for each model to ensure stochastic diversity
            params['random_state'] = self.random_state + i
            
            # 3. Setup early stopping for each individual model
            callbacks = [EarlyStopping(rounds=50, save_best=True)] if use_early_stopping and X_val is not None else None
            params['callbacks'] = callbacks

            print(f"Parameters set for ensemble estimator {i}: ", params)

            model = xgb.XGBRegressor(**{k: v for k, v in params.items() if v is not None})
            
            # 4. Bootstrap sampling: Train each model on a different sample of the data
            n_samples = len(X_train)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X_train[bootstrap_idx]
            y_bootstrap = y_train[bootstrap_idx]
            
            bootstrap_weights = sample_weight[bootstrap_idx] if sample_weight is not None else None
            
            # 5. Prepare the evaluation set for early stopping
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_bootstrap, y_bootstrap), (X_val, y_val)]

            # 6. Train the model on the bootstrapped sample
            model.fit(
                X_bootstrap,
                y_bootstrap,
                sample_weight=bootstrap_weights,
                eval_set=eval_set,
                verbose=False # Keep the output clean
            )

            # To save memory, you can remove the fitted callback object after training
            model.callbacks = None

            if self.verbose and use_early_stopping:
                print(f"Ensemble estimator {i} trained with {model.best_iteration} trees")

            ensemble_models.append(model)
        
        self.models['ensemble'] = ensemble_models
        if self.verbose:
            print(f"Trained ensemble of {len(ensemble_models)} models.")
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
        sample_weight: Optional[np.ndarray] = None,
        use_early_stopping: bool = True,
        **xgb_params
    ):
        """
        Fit the XGBoost yield regression model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets (yields)
            X_val (np.ndarray, optional): Validation features
            y_val (np.ndarray, optional): Validation targets
            feature_names (List[str], optional): Names of features
            sample_weight (np.ndarray, optional): Weights for training samples
            use_early_stopping (bool): Whether to use early stopping
            **xgb_params: Additional XGBoost parameters
        """
        if self.verbose:
            print("Fitting XGBoost Yield Regression Model")
            print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            if X_val is not None:
                print(f"Validation set: {X_val.shape[0]} samples")
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Prepare data
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train, fit_scaler=True)
        if X_val is not None:
            X_val_prep, y_val_prep = self._prepare_data(X_val, y_val, fit_scaler=False)
        else:
            X_val_prep, y_val_prep = None, None
        
        # Train different model types
        start_time = datetime.now()
        
        # Train Quantile regressors
        if self.verbose:
            print("\n1. Training quantile regressors...")
        self._train_quantile_regressors(X_train_prep, y_train_prep, X_val_prep, y_val_prep, sample_weight=sample_weight, use_early_stopping=use_early_stopping, **xgb_params)
        
        # Train Ensemble
        if self.verbose:
            print("\n2. Training ensemble...")
        self._train_ensemble(X_train_prep, y_train_prep, X_val_prep, y_val_prep, sample_weight=sample_weight, use_early_stopping=use_early_stopping, **xgb_params)
        
        training_time = datetime.now() - start_time
        
        # Store feature importance from the median quantile estimator
        if 0.5 in self.models.get('quantile', {}):
            self.feature_importance = self.models['quantile'][0.5].feature_importances_
        
        # Store training info
        self.training_history = {
            'training_time': training_time.total_seconds(),
            'n_train': len(X_train),
            'n_val': len(X_val) if X_val is not None else 0,
            'n_features': X_train.shape[1],
            'quantiles': self.quantiles,
            'n_ensemble': self.n_ensemble
        }
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"\nTraining completed in {training_time.total_seconds():.1f} seconds")
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = True) -> Dict:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X (np.ndarray): Input features
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            Dict: Dictionary containing predictions and uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_prep, _ = self._prepare_data(X, fit_scaler=False)
        
        results = {}
        
        # Point predictions are now taken from the median quantile
        if 0.5 in self.quantiles:
            results['point_prediction'] = self.models['quantile'][0.5].predict(X_prep)
        
        if return_uncertainty:
            # Quantile predictions
            quantile_predictions = {}
            for quantile in self.quantiles:
                y_pred_q = self.models['quantile'][quantile].predict(X_prep)
                quantile_predictions[f'q{quantile:.1f}'] = y_pred_q
            results['quantile_predictions'] = quantile_predictions
            
            # Ensemble predictions
            ensemble_predictions = []
            for model in self.models['ensemble']:
                y_pred_ens = model.predict(X_prep)
                ensemble_predictions.append(y_pred_ens)
            
            ensemble_predictions = np.array(ensemble_predictions)
            ensemble_mean = np.mean(ensemble_predictions, axis=0)
            ensemble_std = np.std(ensemble_predictions, axis=0)
            
            results['ensemble_mean'] = ensemble_mean
            results['ensemble_std'] = ensemble_std
            results['ensemble_predictions'] = ensemble_predictions
            
            # Calculate confidence intervals from quantiles
            if 0.1 in self.quantiles and 0.9 in self.quantiles:
                results['confidence_interval_80'] = {
                    'lower': quantile_predictions['q0.1'],
                    'upper': quantile_predictions['q0.9']
                }
            
            if 0.05 in self.quantiles and 0.95 in self.quantiles:
                results['confidence_interval_90'] = {
                    'lower': quantile_predictions['q0.05'],
                    'upper': quantile_predictions['q0.95']
                }
        
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, transform_target: bool = False) -> Dict:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            transform_target (bool): Whether to inverse transform predictions
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        predictions = self.predict(X_test, return_uncertainty=True)
        # Use median quantile as point prediction, fall back to ensemble mean if not available
        y_pred = predictions.get('point_prediction', predictions.get('ensemble_mean'))
        
        # Inverse transform predictions if the target was transformed
        if transform_target:
            y_pred = np.expm1(y_pred)
            # We must also handle the case of negative predictions from the model before inverse transform
            y_pred[y_pred < 0] = 0

        # Basic regression metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'n_test': len(y_test)
        }
        
        # Uncertainty evaluation
        if 'ensemble_std' in predictions:
            # Calculate calibration metrics
            ensemble_std = predictions['ensemble_std']
            residuals = np.abs(y_test - y_pred)
            
            # Correlation between uncertainty and error
            uncertainty_error_corr = np.corrcoef(ensemble_std, residuals)[0, 1]
            metrics['uncertainty_error_correlation'] = uncertainty_error_corr
            
            # Average uncertainty
            metrics['mean_uncertainty'] = np.mean(ensemble_std)
            metrics['std_uncertainty'] = np.std(ensemble_std)
        
        # Quantile evaluation
        if 'quantile_predictions' in predictions:
            quantile_metrics = {}
            for quantile_name, y_pred_q in predictions['quantile_predictions'].items():
                # Coverage probability (for confidence intervals)
                if quantile_name == 'q0.1':
                    coverage_below = np.mean(y_test <= y_pred_q)
                    quantile_metrics[f'{quantile_name}_coverage'] = coverage_below
                elif quantile_name == 'q0.9':
                    coverage_above = np.mean(y_test >= y_pred_q)
                    quantile_metrics[f'{quantile_name}_coverage'] = coverage_above
                
                # Quantile loss
                quantile_val = float(quantile_name[1:])
                quantile_loss = self._quantile_loss(y_test, y_pred_q, quantile_val)
                quantile_metrics[f'{quantile_name}_loss'] = quantile_loss
            
            metrics['quantile_metrics'] = quantile_metrics
        
        return metrics
    
    def _quantile_loss(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """Calculate quantile loss."""
        errors = y_true - y_pred
        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from the median quantile estimator.
        
        Args:
            importance_type (str): Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Use the median (0.5) quantile model for feature importance
        if 0.5 not in self.models.get('quantile', {}):
            raise ValueError("Median quantile (0.5) model not trained, cannot get feature importance.")
        
        importance_model = self.models['quantile'][0.5]
        
        # Use a more robust importance calculation by querying the booster directly.
        # 'total_gain' is a reliable metric for feature contribution.
        booster = importance_model.get_booster()
        importance_scores = booster.get_score(importance_type='total_gain')

        if not importance_scores:
            warnings.warn("Feature importance scores are empty. The model may be too simple.")
            return pd.DataFrame({'feature': self.feature_names, 'importance': 0})
        
        # Create a dataframe from the scores
        df_importance = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        })

        # The booster uses generic feature names (f0, f1, ...), so we need to map them back
        # to the original names.
        # First, create a mapping from f-string to original name
        f_to_name_map = {f'f{i}': name for i, name in enumerate(self.feature_names)}
        
        # Apply the mapping
        df_importance['feature'] = df_importance['feature'].map(f_to_name_map)
        
        # Normalize the importance scores to sum to 1
        total_importance = df_importance['importance'].sum()
        if total_importance > 0:
            df_importance['importance'] /= total_importance
            
        return df_importance.sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """Save the entire model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'quantiles': self.quantiles,
            'n_ensemble': self.n_ensemble,
            'scale_features': self.scale_features,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pkl.dump(model_data, f)
        
        if self.verbose:
            print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pkl.load(f)
        
        # Restore all attributes
        for key, value in model_data.items():
            setattr(self, key, value)
        
        if self.verbose:
            print(f"Model loaded from: {filepath}")
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None, transform_target: bool = False):
        """Plot predictions vs actual values with uncertainty using Matplotlib."""
        if plt is None:
            print("Matplotlib not available, cannot create plots.")
            return

        predictions = self.predict(X_test, return_uncertainty=True)
        y_pred = predictions.get('point_prediction', predictions.get('ensemble_mean'))

        # Inverse transform if necessary
        if transform_target:
            y_pred = np.expm1(y_pred)
            y_pred[y_pred < 0] = 0
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- Create plot title with config info ---
        title = "Model Performance Evaluation"
        if self.config:
            split = self.config.get('split_strategy', 'N/A')
            rep = self.config.get('representation', 'N/A')
            cond_rep = self.config.get('condition_representation', 'N/A')
            title += f"\nSplit: {split} | Representation: {rep} | Condition Rep: {cond_rep}"
        
        fig.suptitle(title, fontsize=14)

        # --- Predicted vs Actual Plot ---
        r2 = r2_score(y_test, y_pred)
        
        # Scatter plot for predictions
        axes[0].scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
        
        # Add diagonal line
        line_range = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[0].plot(line_range, line_range, color='red', linestyle='--', label='Ideal')

        # Add R² annotation
        axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[0].set_title('Predicted vs Actual Yields')
        axes[0].set_xlabel('Actual Yield')
        axes[0].set_ylabel('Predicted Yield')
        axes[0].legend()
        axes[0].grid(True)
        
        # --- Residuals Plot ---
        residuals = y_test - y_pred
        
        axes[1].scatter(y_pred, residuals, color='blue', alpha=0.6, label='Residuals')
        
        # Add horizontal line at y=0
        axes[1].axhline(y=0, color='red', linestyle='--')
        
        # Add uncertainty bounds if available
        if 'ensemble_std' in predictions:
            ensemble_std = predictions['ensemble_std']
            # Sort values for a clean fill
            sorted_indices = np.argsort(y_pred)
            sorted_y_pred = y_pred[sorted_indices]
            sorted_ensemble_std = ensemble_std[sorted_indices]
            
            upper_bound = 2 * sorted_ensemble_std
            lower_bound = -2 * sorted_ensemble_std
            
            axes[1].fill_between(sorted_y_pred, lower_bound, upper_bound,
                                 color='gray', alpha=0.2, label='±2σ uncertainty')

        axes[1].set_title('Residuals vs Predicted')
        axes[1].set_xlabel('Predicted Yield')
        axes[1].set_ylabel('Residuals')
        axes[1].legend()
        axes[1].grid(True)
        
        # Layout adjustment
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

        if save_path:
            # Ensure the output directory exists
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Change extension to .png for static plot
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                save_path = os.path.splitext(save_path)[0] + '.png'
            
            plt.savefig(save_path)
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        
        plt.show()


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict,
    cv_folds: int = 5,
    scoring: str = 'neg_root_mean_squared_error',
    random_state: int = 42,
    verbose: bool = True,
    config: Dict = None
) -> Dict:
    """
    Perform a robust hyperparameter search for XGBoost without internal early stopping
    to avoid API conflicts with scikit-learn search tools.
    """
    
    # --- Stratified K-Fold for Regression (This logic is good, no changes needed) ---
    n_splits = cv_folds
    cv_splits = None
    try:
        num_bins = 10
        y_binned = pd.qcut(y_train, q=num_bins, labels=False, duplicates='drop')
        bin_counts = np.bincount(y_binned)
        if len(np.unique(y_binned)) > 1 and np.all(bin_counts >= n_splits):
            if verbose:
                print(f"Using stratified cross-validation with {n_splits} splits.")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_splits = list(skf.split(X_train, y_binned))
    except ValueError as e:
        if verbose:
            warnings.warn(f"Binning for stratified CV failed: {e}. Falling back to regular KFold.")

    if cv_splits is None:
        if verbose:
            print(f"Using regular KFold cross-validation with {n_splits} splits and shuffle.")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = list(kf.split(X_train))

    # --- Model and Search Setup ---

    # Base model is simple. n_estimators will be provided by the param_grid.
    # NO early stopping parameters are defined here.
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=-1
    )

    if verbose:
        print("Starting hyperparameter search...")

    # We use a standard GridSearchCV or HalvingRandomSearchCV.
    # It is robust and avoids the API conflicts you were seeing.
    search_tool = None
    if config.get('hyperparameter_search_type', 'grid') == 'grid':
        search_tool = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,  # This grid MUST include 'n_estimators'
            cv=cv_splits,
            scoring=scoring,
            n_jobs=-1,
            verbose=3 if verbose else 0
        )
    elif config['hyperparameter_search_type'] == 'halving':
        search_tool = HalvingRandomSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_candidates=300,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=-1,
            verbose=3 if verbose else 0
        )

    # --- Fitting ---
    
    # Fit the search tool without any extra parameters. This is the key to avoiding the errors.
    search_tool.fit(X_train, y_train)
    
    # --- Results ---
    
    # We only need the best parameters and the score from this step.
    # The 'best_iteration' is not relevant here; it will be found in the final training.
    results = {
        'best_params': search_tool.best_params_,
        'best_score': search_tool.best_score_,
    }

    if verbose:
        print(f"Best parameters found by search: {results['best_params']}")
        print(f"Best CV score: {results['best_score']:.4f}")
    
    return results
