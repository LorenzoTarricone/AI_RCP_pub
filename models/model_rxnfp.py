import numpy as np
import time
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss
from scipy.special import expit

import wandb

class FNN(nn.Module):

    def __init__(self, in_feats, n_classes,
                 predict_hidden_feats = 1024,
                 dropout_rate = 0.1):
        
        super(FNN, self).__init__()

        self.predict = nn.Sequential(
            nn.Linear(in_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(dropout_rate),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(dropout_rate),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(dropout_rate),
            nn.Linear(predict_hidden_feats, n_classes)
        )
                    
    def forward(self, inp):

        out = self.predict(inp)

        return out


class Trainer:

    def __init__(self, net, cuda, config, wandb_project="ReactionVAE_Baseline_rxnfp"):
    
        self.net = net.to(cuda)
        self.n_classes = config["n_classes"]
        self.rmol_max_cnt = config["rmol_max_cnt"]
        self.pmol_max_cnt = config["pmol_max_cnt"]
        self.batch_size = config["batch_size"]
        self.model_path = config["model_path"]
        self.cuda = cuda
        self.wandb_project = wandb_project
        self.config = config

        # Initialize wandb (ensure it only happens once if needed)
        if self.config["wandb"] and wandb.run is None:
            wandb.init(project=self.wandb_project, config={
                 "batch_size": self.batch_size,
                 "model_path": self.model_path,
                 "n_classes": self.n_classes,
                 # Add other relevant hyperparameters?
             })


    def load(self):

        try:
            self.net.load_state_dict(torch.load(self.model_path, map_location=self.cuda))
            print(f"INFO: Model loaded successfully from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False 
        
    def _calculate_class_weights(self, pos_count_matrix_1, pos_count_matrix_0, neg_count_matrix_1, neg_count_matrix_0):
        """
        Calculate class weights for positive and negative samples separately, matching the baseline logic.
        Args:
            pos_count_matrix_1: numpy array of shape (num_samples, num_classes) for positive samples (1s)
            pos_count_matrix_0: numpy array of shape (num_samples, num_classes) for positive samples (0s)
            neg_count_matrix_1: numpy array of shape (num_samples, num_classes) for negative samples (1s)
            neg_count_matrix_0: numpy array of shape (num_samples, num_classes) for negative samples (0s)
        Returns:
            tuple: (pos_weights, neg_weights) - both torch.Tensor of shape (num_classes,)
        """
        def _get_weights(count_matrix_1, count_matrix_0):
            if count_matrix_1 is None or count_matrix_1.shape[0] == 0:
                return None
            pos_count = count_matrix_1.sum(axis=0)
            neg_count = count_matrix_0.sum(axis=0)
            ratio = neg_count / (pos_count + 1e-8)
            ratio = np.clip(ratio, 0.1, 10.0)
            return torch.tensor(ratio, dtype=torch.float32, device=self.cuda)
        pos_weights = _get_weights(pos_count_matrix_1, pos_count_matrix_0)
        neg_weights = _get_weights(neg_count_matrix_1, neg_count_matrix_0)
        return pos_weights, neg_weights
               
    def training(self, train_loader, val_loader, max_epochs = 500):
        # Setup loss functions
        loss_fns = {}
        if self.config["class_weights"]:
            assert self.config["train_pos_count_matrix_1"] is not None, "train_pos_count_matrix_1 is not set"
            assert self.config["train_pos_count_matrix_0"] is not None, "train_pos_count_matrix_0 is not set"
            if self.config["data_type"] == "all":
                assert self.config["train_neg_count_matrix_1"] is not None, "train_neg_count_matrix_1 is not set"
                assert self.config["train_neg_count_matrix_0"] is not None, "train_neg_count_matrix_0 is not set"
            pos_weights, neg_weights = self._calculate_class_weights(
                self.config["train_pos_count_matrix_1"],
                self.config["train_pos_count_matrix_0"],
                self.config["train_neg_count_matrix_1"] if self.config["data_type"] == "all" else None,
                self.config["train_neg_count_matrix_0"] if self.config["data_type"] == "all" else None
            )
            if pos_weights is not None:
                loss_fns['pos'] = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
            if neg_weights is not None:
                loss_fns['neg'] = nn.BCEWithLogitsLoss(pos_weight=neg_weights, reduction='none')
        # Always have unweighted fallback
        loss_fns['unweighted'] = nn.BCEWithLogitsLoss(reduction='none')
        
        # Ensure lr is float (WandB sometimes passes it as string)
        lr = float(self.config["lr"])
        optimizer = Adam(self.net.parameters(), lr=lr, weight_decay=1e-10)
        #patience as 20% of max_epochs
        patience = int(max_epochs * 0.2)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-6)
    
        train_size = train_loader.dataset.__len__()
        val_size = val_loader.dataset.__len__()
        
        # Get validation labels for metrics calculation
        val_y = val_loader.dataset.y
        # Create two val_y for positive and negative conditions
        val_y_pos = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in val_y]
        val_y_neg = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 0] for sublist in val_y] if self.config["data_type"] == "all" else None
        
        # Filter out empty lists for initial best metrics calculation
        filtered_val_y_pos = [y for y in val_y_pos if y]
        best_macro_recall_pos = np.mean([1/len(y) for y in filtered_val_y_pos]) if filtered_val_y_pos else 0.0
        best_micro_recall_pos = len(filtered_val_y_pos) / np.sum([len(a) for a in filtered_val_y_pos]) if filtered_val_y_pos else 0.0
        
        if self.config["data_type"] == "all" and val_y_neg:
            filtered_val_y_neg = [y for y in val_y_neg if y]
            best_macro_recall_neg = np.mean([1/len(y) for y in filtered_val_y_neg]) if filtered_val_y_neg else 0.0
            best_micro_recall_neg = len(filtered_val_y_neg) / np.sum([len(a) for a in filtered_val_y_neg]) if filtered_val_y_neg else 0.0
        else:
            best_macro_recall_neg = 0.0
            best_micro_recall_neg = 0.0
        
        val_log = np.zeros(max_epochs)
        best_model_epoch = -1  # Track the epoch of the best model
        
        for epoch in range(max_epochs):
            # Training
            self.net.train()
            start_time = time.time()
            grad_norm_list = []
            train_loss_epoch = 0.
            
            for batchidx, batchdata in enumerate(train_loader):
                if batchdata is None:
                    continue
                    
                inputs = batchdata[0].to(self.cuda)
                #check if inputs is all zeroes
                if torch.all(inputs == 0):
                    print(f"WARNING: All inputs are zero at epoch {epoch}, batch {batchidx}")
                labels = batchdata[1].to(self.cuda)
                pos_neg_reac = batchdata[2].to(self.cuda).squeeze(1)
                inputs = torch.cat([inputs, pos_neg_reac], dim=1)

                
                preds = self.net(inputs)
                
                # Clamp the predictions to prevent BCE explosion
                preds = torch.clamp(preds, -10, 10)
                
                # Calculate loss using appropriate weights based on pos_neg_reac
                if self.config["class_weights"]:
                    # Split batch by positive/negative samples
                    pos_mask = (pos_neg_reac == 1).squeeze(-1)
                    neg_mask = (pos_neg_reac == 0).squeeze(-1)
                    
                    # Calculate losses separately for better numerical stability
                    total_loss_bce = 0.0
                    total_samples = 0
                    
                    if pos_mask.any() and 'pos' in loss_fns:
                        pos_loss_per_sample = loss_fns['pos'](preds[pos_mask], labels[pos_mask]).sum(dim=1)
                        pos_loss = pos_loss_per_sample.mean()
                        total_loss_bce += pos_loss * pos_mask.sum().float()
                        total_samples += pos_mask.sum().float()
                        
                    if neg_mask.any() and 'neg' in loss_fns:
                        neg_loss_per_sample = loss_fns['neg'](preds[neg_mask], labels[neg_mask]).sum(dim=1)
                        neg_loss = neg_loss_per_sample.mean()
                        total_loss_bce += neg_loss * neg_mask.sum().float()
                        total_samples += neg_mask.sum().float()
                    
                    if total_samples > 0:
                        loss = total_loss_bce / total_samples
                    else:
                        # Fallback to unweighted loss
                        loss = loss_fns['unweighted'](preds, labels).sum(dim=1).mean()
                else:
                    loss = loss_fns['unweighted'](preds, labels).sum(dim=1).mean()
                
                # Add safety check for negative or invalid BCE loss
                if loss < 0 or torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Invalid BCE loss detected: {loss.item()}. Using unweighted fallback.")
                    loss = loss_fns['unweighted'](preds, labels).sum(dim=1).mean()
                
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN loss and skip if found
                if torch.isnan(loss):
                    print(f"WARNING: NaN loss detected at epoch {epoch}, batch {batchidx}. Skipping this batch.")
                    continue
                    
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                grad_norm_list.append(grad_norm.cpu().numpy())
                optimizer.step()
                
                train_loss_epoch += loss.detach().item()
            
            train_loss_epoch /= len(train_loader)
            
            print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f, max gradient norm %.3f'
                  % (epoch, optimizer.param_groups[0]['lr'], train_size, train_size, train_loss_epoch, 
                     (time.time()-start_time)/60, np.max(grad_norm_list)))
            
            # Validation
            start_time = time.time()
            val_y_preds_pos, val_y_preds_neg = self.inference(val_loader, temperature=self.config["temperature"])

            if self.config["verbose"]:
                print(f"y pred pos validation at epoch {epoch}  ", val_y_preds_pos[:10])
                print(f"y true pos validation at epoch {epoch}  ", [i[:3] for i in val_y_pos[:10]])
                if self.config["data_type"] == "all":
                    print(f"y pred neg validation at epoch {epoch}  ", val_y_preds_neg[:10])
                    print(f"y true neg validation at epoch {epoch}  ", [i[:3] for i in val_y_neg[:10]])
            
            # Process positive predictions
            pos_pairs = zip(val_y_pos, val_y_preds_pos)
            filtered_pos_pairs = [(y_true, y_pred) for y_true, y_pred in pos_pairs if y_true]
            
            if filtered_pos_pairs:
                filtered_val_y_pos, filtered_val_y_preds_pos = map(list, zip(*filtered_pos_pairs))
            else:
                filtered_val_y_pos, filtered_val_y_preds_pos = [], []
            
            # Calculate metrics for positive samples
            accuracy_pos = np.mean([np.max([(c in filtered_val_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) if filtered_val_y_pos else 0
            macro_recall_pos = np.mean([np.mean([(c in filtered_val_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) if filtered_val_y_pos else 0
            micro_recall_pos = np.sum([np.sum([(c in filtered_val_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) / np.sum([len(a) for a in filtered_val_y_pos]) if filtered_val_y_pos else 0
            
            # Calculate metrics for negative samples only if data_type is "all"
            if self.config["data_type"] == "all" and val_y_neg:
                neg_pairs = zip(val_y_neg, val_y_preds_neg)
                filtered_neg_pairs = [(y_true, y_pred) for y_true, y_pred in neg_pairs if y_true]
                
                if filtered_neg_pairs:
                    filtered_val_y_neg, filtered_val_y_preds_neg = map(list, zip(*filtered_neg_pairs))
                else:
                    filtered_val_y_neg, filtered_val_y_preds_neg = [], []
                
                accuracy_neg = np.mean([np.max([(c in filtered_val_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) if filtered_val_y_neg else 0
                macro_recall_neg = np.mean([np.mean([(c in filtered_val_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) if filtered_val_y_neg else 0
                micro_recall_neg = np.sum([np.sum([(c in filtered_val_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) / np.sum([len(a) for a in filtered_val_y_neg]) if filtered_val_y_neg else 0
            else:
                accuracy_neg = np.nan
                macro_recall_neg = np.nan
                micro_recall_neg = np.nan
            
            if self.config["data_type"] == "all":
                # Calculate overall validation loss
                val_loss = 1 - (macro_recall_pos + micro_recall_pos + macro_recall_neg + micro_recall_neg) / 4
            else:
                val_loss = 1 - (macro_recall_pos + micro_recall_pos) / 2
    
            old_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            val_log[epoch] = val_loss
            
            if new_lr < old_lr:
                print(f"\nEpoch {epoch}: Reducing learning rate from {old_lr:.6e} to {new_lr:.6e}\n")
            
            print('--- validation at epoch %d, total processed %d, ACC %.3f/1.000, macR %.3f/%.3f, micR %.3f/%.3f, monitor %d, time elapsed(min) %.2f'
                  % (epoch, val_size, accuracy_pos, macro_recall_pos, best_macro_recall_pos, micro_recall_pos, best_micro_recall_pos, 
                     epoch - np.argmin(val_log[:epoch + 1]), (time.time()-start_time)/60))
            
            # Log metrics to wandb
            if self.config["wandb"]:
                wandb.log({
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "train_loss": train_loss_epoch,
                    "validation_loss": val_loss,
                    "accuracy_pos": accuracy_pos,
                    "macro_recall_pos": macro_recall_pos,
                    "micro_recall_pos": micro_recall_pos,
                })
                
                # Only log negative metrics if data_type is "all"
                if self.config["data_type"] == "all":
                    wandb.log({
                        "accuracy_neg": accuracy_neg,
                        "macro_recall_neg": macro_recall_neg,
                        "micro_recall_neg": micro_recall_neg,
                    })
            
            # Save best model
            if np.argmin(val_log[:epoch + 1]) == epoch:
                torch.save(self.net.state_dict(), self.model_path)
                print('--- model saved at epoch %d' % epoch)
                best_model_epoch = epoch
                
                # Save model to WandB
                if self.config.get("wandb_save_model", True) and self.config["wandb"] and wandb.run:
                    wandb.save(self.model_path)
                    print(f'--- model saved to WandB: {self.model_path}')
        
        print('training terminated at epoch %d' % epoch)
        
        # Load the best model if one was saved
        if best_model_epoch != -1:
            print(f'Best model found at epoch {best_model_epoch}')
            was_loaded = self.load()
            if was_loaded:
                print(f'Best model from epoch {best_model_epoch} loaded successfully')
            else:
                print('Warning: Could not load best model, keeping final training state')
        else:
            print('No model improvement during training, keeping final training state')

    def inference(self, tst_loader, temperature=1.0):
        """
        Perform inference on the test loader.
        Returns predictions for both positive and negative samples if data_type is "all",
        otherwise returns only positive predictions.
        """
        self.net.eval()
        with torch.no_grad():
            y_preds_pos = []
            y_preds_neg = []
               
            for batchidx, batchdata in enumerate(tst_loader):
                if batchdata is None:
                    continue
                    
                inputs = batchdata[0].to(self.cuda)
                
                # Run inference for positive samples (pos_neg_reac = 1)
                pos_neg_reac_pos = torch.ones((inputs.shape[0], 1), dtype=torch.float32, device=self.cuda)
                inputs_pos = torch.cat([inputs, pos_neg_reac_pos], dim=1)
                preds_pos = self.net(inputs_pos)
                
                # Clamp predictions to prevent numerical instability
                preds_pos = torch.clamp(preds_pos, -10, 10)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    preds_pos = preds_pos / temperature
                
                preds_list_pos = preds_pos.cpu().numpy()
                y_preds_pos.append(preds_list_pos)
                
                # Only run inference for negative samples if data_type is "all"
                if self.config["data_type"] == "all":
                    pos_neg_reac_neg = torch.zeros((inputs.shape[0], 1), dtype=torch.float32, device=self.cuda)
                    inputs_neg = torch.cat([inputs, pos_neg_reac_neg], dim=1)
                    preds_neg = self.net(inputs_neg)
                    
                    # Clamp predictions to prevent numerical instability
                    preds_neg = torch.clamp(preds_neg, -10, 10)
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        preds_neg = preds_neg / temperature
                    
                    preds_list_neg = preds_neg.cpu().numpy()
                    y_preds_neg.append(preds_list_neg)
            
            # Stack all predictions
            y_preds_pos = np.vstack(y_preds_pos)
            y_preds_pos = expit(y_preds_pos)
            treshold = self.config["treshold"]
            y_preds_pos = [np.where(x > treshold)[0].tolist() for x in y_preds_pos]
            
            if self.config["data_type"] == "all":
                y_preds_neg = np.vstack(y_preds_neg)
                y_preds_neg = expit(y_preds_neg)
                y_preds_neg = [np.where(x > treshold)[0].tolist() for x in y_preds_neg]
            else:
                y_preds_neg = None
            
            return y_preds_pos, y_preds_neg
