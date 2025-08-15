import numpy as np
import time
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.special import expit

import dgl
from dgl.nn.pytorch import NNConv, Set2Set

import wandb

class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['node_attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 readout_feats):
        
        super(reactionMPNN, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats, readout_feats=readout_feats)

    def forward(self, rmols, pmols):

        r_graph_feats = torch.cat([self.mpnn(mol) for mol in rmols], 1)
        p_graph_feats = self.mpnn(pmols[0])

        return torch.cat([r_graph_feats, p_graph_feats], 1)


class encoder(nn.Module):

    def __init__(self, n_classes,
                 latent_feats, readout_feats, predict_hidden_feats):
        
        super(encoder, self).__init__()
          
        self.latent_feats = latent_feats
          
        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 3 + n_classes + 1, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, latent_feats * 2)
        )   

    def forward(self, vector, graph_embed=None, pos_neg_sample=1):
        
        # Handle pos_neg_sample - ensure it's a 2D tensor with shape [batch_size, 1]
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=vector.device).expand(vector.size(0), 1)
        elif isinstance(pos_neg_sample, torch.Tensor):
            # Ensure it's 2D with shape [batch_size, 1]
            if pos_neg_sample.dim() == 1:
                pos_neg_sample = pos_neg_sample.unsqueeze(1)
            elif pos_neg_sample.dim() > 2:
                pos_neg_sample = pos_neg_sample.view(pos_neg_sample.size(0), -1)
        
        mu, log_var = torch.split(self.predict(torch.cat([vector, graph_embed, pos_neg_sample], 1)), [self.latent_feats, self.latent_feats], dim = 1)

        # Clamp the mu and log_var and prevent KLD explosion
        mu = torch.clamp(mu, -10, 10)
        log_var = torch.clamp(log_var, -10, 10)

        return mu, log_var
        
        
class decoder(nn.Module):  

    def __init__(self, n_classes,
                 latent_feats, readout_feats, predict_hidden_feats):
        
        super(decoder, self).__init__()

        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, n_classes)
        )   

    def forward(self, latent, graph_embed=None, pos_neg_sample=1):
        
        # Handle pos_neg_sample - ensure it's a 2D tensor with shape [batch_size, 1]
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=latent.device).expand(latent.size(0), 1)
        elif isinstance(pos_neg_sample, torch.Tensor):
            # Ensure it's 2D with shape [batch_size, 1]
            if pos_neg_sample.dim() == 1:
                pos_neg_sample = pos_neg_sample.unsqueeze(1)
            elif pos_neg_sample.dim() > 2:
                pos_neg_sample = pos_neg_sample.view(pos_neg_sample.size(0), -1)

        return self.predict(torch.cat([latent, graph_embed, pos_neg_sample], 1))
        

class VAE(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, n_classes,
                 latent_feats = 128, readout_feats = 1024, predict_hidden_feats = 512):
        
        super(VAE, self).__init__()

        self.latent_feats = latent_feats

        self.rmpnn = reactionMPNN(node_in_feats, edge_in_feats, readout_feats)
        self.encoder = encoder(n_classes, latent_feats, readout_feats, predict_hidden_feats)
        self.decoder = decoder(n_classes, latent_feats, readout_feats, predict_hidden_feats)

    def forward(self, rmols=None, pmols=None, labels=None, pos_neg_sample=1):

        graph_embed = self.rmpnn(rmols, pmols)
        
        mu, log_var = self.encoder(labels, graph_embed, pos_neg_sample)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        latent = mu + eps * std
        
        vector_pred = self.decoder(latent, graph_embed, pos_neg_sample)

        # Clamp the vector_pred and prevent BCE explosion
        vector_pred = torch.clamp(vector_pred, -10, 10)

        return vector_pred, mu, log_var
    
    def sampling(self, rmols=None, pmols=None, pos_neg_sample=1, batch_size=None, device=None, temperature=1.0):
        
        graph_embed = self.rmpnn(rmols, pmols)
        latent = torch.randn(len(graph_embed), self.latent_feats, device=graph_embed.device)
        
        vector_pred = self.decoder(latent, graph_embed, pos_neg_sample)
        
        # Apply temperature scaling
        if temperature != 1.0:
            vector_pred = vector_pred / temperature
        
        return vector_pred


class Trainer:

    def __init__(self, net, cuda, config, wandb_project="ReactionVAE_Baseline"):
    
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
            print(f"INFO: No model found at {self.model_path}. Starting from scratch.")
            return False
        except Exception as e:
            print(f"ERROR: Loading model from {self.model_path}: {e}")
            return False
        
    def _calculate_class_weights(self, pos_count_matrix_1, pos_count_matrix_0, neg_count_matrix_1, neg_count_matrix_0):
        """
        Calculate class weights for positive and negative samples separately.
        
        Args:
            pos_count_matrix: numpy array of shape (num_samples, num_classes) for positive samples
            neg_count_matrix: numpy array of shape (num_samples, num_classes) for negative samples
            
        Returns:
            tuple: (pos_weights, neg_weights) - both torch.Tensor of shape (num_classes,)
        """
        def _get_weights(count_matrix_1, count_matrix_0):
            if count_matrix_1 is None or count_matrix_1.shape[0] == 0:
                return None
            
            pos_count = count_matrix_1.sum(axis=0)
            neg_count = count_matrix_0.sum(axis=0)
            
            # Calculate pos_weight for BCEWithLogitsLoss (ratio of negatives to positives)
            # Add small epsilon to avoid division by zero
            ratio = neg_count / (pos_count + 1e-8)
            
            # Clamp to reasonable range to prevent numerical instability
            ratio = np.clip(ratio, 0.1, 10.0)
            
            return torch.tensor(ratio, dtype=torch.float32, device=self.cuda)
        
        pos_weights = _get_weights(pos_count_matrix_1, pos_count_matrix_0)
        neg_weights = _get_weights(neg_count_matrix_1, neg_count_matrix_0) 
        
        return pos_weights, neg_weights
               
    def training(self, train_loader, val_loader, max_epochs = 250, calc_diversity = True):

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
        
        # Ensure lr and weight_decay are floats (WandB sometimes passes them as strings)
        lr = float(self.config["lr"])
        weight_decay = float(self.config["weight_decay"])
        optimizer = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        #patience as 20% of max_epochs
        patience = int(max_epochs * 0.2)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-6)
    

        #log reaction type from model_path
        rtype = self.model_path.split('_')[-2]
        weight_bce = self.config["weight_bce"]
        if self.config["wandb"]:
            wandb.log({"rxy_type": rtype,
                      "weight_bce": weight_bce})

        val_y = val_loader.dataset.y
        best_macro_recall = 1.
        best_micro_recall = 1.
        
        val_log = np.zeros(max_epochs)
        best_model_epoch = 0
        print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6e}") # Optional: print initial LR

        for epoch in range(max_epochs):
            
            # training
            self.net.train()
            start_time = time.time()
            grad_norm_list = []
            train_loss_epoch = 0.
            bce_loss_epoch = 0.
            kld_loss_epoch = 0.

            for batchidx, batchdata in enumerate(train_loader):
    
                inputs_rmol = [b.to(self.cuda) for b in batchdata[:self.rmol_max_cnt]]
                inputs_pmol = [b.to(self.cuda) for b in batchdata[self.rmol_max_cnt:self.rmol_max_cnt+self.pmol_max_cnt]]
                
                labels = batchdata[-2].to(self.cuda)
                pos_neg_sample = batchdata[-1].to(self.cuda)
                
                preds, mu, log_var = self.net(inputs_rmol, inputs_pmol, labels, pos_neg_sample)

                # Calculate BCE loss using appropriate weights based on pos_neg_sample
                if self.config["class_weights"]:
                    # Split batch by positive/negative samples
                    pos_mask = (pos_neg_sample == 1).squeeze(-1)
                    neg_mask = (pos_neg_sample == 0).squeeze(-1)
                    
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
                        loss_bce = total_loss_bce / total_samples
                    else:
                        # Fallback to unweighted loss
                        loss_bce = loss_fns['unweighted'](preds, labels).sum(dim=1).mean()
                else:
                    loss_bce = loss_fns['unweighted'](preds, labels).sum(dim=1).mean()
                
                # Add safety check for negative or invalid BCE loss
                if loss_bce < 0 or torch.isnan(loss_bce) or torch.isinf(loss_bce):
                    print(f"WARNING: Invalid BCE loss detected: {loss_bce.item()}. Using unweighted fallback.")
                    loss_bce = loss_fns['unweighted'](preds, labels).sum(dim=1).mean()
                
                loss_kld = - 0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(axis=1).mean()
                #KL annealing to avoid  KL collapse
                kl_weight = min(1.0, epoch / self.config["kl_annealing_epochs"]) if self.config.get("kl_annealing_epochs", 0) else 1.0
                loss_kld = kl_weight * loss_kld
                loss = (weight_bce / (weight_bce + 1)) * loss_bce + (1 / (weight_bce + 1)) * loss_kld

                # --- REMOVED DEBUG PRINT STATEMENTS ---
    
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
                bce_loss_epoch += loss_bce.detach().item()
                kld_loss_epoch += loss_kld.detach().item()

            
            train_loss_epoch /= len(train_loader)
            bce_loss_epoch /= len(train_loader)
            kld_loss_epoch /= len(train_loader)

            print('--- training epoch %d, lr %f, loss %.3f, bce_loss %.3f, kld_loss %.3f, time elapsed(min) %.2f, max gradient norm %.3f'
                 % (epoch, optimizer.param_groups[0]['lr'], train_loss_epoch, bce_loss_epoch, kld_loss_epoch, (time.time()-start_time)/60, np.max(grad_norm_list))) 
            
            # validation
            start_time = time.time()
            # Use configurable n_sampling for validation consistency
            n_sampling_val = self.config.get("n_sampling_val", 50)  # Default to 50 if not specified
            #tst_y_preds_pos, tst_y_preds_neg: list of lists, external list has len(val_loader) elements, each element is of len n_sampling, 
            #each element of this list is in turn is a predicted reaction condition set.
            tst_y_preds_pos, tst_y_preds_neg = self.inference(val_loader, n_sampling = n_sampling_val, temperature=self.config["temperature"])

            #Calculate diversity for the set of positive samples to see how many unique reaction conditions are predicted for each sample
            # Initialize diversity variables with default values
            avg_diversity_pos = 0.0
            avg_diversity_neg = 0.0
            
            if calc_diversity:
                avg_diversity_pos = np.mean(np.array([len(set(tuple(pred) for pred in i)) for i in tst_y_preds_pos]))
                if self.config["verbose"]:
                    print(f"avg diversity pos at epoch {epoch}  ", avg_diversity_pos)
                if self.config["wandb"]:
                    wandb.log({"avg_diversity_pos": avg_diversity_pos})
                if self.config["data_type"] == "all":
                    avg_diversity_neg = np.mean(np.array([len(set(tuple(pred) for pred in i)) for i in tst_y_preds_neg]))
                    if self.config["verbose"]:
                        print(f"avg diversity neg at epoch {epoch}  ", avg_diversity_neg)
                    if self.config["wandb"]:
                        wandb.log({"avg_diversity_neg": avg_diversity_neg})

            #If using both positive and negative samples, calculate diversity between the set of negative samples and the set of positive samples
            if self.config["data_type"] == "all":
                jaccard_distances = []
                # Iterate over each sample in the validation set
                for pos_preds, neg_preds in zip(tst_y_preds_pos, tst_y_preds_neg):
                    # Convert lists of predictions to sets of hashable tuples
                    unique_pos = set(tuple(p) for p in pos_preds)
                    unique_neg = set(tuple(p) for p in neg_preds)
                    
                    # Calculate intersection and union
                    intersection = len(unique_pos.intersection(unique_neg))
                    union = len(unique_pos.union(unique_neg))
                    
                    # Jaccard distance is 1 - Jaccard similarity
                    jaccard_dist = 1.0 - (intersection / union) if union > 0 else 0.0
                    jaccard_distances.append(jaccard_dist)
                
                # Average the diversity score across all samples
                avg_inter_diversity = np.mean(jaccard_distances)
                if self.config["verbose"]:
                    print(f"avg inter diversity: {avg_inter_diversity}")
                if self.config["wandb"]:
                    wandb.log({"avg_inter_diversity": avg_inter_diversity})
            else:
                avg_inter_diversity = 0.0 # Not applicable if not using negative samples
            
            # Create two val_y for positive and negative conditions
            val_y_pos = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in val_y]
            val_y_neg = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 0] for sublist in val_y]

            # Pair up the true and predicted lists
            pos_pairs = zip(val_y_pos, tst_y_preds_pos)
            neg_pairs = zip(val_y_neg, tst_y_preds_neg)

            # Create a new list of pairs, keeping only those where the true list (y_true) is not empty
            filtered_pos_pairs = [(y_true, y_pred) for y_true, y_pred in pos_pairs if y_true]
            filtered_neg_pairs = [(y_true, y_pred) for y_true, y_pred in neg_pairs if y_true]

            # Handle the edge case where ALL lists might be empty
            if filtered_pos_pairs:
                filtered_val_y_pos, filtered_tst_y_preds_pos = map(list, zip(*filtered_pos_pairs))
            else:
                filtered_val_y_pos, filtered_tst_y_preds_pos = [], []

            if filtered_neg_pairs:
                filtered_val_y_neg, filtered_tst_y_preds_neg = map(list, zip(*filtered_neg_pairs))
            else:
                filtered_val_y_neg, filtered_tst_y_preds_neg = [], []

            if self.config["verbose"]:
                print(f"y pred pos validation at epoch {epoch}  ", [i[:5] for i in filtered_tst_y_preds_pos[:8]])
                print(f"y true pos validation at epoch {epoch}  ", [i[:3] for i in filtered_val_y_pos[:10]])
                if self.config["data_type"] == "all":
                    print(f"y pred neg validation at epoch {epoch}  ", [i[:5] for i in filtered_tst_y_preds_neg[:8]])
                    print(f"y true neg validation at epoch {epoch}  ", [i[:3] for i in filtered_val_y_neg[:10]])


            accuracy_pos = np.mean([np.max([(c in filtered_tst_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) if filtered_val_y_pos else 0.0
            macro_recall_pos = np.mean([np.mean([(c in filtered_tst_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) if filtered_val_y_pos else 0.0
            micro_recall_pos = np.sum([np.sum([(c in filtered_tst_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) / np.sum([len(a) for a in filtered_val_y_pos]) if filtered_val_y_pos else 0.0

            if self.config["data_type"] == "all":
                accuracy_neg = np.mean([np.max([(c in filtered_tst_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) if filtered_val_y_neg else 0.0
                macro_recall_neg = np.mean([np.mean([(c in filtered_tst_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) if filtered_val_y_neg else 0.0
                micro_recall_neg = np.sum([np.sum([(c in filtered_tst_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) / np.sum([len(a) for a in filtered_val_y_neg]) if filtered_val_y_neg else 0.0
            else:
                accuracy_neg = 0.0
                macro_recall_neg = 0.0
                micro_recall_neg = 0.0

            if self.config["data_type"] == "all":
                val_loss = 1 - (macro_recall_pos + micro_recall_pos + macro_recall_neg + micro_recall_neg + accuracy_pos + accuracy_neg) / 6
            else:
                val_loss = 1 - (macro_recall_pos + micro_recall_pos + accuracy_pos) / 3
    
            old_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step(val_loss) # Step the scheduler based on validation loss
            new_lr = optimizer.param_groups[0]['lr']

            val_log[epoch] = val_loss

            # Print message if LR changed 
            if new_lr < old_lr:
                print(f"\nEpoch {epoch}: Reducing learning rate from {old_lr:.6e} to {new_lr:.6e}\n")

            if self.config["data_type"] == "all":   
                print(f'--- validation at epoch {epoch}, '
                    f'ACC (+) {accuracy_pos:.3f}/1.000, '
                    f'macR (+) {macro_recall_pos:.3f}/1.000, '
                    f'micR (+) {micro_recall_pos:.3f}/1.000, '
                    f'ACC (-) {accuracy_neg:.3f}/1.000, '
                    f'macR (-) {macro_recall_neg:.3f}/1.000, '
                    f'micR (-) {micro_recall_neg:.3f}/1.000, '
                    f'time elapsed(min) {(time.time() - start_time) / 60:.2f}')
            else:
                print(f'--- validation at epoch {epoch}, '
                    f'ACC (+) {accuracy_pos:.3f}/1.000, '
                    f'macR (+) {macro_recall_pos:.3f}/1.000, '
                    f'micR (+) {micro_recall_pos:.3f}/1.000, '
                    f'time elapsed(min) {(time.time() - start_time) / 60:.2f}')

            # Log metrics to wandb
            if self.config["wandb"]:
                if self.config["data_type"] == "all":
                    wandb.log({
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "train_loss": train_loss_epoch,
                        "bce_loss": bce_loss_epoch,
                        "kld_loss": kld_loss_epoch,
                        "validation_loss": val_loss,
                        "avg_diversity_pos": avg_diversity_pos,
                        "avg_diversity_neg": avg_diversity_neg,
                        "avg_inter_diversity": avg_inter_diversity,
                        "accuracy_pos": accuracy_pos,
                        "macro_recall_pos": macro_recall_pos,
                        "micro_recall_pos": micro_recall_pos,
                        "accuracy_neg": accuracy_neg,
                        "macro_recall_neg": macro_recall_neg,
                        "micro_recall_neg": micro_recall_neg,
                    })
                else:
                    wandb.log({
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "train_loss": train_loss_epoch,
                        "bce_loss": bce_loss_epoch,
                        "kld_loss": kld_loss_epoch,
                        "validation_loss": val_loss,
                        "avg_diversity_pos": avg_diversity_pos,
                        "accuracy_pos": accuracy_pos,
                        "macro_recall_pos": macro_recall_pos,
                        "micro_recall_pos": micro_recall_pos,
                    })

            # update model saved
            if np.argmin(val_log[:epoch + 1]) == epoch:
                best_model_epoch = epoch
                torch.save(self.net.state_dict(), self.model_path) 
                print('--- model saved at epoch %d' % best_model_epoch)
                
                # Save model to WandB
                if self.config.get("wandb_save_model", True) and self.config["wandb"] and wandb.run:
                    wandb.save(self.model_path)
                    print(f'--- model saved to WandB: {self.model_path}')
            
            # elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            #     break
    
        print('training terminated at epoch %d' %epoch)
        print('best model saved at epoch %d' % best_model_epoch)
        self.load()
        
    
    def inference(self, tst_loader, n_sampling = 100, temperature=1.0):
        """
        In this modified version of the function, we not only return a prediction for the positive but also for the negative samples
        """
        def forward(tst_loader, pos_neg_flag=1):
            """
            Test loader is where we get the data from and pos_neg_flag is telling us if we should 
            sample positive or negative sample from the conditional VAE
            """
            tst_y_scores = []
            with torch.no_grad():
                for batchidx, batchdata in enumerate(tst_loader):
                    if not batchdata: # Handle empty batches
                        print("WARNING: Empty batch detected. Skipping this batch.")
                        continue
        
                    inputs_rmol = [b.to(self.cuda) for b in batchdata[:self.rmol_max_cnt]]
                    inputs_pmol = [b.to(self.cuda) for b in batchdata[self.rmol_max_cnt:self.rmol_max_cnt+self.pmol_max_cnt]]
        
                    preds_list = self.net.sampling(inputs_rmol, inputs_pmol, pos_neg_flag, temperature=temperature).cpu().numpy() #produces logits for n_classes
                    tst_y_scores.append(preds_list)

            if not tst_y_scores: # Handle case where loader was empty
                print("WARNING: No predictions were made. Returning empty lists.")
                return []

            treshold = self.config["treshold"]
            tst_y_scores = expit(np.vstack(tst_y_scores))
            tst_y_preds = [[np.where(x > treshold)[0].tolist()] for x in tst_y_scores] #O.G. was 0.5
        
            return tst_y_preds

        self.net.eval()   
        tst_y_preds_pos_list = [forward(tst_loader, pos_neg_flag=1) for _ in range(n_sampling)] #external list: n_sampling elements, each element is a list of lists is a predicted reaction condition set for each sample
        tst_y_preds_neg_list = [forward(tst_loader, pos_neg_flag=0) for _ in range(n_sampling)]

        if not tst_y_preds_pos_list or not tst_y_preds_pos_list[0]:
            print("WARNING: No positive predictions were made. Returning empty list.")
            tst_y_preds_pos = []
        else:
            tst_y_preds_pos = [sum([tst_y_preds_pos_list[j][i] for j in range(n_sampling)], []) for i in range(len(tst_y_preds_pos_list[0]))]
        
        if not tst_y_preds_neg_list or not tst_y_preds_neg_list[0]:
            print("WARNING: No negative predictions were made. Returning empty list.")
            tst_y_preds_neg = []
        else:
            tst_y_preds_neg = [sum([tst_y_preds_neg_list[j][i] for j in range(n_sampling)], []) for i in range(len(tst_y_preds_neg_list[0]))]

        return tst_y_preds_pos, tst_y_preds_neg
