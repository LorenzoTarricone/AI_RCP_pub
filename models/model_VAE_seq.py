import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss
import random

import dgl
from dgl.nn.pytorch import NNConv, Set2Set

import warnings
import wandb
import os

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
        
        # Store readout_feats as instance variable
        self.readout_feats = readout_feats
             
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
        # Store readout_feats as instance variable
        self.readout_feats = readout_feats

    def forward(self, rmols, pmols):

        r_graph_feats = torch.cat([self.mpnn(mol) for mol in rmols], 1)
        p_graph_feats = self.mpnn(pmols[0])

        return torch.cat([r_graph_feats, p_graph_feats], 1)


class encoder(nn.Module):

    def __init__(self, n_classes,
                 latent_feats, readout_feats, predict_hidden_feats):
        
        super(encoder, self).__init__()
          
        self.latent_feats = latent_feats
          
        #Readout features are tripled because we concatenate 2 reactants + 1 product
        #n_classes is the number of classes
        #1 is the binary indicator for positive/negative sample
        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 3 + n_classes  + 1, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, latent_feats * 2)
        )   

    def forward(self, vector, graph_embed=None, pos_neg_sample=1, embed=None):
        """
        Forward pass for the encoder.

        Args:
            vector (torch.Tensor): Input condition vector (labels).
            graph_embed (torch.Tensor, optional): Concatenated GNN embeddings. Defaults to None.
            pos_neg_sample (int): Binary indicator for positive/negative sample. Defaults to 1.


        Returns:
            tuple: (mu, log_var) tensors for the latent distribution.
        """
        # Build input by concatenating available features
        inputs_to_cat = [vector]

        if graph_embed is not None:
            inputs_to_cat.append(graph_embed)

        # Add pos_neg_sample as a tensor with shape [batch_size, 1]
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=vector.device).expand(vector.size(0), 1)
        inputs_to_cat.append(pos_neg_sample)

        if len(inputs_to_cat) == 1:
            warnings.warn("Encoder received only label vector. Ensure Pos neg flag provided.")

        combined_input = torch.cat(inputs_to_cat, 1)

        # Predict mu and log_var
        mu_log_var = self.predict(combined_input)
        mu, log_var = torch.split(mu_log_var, [self.latent_feats, self.latent_feats], dim=1)

        # Clamp the mu and log_var and prevent KLD explosion
        mu = torch.clamp(mu, -10, 10)
        log_var = torch.clamp(log_var, -10, 10)

        return mu, log_var
        

        
class decoder_cat(nn.Module):  

    def __init__(self, rtype, n_info, latent_feats, readout_feats, predict_hidden_feats):
        super(decoder_cat, self).__init__()

        self.rtype = rtype
        (n_cats, n_sol_1, n_sol_2, n_add, n_base) = n_info

        if rtype == 'bh':
            self.predict = nn.Sequential(
                nn.Linear(readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_cats)
            )
        if rtype == 'sm':
            self.predict = nn.Sequential(
                nn.Linear(n_sol_1 + n_sol_2 + 1 + n_add + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_cats)
            )
    def forward(self, input, pos_neg_sample=1):
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=input[0].device).expand(input[0].size(0), 1)

        if self.rtype == "bh":
            latent, graph_embed = input
            logits = self.predict(torch.cat([latent, graph_embed, pos_neg_sample], 1))
            return logits
        if self.rtype == 'sm':
            latent, graph_embed, solv_1_pred, solv_2_pred, water_pred, add_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, solv_1_pred, solv_2_pred, water_pred, add_pred, pos_neg_sample], 1))
            return logits
        
    
class decoder_base(nn.Module):
    def __init__(self, rtype, n_info, latent_feats, readout_feats, predict_hidden_feats):
        super(decoder_base, self).__init__()

        self.rtype = rtype
        (n_cats, n_sol_1, n_sol_2, n_add, n_base) = n_info

        if rtype == 'bh':
            self.predict = nn.Sequential(
                nn.Linear(n_cats + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_base)
            )
        if rtype == 'sm':
            self.predict = nn.Sequential(
                nn.Linear(n_sol_1 + n_sol_2 + 1 + n_add + n_cats + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_base)
            )
        
    def forward(self, input, pos_neg_sample=1):
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=input[0].device).expand(input[0].size(0), 1)

        if self.rtype == "bh":
            latent, graph_embed, cat_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, cat_pred, pos_neg_sample], 1))
            return logits
        if self.rtype == 'sm':
            latent, graph_embed, solv_1_pred, solv_2_pred, water_pred, add_pred, cat_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, solv_1_pred, solv_2_pred, water_pred, add_pred, cat_pred, pos_neg_sample], 1))
            return logits


class decoder_solv_1(nn.Module):  

    def __init__(self, rtype, n_info, latent_feats, readout_feats, predict_hidden_feats):
        
        super(decoder_solv_1, self).__init__()

        self.rtype = rtype
        (n_cats, n_sol_1, n_sol_2, n_add, n_base) = n_info

        if rtype == 'bh':
            self.predict = nn.Sequential(
                nn.Linear(n_base + n_cats + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_sol_1)
            )
        if rtype == 'sm':
            self.predict = nn.Sequential(
                nn.Linear(readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_sol_1)
            )
    def forward(self, input, pos_neg_sample=1):
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=input[0].device).expand(input[0].size(0), 1)

        if self.rtype == "bh":
            latent, graph_embed, cat_pred, base_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, cat_pred, base_pred, pos_neg_sample], 1))
            return logits
        if self.rtype == 'sm':
            latent, graph_embed = input
            logits = self.predict(torch.cat([latent, graph_embed, pos_neg_sample], 1))
            return logits


class decoder_add(nn.Module):  

    def __init__(self, rtype, n_info, latent_feats, readout_feats, predict_hidden_feats):
        
        super(decoder_add, self).__init__()

        self.rtype = rtype
        (n_cats, n_sol_1, n_sol_2, n_add, n_base) = n_info

        if rtype == 'bh':
            self.predict = nn.Sequential(
                nn.Linear(n_base + n_cats + n_sol_1 + 1 + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_add)
            )
        if rtype == 'sm':
            self.predict = nn.Sequential(
                nn.Linear(n_sol_1 + n_sol_2 + 1 + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, n_add)
            )
    def forward(self, input, pos_neg_sample=1):
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=input[0].device).expand(input[0].size(0), 1)

        if self.rtype == "bh":
            latent, graph_embed, cat_pred, base_pred, solv_1_pred, water_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, cat_pred, base_pred, solv_1_pred, water_pred, pos_neg_sample], 1))
            return logits
        if self.rtype == 'sm':
            latent, graph_embed, sol_1_pred, sol_2_pred, water_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, sol_1_pred, sol_2_pred, water_pred, pos_neg_sample], 1))
            return logits
    
        
class decoder_water(nn.Module):  

    def __init__(self,rtype, n_info, latent_feats, readout_feats, predict_hidden_feats):
        
        super(decoder_water, self).__init__()

        self.rtype = rtype
        (n_cats, n_sol_1, n_sol_2, n_add, n_base) = n_info

        if rtype == 'bh':
            self.predict = nn.Sequential(
                nn.Linear(n_base + n_cats + n_sol_1 + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, 1)
            )
        if rtype == 'sm':
            self.predict = nn.Sequential(
                nn.Linear(n_sol_1 + n_sol_2 + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
                nn.Linear(predict_hidden_feats, 1)
            )

    def forward(self, input, pos_neg_sample=1):
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=input[0].device).expand(input[0].size(0), 1)

        if self.rtype == "bh":
            latent, graph_embed, cat_pred, base_pred, solv_1_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, cat_pred, base_pred, solv_1_pred, pos_neg_sample], 1))
            return logits  # Use sigmoid for binary classification
        if self.rtype == 'sm':
            latent, graph_embed, sol_1_pred, sol_2_pred = input
            logits = self.predict(torch.cat([latent, graph_embed, sol_1_pred, sol_2_pred, pos_neg_sample], 1))
            return logits  # Use sigmoid for binary classification
        

class decoder_solv_2(nn.Module):  

    def __init__(self, rtype, n_info, latent_feats, readout_feats, predict_hidden_feats):
        
        super(decoder_solv_2, self).__init__()

        self.rtype = rtype
        (n_cats, n_sol_1, n_sol_2, n_add, n_base) = n_info

        self.predict = nn.Sequential(
            nn.Linear(n_sol_1 + readout_feats * 3 + latent_feats + 1, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(),
            nn.Linear(predict_hidden_feats, n_sol_2)
        )

    def forward(self, input, pos_neg_sample=1):
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=input[0].device).expand(input[0].size(0), 1)

        latent, graph_embed, solv_1_pred = input
        logits = self.predict(torch.cat([latent, graph_embed, solv_1_pred, pos_neg_sample], 1))
        return logits
    
    
class VAE_seq(nn.Module):

    def __init__(self, rtype, node_in_feats, edge_in_feats,
                n_classes, n_info,
                latent_feats = 128, readout_feats = 1024, predict_hidden_feats = 512):
        
        super(VAE_seq, self).__init__()

        self.latent_feats = latent_feats
        self.rtype = rtype
        self.n_info = n_info

        self.rmpnn = reactionMPNN(node_in_feats, edge_in_feats, readout_feats)
        self.encoder = encoder(n_classes, latent_feats, readout_feats, predict_hidden_feats)

        #Build decoder depending on the reaction type
        
        self.decoder_cat = decoder_cat( rtype, n_info, latent_feats, readout_feats, predict_hidden_feats)
        self.decoder_base = decoder_base(rtype, n_info, latent_feats, readout_feats, predict_hidden_feats)
        self.decoder_solv_1 = decoder_solv_1(rtype, n_info, latent_feats, readout_feats, predict_hidden_feats)
        self.decoder_add = decoder_add(rtype, n_info, latent_feats, readout_feats, predict_hidden_feats)
        self.decoder_water = decoder_water( rtype, n_info, latent_feats, readout_feats, predict_hidden_feats)
        if rtype == 'sm':
            self.decoder_solv_2 = decoder_solv_2(rtype, n_info, latent_feats, readout_feats, predict_hidden_feats)

    def forward(self, rmols, pmols, vector, pos_neg_sample=1):
        """
        Forward pass through the VAE.

        Args:
            rmols (list): List of reactant DGL graphs.
            pmols (list): List of product DGL graphs.
            vector (torch.Tensor): Input condition vector (labels).
            pos_neg_sample (int or torch.Tensor): Binary indicator for positive/negative sample. Defaults to 1.

        Returns:
            tuple: (latent, graph_embed, mu, log_var) containing latent vector, graph embeddings, and distribution parameters.
        """
        graph_embed = self.rmpnn(rmols, pmols)
        
        #Encoding
        mu, log_var = self.encoder(vector, graph_embed, pos_neg_sample)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        latent = mu + eps * std

        # #Decoding
        # if self.rtype == 'bh':
        #     ## BH: Cat -> Base -> Solv1 -> Wat -> Add
        #     vector_pred_cat = self.decoder_cat([latent, graph_embed])
        #     vector_pred_base = self.decoder_base([latent, graph_embed, vector_pred_cat])
        #     vector_pred_solv_1 = self.decoder_solv_1([latent, graph_embed, vector_pred_cat, vector_pred_base])
        #     vector_pred_water = self.decoder_water(latent, graph_embed, vector_pred_cat, vector_pred_base, vector_pred_solv_1)
        #     vector_pred_add = self.decoder_solv_1([latent, graph_embed, vector_pred_cat, vector_pred_base, vector_pred_solv_1, vector_pred_water])
        #     return (vector_pred_cat, vector_pred_base, vector_pred_solv_1, vector_pred_water, vector_pred_add), mu, log_var
        # if self.rtype == 'sm':
        #     ## SM: Solv1 -> Solv2 -> Wat -> Add -> Cat -> Base
        #     vector_pred_solv_1 = self.decoder_solv_1([latent, graph_embed])
        #     vector_pred_solv_2 = self.decoder_solv_2([latent, graph_embed, vector_pred_solv_1])
        #     vector_pred_water = self.decoder_water([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2])
        #     vector_pred_add = self.decoder_add([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2, vector_pred_water])
        #     vector_pred_cat = self.decoder_cat([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2, vector_pred_water, vector_pred_add])
        #     vector_pred_base = self.decoder_base([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2, vector_pred_water, vector_pred_add, vector_pred_cat])
        #     return (vector_pred_solv_1, vector_pred_solv_2, vector_pred_water, vector_pred_add, vector_pred_cat, vector_pred_base), mu, log_var
    
        return latent, graph_embed, mu, log_var
    
    def sampling(self, rmols, pmols, pos_neg_sample=1, config=None, temperature=1.0):
        """
        Samples from the VAE decoder using random latent vectors.

        Args:
            rmols (list): List of reactant DGL graphs.
            pmols (list): List of product DGL graphs.
            pos_neg_sample (int or torch.Tensor): Binary indicator for positive/negative sample. Defaults to 1.
            embed (torch.Tensor, optional): Additional embeddings. Not used in this model but kept for consistency.
            batch_size (int, optional): The batch size. Required if no embeddings are provided to infer it.
            device (torch.device, optional): Device to place the latent vector on. Required if no embeddings.
            temperature (float): Temperature parameter for scaling logits. Default is 1.0.

        Returns:
            torch.Tensor: Predicted logits from the decoder.
        """
        graph_embed = self.rmpnn(rmols, pmols)
        current_batch_size = graph_embed.shape[0]
        current_device = graph_embed.device

        # Convert pos_neg_sample to tensor if it's an int
        if isinstance(pos_neg_sample, int):
            pos_neg_sample = torch.tensor([[pos_neg_sample]], device=current_device).expand(current_batch_size, 1)

        latent = torch.randn(current_batch_size, self.latent_feats, device=current_device)
        (n_cat, n_sol_1, n_sol_2, n_add, n_base) = self.n_info
        
        if self.rtype == 'bh':
            ## BH: Cat -> Base -> Solv1 -> Wat -> Add
            vector_pred_cat = self.decoder_cat([latent, graph_embed], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_cat = vector_pred_cat / temperature
            #Use probability to sample the class
            prob_pred_cat = torch.softmax(vector_pred_cat, dim=1)
            vector_pred_idx = torch.multinomial(prob_pred_cat, 1).squeeze(-1)
            
            vector_pred_cat =  F.one_hot(vector_pred_idx, num_classes=n_cat)

            vector_pred_base = self.decoder_base([latent, graph_embed, vector_pred_cat], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_base = vector_pred_base / temperature
            prob_pred_base = torch.softmax(vector_pred_base, dim=1)
            vector_pred_idx = torch.multinomial(prob_pred_base, 1).squeeze(-1)
            vector_pred_base =  F.one_hot(vector_pred_idx, num_classes=n_base)

            vector_pred_solv_1 = self.decoder_solv_1([latent, graph_embed, vector_pred_cat, vector_pred_base], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_solv_1 = vector_pred_solv_1 / temperature
            prob_pred_solv_1 = torch.softmax(vector_pred_solv_1, dim=1)
            vector_pred_idx = torch.multinomial(prob_pred_solv_1, 1).squeeze(-1)
            vector_pred_solv_1 =  F.one_hot(vector_pred_idx, num_classes=n_sol_1)

            vector_pred_water = self.decoder_water([latent, graph_embed, vector_pred_cat, vector_pred_base, vector_pred_solv_1], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_water = vector_pred_water / temperature
            prob_pred_water =  torch.sigmoid(vector_pred_water)
            vector_pred_water =  (torch.bernoulli(prob_pred_water)).float()

            vector_pred_add = self.decoder_add([latent, graph_embed, vector_pred_cat, vector_pred_base, vector_pred_solv_1, vector_pred_water], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_add = vector_pred_add / temperature
            prob_pred_add =  torch.sigmoid(vector_pred_add)
            vector_pred_add =  (torch.bernoulli(prob_pred_add)).float()

            vector_pred = torch.cat([vector_pred_cat, vector_pred_base, vector_pred_solv_1,  vector_pred_water, vector_pred_add], 1)

        if self.rtype == 'sm':
            ## SM: Solv1 -> Solv2 -> Wat -> Add -> Cat -> Base
            vector_pred_solv_1 = self.decoder_solv_1([latent, graph_embed], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_solv_1 = vector_pred_solv_1 / temperature
            prob_pred_solv_1 = torch.softmax(vector_pred_solv_1, dim=1)
            vector_pred_idx = torch.multinomial(prob_pred_solv_1, 1).squeeze(-1)
            vector_pred_solv_1 =  F.one_hot(vector_pred_idx, num_classes=n_sol_1)

            vector_pred_solv_2 = self.decoder_solv_2([latent, graph_embed, vector_pred_solv_1], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_solv_2 = vector_pred_solv_2 / temperature
            prob_pred_solv_2 =  torch.sigmoid(vector_pred_solv_2)
            vector_pred_solv_2 = (torch.bernoulli(prob_pred_solv_2)).float()

            vector_pred_water = self.decoder_water([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_water = vector_pred_water / temperature
            prob_pred_water =  torch.sigmoid(vector_pred_water)
            vector_pred_water =  (torch.bernoulli(prob_pred_water)).float()

            vector_pred_add = self.decoder_add([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2, vector_pred_water], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_add = vector_pred_add / temperature
            prob_pred_add =  torch.sigmoid(vector_pred_add)
            vector_pred_add =  (torch.bernoulli(prob_pred_add)).float()

            vector_pred_cat = self.decoder_cat([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2, vector_pred_water, vector_pred_add], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_cat = vector_pred_cat / temperature
            prob_pred_cat = torch.softmax(vector_pred_cat, dim=1)
            vector_pred_idx = torch.multinomial(prob_pred_cat, 1).squeeze(-1)
            vector_pred_cat =  F.one_hot(vector_pred_idx, num_classes=n_cat)

            vector_pred_base = self.decoder_base([latent, graph_embed, vector_pred_solv_1, vector_pred_solv_2, vector_pred_water, vector_pred_add, vector_pred_cat], pos_neg_sample)
            if temperature != 1.0:
                vector_pred_base = vector_pred_base / temperature
            prob_pred_base = torch.softmax(vector_pred_base, dim=1)
            vector_pred_idx = torch.multinomial(prob_pred_base, 1).squeeze(-1)
            vector_pred_base =  F.one_hot(vector_pred_idx, num_classes=n_base)

            vector_pred = torch.cat([vector_pred_solv_1, vector_pred_solv_2, vector_pred_water, vector_pred_add, vector_pred_cat,  vector_pred_base], 1)

        return vector_pred


class Trainer:

    def __init__(self, net, cuda, config, wandb_project="ReactionVAE_Seq"):
    
        self.net = net.to(cuda)
        self.n_classes = config["n_classes"]
        self.rmol_max_cnt = config["rmol_max_cnt"]
        self.pmol_max_cnt = config["pmol_max_cnt"]
        self.batch_size = config["batch_size"]
        self.model_path = config["model_path"]
        self.cuda = cuda
        self.wandb_project = wandb_project
        self.config = config

        # --- Scheduled Sampling Parameters ---
        self.ss_mode = config.get("ss_mode", "on") # 'on' or 'off'
        self.ss_epsilon_start = config.get("ss_epsilon_start", 1.0)
        self.ss_epsilon_end = config.get("ss_epsilon_end", 0.1)
        self.ss_decay_epochs = config.get("ss_decay_epochs", 100)
        # --- End New Parameters ---

        # Initialize wandb if config indicates it should be used
        if config["wandb"]:
            if wandb.run is None:
                wandb.init(project=self.wandb_project, config={
                    "batch_size": self.batch_size,
                    "model_path": self.model_path,
                    "n_classes": self.n_classes,
                    "latent_feats": net.latent_feats,
                    "readout_feats": net.rmpnn.mpnn.readout_feats,
                    "predict_hidden_feats": net.encoder.predict[0].out_features,
                    "rtype": net.rtype,
                    "ss_mode": self.ss_mode,
                    "ss_epsilon_start": self.ss_epsilon_start,
                    "ss_epsilon_end": self.ss_epsilon_end,
                    "ss_decay_epochs": self.ss_decay_epochs,
                })

    def load(self):
        """Loads model weights from the specified path."""
        print(f"Loading model weights from {self.model_path}")
        try:
            # Use map_location to load onto the correct device specified by self.cuda
            self.net.load_state_dict(torch.load(self.model_path, map_location=self.cuda))
            print("Model weights loaded successfully.")
            if self.config["mode"] == "inference":
                return True
            return True
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}. Starting from scratch.")
            return False
        except Exception as e:
            print(f"Error loading model weights: {e}. Starting from scratch.")
            return False
        
    def _calculate_epsilon(self, epoch):
        """Calculates the epsilon for scheduled sampling based on the current epoch."""

        if self.ss_mode != 'on' or epoch >= self.ss_decay_epochs:
            return self.ss_epsilon_end
        
        # Linear decay
        return self.ss_epsilon_start - (self.ss_epsilon_start - self.ss_epsilon_end) * (epoch / self.ss_decay_epochs)
    

    def _calculate_class_weights(self, pos_count_matrix_1, pos_count_matrix_0, neg_count_matrix_1, neg_count_matrix_0):
        """
        Calculates the class weights for the different components of the model.
        """
        #Get the class weights for the different components of the model.
        (n_cats, n_sol_1, n_sol_2, n_add, n_base) = self.net.n_info
        n_water = 1
        rtype = self.net.rtype.lower()

        component_slices = {}
        off = 0
        if rtype == 'bh':
            component_slices.update({
                'cat': (off, off+n_cats), 'base': (off+n_cats, off+n_cats+n_base),
                'solv_1': (off+n_cats+n_base, off+n_cats+n_base+n_sol_1),
                'solv_2': (off+n_cats+n_base+n_sol_1, off+n_cats+n_base+n_sol_1+n_sol_2),
                'water': (off+n_cats+n_base+n_sol_1+n_sol_2, off+n_cats+n_base+n_sol_1+n_sol_2+n_water),
                'add': (off+n_cats+n_base+n_sol_1+n_sol_2+n_water, off+n_cats+n_base+n_sol_1+n_sol_2+n_water+n_add)
            })
        elif rtype in ['sm', 'suzuki']:
            component_slices.update({
                'solv_1': (off, off+n_sol_1), 'solv_2': (off+n_sol_1, off+n_sol_1+n_sol_2),
                'water': (off+n_sol_1+n_sol_2, off+n_sol_1+n_sol_2+n_water),
                'add': (off+n_sol_1+n_sol_2+n_water, off+n_sol_1+n_sol_2+n_water+n_add),
                'cat': (off+n_sol_1+n_sol_2+n_water+n_add, off+n_sol_1+n_sol_2+n_water+n_add+n_cats),
                'base': (off+n_sol_1+n_sol_2+n_water+n_add+n_cats, off+n_sol_1+n_sol_2+n_water+n_add+n_cats+n_base)
            })
        else:
            raise ValueError(f"Unsupported rtype '{rtype}' for weight calculation.")

        multi_class_components = ['cat', 'solv_1', 'base']
        multi_label_components = ['add', 'solv_2', 'water']

        def _get_weights(count_matrix_1, count_matrix_0):
            weights = {}
            if count_matrix_1 is None or count_matrix_1.shape[0] == 0:
                return weights

            # For multi-class, we use inverse frequency from the `_1` matrix (one-hot labels)
            matrix_1 = torch.tensor(count_matrix_1, dtype=torch.float32).to(self.cuda)
            num_samples = matrix_1.shape[0]
            for name in multi_class_components:
                if name not in component_slices: continue
                start, end = component_slices[name]
                if start == end: continue
                class_counts = matrix_1[:, start:end].sum(axis=0)
                # Inverse frequency weighting
                class_w = num_samples / ((end - start) * class_counts + 1e-8)
                weights[f'{name}_weights'] = class_w.clamp_(max=100)

            # For multi-label, we use neg/pos ratio from both `_1` and `_0` matrices
            if count_matrix_0 is None:
                # If we don't have the matrix of negative counts, we can't compute BCE weights this way.
                # However, we can fall back to the old logic for this case.
                for name in multi_label_components:
                    if name not in component_slices: continue
                    start, end = component_slices[name]
                    if start == end: continue
                    pos_count = matrix_1[:, start:end].sum(axis=0)
                    pos_w = (num_samples - pos_count) / (pos_count + 1e-8)
                    weights[f'{name}_pos_weights'] = pos_w.clamp_(0.1, 10.0)
                return weights

            for name in multi_label_components:
                if name not in component_slices: continue
                start, end = component_slices[name]
                if start == end: continue
                
                # Sum counts for the slice from the pre-computed matrices
                pos_count = np.sum(count_matrix_1[:, start:end], axis=0)
                neg_count = np.sum(count_matrix_0[:, start:end], axis=0)
                
                ratio = neg_count / (pos_count + 1e-8)
                clipped_ratio = np.clip(ratio, 0.1, 10.0) # Clipping for stability
                
                weights[f'{name}_pos_weights'] = torch.tensor(clipped_ratio, dtype=torch.float32, device=self.cuda)

            return weights

        pos_weights = _get_weights(pos_count_matrix_1, pos_count_matrix_0)
        neg_weights = _get_weights(neg_count_matrix_1, neg_count_matrix_0)
        
        return pos_weights, neg_weights

    def _get_losses_by_type(self, sample_type, loss_fns, latent, graph_embed, labels_tuple, pos_neg_sample, epsilon):
        """Calculates the sequential reconstruction loss for a given sub-batch."""
        (labels_full, labels_cat_onehot, labels_solv_1_onehot, labels_add, 
         labels_solv_2, labels_base_onehot, labels_water) = labels_tuple
        
        # Get integer labels for CrossEntropyLoss
        labels_cat = torch.argmax(labels_cat_onehot, dim=1)
        labels_base = torch.argmax(labels_base_onehot, dim=1)
        labels_solv_1 = torch.argmax(labels_solv_1_onehot, dim=1)
        
        all_losses = {}
        
        # Helper to select the correct loss function based on weighting scheme and sample type
        def select_loss(name, loss_type='CE'):
            if self.config.get("class_weights"):
                key = f'{name}_{sample_type}'
                return loss_fns[key]
            else:
                return loss_fns[f'{loss_type}_unweighted']

        if self.net.rtype == 'bh':
            # --- Catalyst ---
            cat_logits = self.net.decoder_cat([latent, graph_embed], pos_neg_sample)
            cat_logits = torch.clamp(cat_logits, -10, 10)
            all_losses['cat'] = select_loss('cat', 'CE')(cat_logits, labels_cat)
            next_cat_input = labels_cat_onehot if random.random() < epsilon else F.one_hot(torch.argmax(cat_logits.detach(), dim=1), num_classes=self.net.n_info[0]).float()
            
            # --- Base ---
            base_logits = self.net.decoder_base([latent, graph_embed, next_cat_input], pos_neg_sample)
            base_logits = torch.clamp(base_logits, -10, 10)
            all_losses['base'] = select_loss('base', 'CE')(base_logits, labels_base)
            next_base_input = labels_base_onehot if random.random() < epsilon else F.one_hot(torch.argmax(base_logits.detach(), dim=1), num_classes=self.net.n_info[4]).float()

            # --- Solvent 1 ---
            solv_1_logits = self.net.decoder_solv_1([latent, graph_embed, next_cat_input, next_base_input], pos_neg_sample)
            solv_1_logits = torch.clamp(solv_1_logits, -10, 10)
            all_losses['solv_1'] = select_loss('solv_1', 'CE')(solv_1_logits, labels_solv_1)
            next_solv_1_input = labels_solv_1_onehot if random.random() < epsilon else F.one_hot(torch.argmax(solv_1_logits.detach(), dim=1), num_classes=self.net.n_info[1]).float()

            # --- Water ---
            water_logits = self.net.decoder_water([latent, graph_embed, next_cat_input, next_base_input, next_solv_1_input], pos_neg_sample)
            water_logits = torch.clamp(water_logits, -10, 10)
            all_losses['water'] = select_loss('water', 'BCE')(water_logits, labels_water).sum(axis=1).mean()
            next_water_input = labels_water if random.random() < epsilon else (torch.sigmoid(water_logits.detach()) >= 0.5).float()
            
            # --- Additive ---
            add_logits = self.net.decoder_add([latent, graph_embed, next_cat_input, next_base_input, next_solv_1_input, next_water_input], pos_neg_sample)
            add_logits = torch.clamp(add_logits, -10, 10)
            all_losses['add'] = select_loss('add', 'BCE')(add_logits, labels_add).sum(axis=1).mean()

        elif self.net.rtype == 'sm':
            # --- Solvent 1 ---
            solv_1_logits = self.net.decoder_solv_1([latent, graph_embed], pos_neg_sample)
            solv_1_logits = torch.clamp(solv_1_logits, -10, 10)
            all_losses['solv_1'] = select_loss('solv_1', 'CE')(solv_1_logits, labels_solv_1)
            next_solv_1_input = labels_solv_1_onehot if random.random() < epsilon else F.one_hot(torch.argmax(solv_1_logits.detach(), dim=1), num_classes=self.net.n_info[1]).float()
            
            # --- Solvent 2 ---
            solv_2_logits = self.net.decoder_solv_2([latent, graph_embed, next_solv_1_input], pos_neg_sample)
            solv_2_logits = torch.clamp(solv_2_logits, -10, 10)
            all_losses['solv_2'] = select_loss('solv_2', 'BCE')(solv_2_logits, labels_solv_2).sum(axis=1).mean()
            next_solv_2_input = labels_solv_2 if random.random() < epsilon else (torch.sigmoid(solv_2_logits.detach()) >= 0.5).float()

            # --- Water ---
            water_logits = self.net.decoder_water([latent, graph_embed, next_solv_1_input, next_solv_2_input], pos_neg_sample)
            water_logits = torch.clamp(water_logits, -10, 10)
            all_losses['water'] = select_loss('water', 'BCE')(water_logits, labels_water).sum(axis=1).mean()
            next_water_input = labels_water if random.random() < epsilon else (torch.sigmoid(water_logits.detach()) >= 0.5).float()

            # --- Additive ---
            add_logits = self.net.decoder_add([latent, graph_embed, next_solv_1_input, next_solv_2_input, next_water_input], pos_neg_sample)
            add_logits = torch.clamp(add_logits, -10, 10)
            all_losses['add'] = select_loss('add', 'BCE')(add_logits, labels_add).sum(axis=1).mean()
            next_add_input = labels_add if random.random() < epsilon else (torch.sigmoid(add_logits.detach()) >= 0.5).float()

            # --- Catalyst ---
            cat_logits = self.net.decoder_cat([latent, graph_embed, next_solv_1_input, next_solv_2_input, next_water_input, next_add_input], pos_neg_sample)
            cat_logits = torch.clamp(cat_logits, -10, 10)
            all_losses['cat'] = select_loss('cat', 'CE')(cat_logits, labels_cat)
            next_cat_input = labels_cat_onehot if random.random() < epsilon else F.one_hot(torch.argmax(cat_logits.detach(), dim=1), num_classes=self.net.n_info[0]).float()

            # --- Base ---
            base_logits = self.net.decoder_base([latent, graph_embed, next_solv_1_input, next_solv_2_input, next_water_input, next_add_input, next_cat_input], pos_neg_sample)
            base_logits = torch.clamp(base_logits, -10, 10)
            all_losses['base'] = select_loss('base', 'CE')(base_logits, labels_base)
            
        return all_losses

    def training(self, train_loader, val_loader, max_epochs=250, calc_diversity=True):
        loss_fns = {}
        if self.config.get("class_weights", False):
            pos_weights, neg_weights = self._calculate_class_weights(
                self.config["train_pos_count_matrix_1"], 
                self.config["train_pos_count_matrix_0"],
                self.config["train_neg_count_matrix_1"],
                self.config["train_neg_count_matrix_0"]
            )
            for name, w in pos_weights.items():
                key = name.replace('_pos_weights', '_pos').replace('_weights', '_pos')
                if 'pos_weights' in name:
                    loss_fns[key] = nn.BCEWithLogitsLoss(pos_weight=w, reduction='none')
                else:
                    loss_fns[key] = nn.CrossEntropyLoss(weight=w, reduction='mean')
            for name, w in neg_weights.items():
                key = name.replace('_pos_weights', '_neg').replace('_weights', '_neg')
                if 'pos_weights' in name:
                    loss_fns[key] = nn.BCEWithLogitsLoss(pos_weight=w, reduction='none')
                else:
                    loss_fns[key] = nn.CrossEntropyLoss(weight=w, reduction='mean')
        
        # Always have unweighted fallbacks
        loss_fns['CE_unweighted'] = nn.CrossEntropyLoss(reduction='mean')
        loss_fns['BCE_unweighted'] = nn.BCEWithLogitsLoss(reduction='none')

        # Ensure lr and weight_decay are floats (WandB sometimes passes them as strings)
        lr = float(self.config["lr"])
        weight_decay = float(self.config["weight_decay"])
        optimizer = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        #patience as 20% of max_epochs
        patience = int(max_epochs * 0.2)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, min_lr=1e-6)
        
        val_log = np.zeros(max_epochs)

        for epoch in range(max_epochs):
            self.net.train()
            start_time = time.time()
            epsilon = self._calculate_epsilon(epoch)
            
            # Initialize epoch-level loss trackers for all components
            epoch_losses = {'total': 0., 'recon': 0., 'kld': 0., 'cat': 0., 'base': 0., 'solv_1': 0., 'solv_2': 0., 'add': 0., 'water': 0.}
            
            for batchidx, batchdata in enumerate(train_loader):
                inputs_rmol = [b.to(self.cuda) for b in batchdata[:2]]
                inputs_pmol = [b.to(self.cuda) for b in batchdata[2:3]]
                
                labels_tuple = [t.to(self.cuda) for t in batchdata[-2]]
                pos_neg_sample = batchdata[-1].to(self.cuda)
                
                latent, graph_embed, mu, log_var = self.net(inputs_rmol, inputs_pmol, labels_tuple[0], pos_neg_sample)

                all_losses_for_batch = []
                
                def process_sub_batch(mask, sample_type):
                    if not mask.any(): return
                    
                    sub_batch_labels = [t[mask] for t in labels_tuple]
                    sub_batch_losses_dict = self._get_losses_by_type(sample_type, loss_fns, latent[mask], graph_embed[mask], sub_batch_labels, pos_neg_sample[mask], epsilon)
                    
                    # Update epoch trackers and collect losses for backprop
                    for name, loss_val in sub_batch_losses_dict.items():
                        if loss_val is not None:
                            # Scale loss by the size of the sub-batch to normalize across the epoch
                            scaled_loss = loss_val * (mask.sum() / len(pos_neg_sample))
                            epoch_losses[name] += scaled_loss.item()
                            all_losses_for_batch.append(scaled_loss)

                if self.config.get("class_weights", False):
                    process_sub_batch((pos_neg_sample == 1).squeeze(-1), 'pos')
                    process_sub_batch((pos_neg_sample == 0).squeeze(-1), 'neg')
                else:
                    # In unweighted training, process the whole batch
                    process_sub_batch(torch.ones_like(pos_neg_sample, dtype=torch.bool).squeeze(-1), 'unweighted')

                if not all_losses_for_batch: continue
                
                loss_recon = torch.sum(torch.stack(all_losses_for_batch))
                #KL annealing to avoid  KL collapse
                kl_weight = min(1.0, epoch / self.config["kl_annealing_epochs"]) if self.config.get("kl_annealing_epochs", 0) else 1.0
                loss_kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(axis=1).mean()

                loss = (self.config["weight_bce"] / (self.config["weight_bce"] + 1)) * loss_recon + (1 / (self.config["weight_bce"] + 1)) * kl_weight * loss_kld
                
                if torch.isnan(loss):
                    warnings.warn(f"NaN loss detected at epoch {epoch}, batch {batchidx}. Skipping update.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses['total'] += loss.item()
                epoch_losses['recon'] += loss_recon.item()
                epoch_losses['kld'] += loss_kld.item()

            # --- End of epoch logging and validation ---
            num_batches = len(train_loader)
            if num_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches

            print(f"\n--- training epoch {epoch}, lr {optimizer.param_groups[0]['lr']:.2e}, loss {epoch_losses['total']:.3f}, time {(time.time()-start_time)/60:.2f} min")
            print(f"    Recon: {epoch_losses['recon']:.3f}, KLD: {epoch_losses['kld']:.3f}")
            print(f"    Cat: {epoch_losses['cat']:.3f}, Base: {epoch_losses['base']:.3f}, Solv1: {epoch_losses['solv_1']:.3f}, Solv2: {epoch_losses['solv_2']:.3f}, Add: {epoch_losses['add']:.3f}, Water: {epoch_losses['water']:.3f}")

            if self.config.get("wandb", False) and wandb is not None:
                log_data = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']}
                for key, value in epoch_losses.items():
                    log_data[f"loss_{key}"] = value # Add prefix to avoid clashes
                wandb.log(log_data) 
            
            # validation

            val_y = val_loader.dataset.y

            # create two val_y for positive and negative conditions
            val_y_pos = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 1] for sublist in val_y]
            val_y_neg = [[item_tuple[0] for item_tuple in sublist if item_tuple[1] == 0] for sublist in val_y]

            
            start_time = time.time()
            # Use configurable n_sampling for validation consistency
            n_sampling_val = self.config.get("n_sampling_val", 50)  # Default to 50 if not specified
            val_y_preds_pos, val_y_preds_neg = self.inference(val_loader, n_sampling = n_sampling_val, temperature=self.config["temperature"])
            
            if calc_diversity:
                avg_diversity_pos = np.mean(np.array([len(set(tuple(pred) for pred in i)) for i in val_y_preds_pos]))
                if self.config["verbose"]:
                    print(f"avg diversity pos at epoch {epoch}  ", avg_diversity_pos)
                if self.config["wandb"]:
                    wandb.log({"avg_diversity_pos": avg_diversity_pos})
                if self.config["data_type"] == "all":
                    avg_diversity_neg = np.mean(np.array([len(set(tuple(pred) for pred in i)) for i in val_y_preds_neg]))
                    if self.config["verbose"]:
                        print(f"avg diversity neg at epoch {epoch}  ", avg_diversity_neg)
                    if self.config["wandb"]:
                        wandb.log({"avg_diversity_neg": avg_diversity_neg})

            #If using both positive and negative samples, calculate diversity between the set of negative samples and the set of positive samples
            if self.config["data_type"] == "all":
                jaccard_distances = []
                # Iterate over each sample in the validation set
                for pos_preds, neg_preds in zip(val_y_preds_pos, val_y_preds_neg):
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

            if self.config["verbose"]:
                print(f"y pred pos validation at epoch {epoch}  ", [i[:5] for i in val_y_preds_pos[:8]])
                print(f"y true pos validation at epoch {epoch}  ", [i[:3] for i in val_y_pos[:10]])
                if self.config["data_type"] == "all":
                    print(f"y pred neg validation at epoch {epoch}  ", [i[:5] for i in val_y_preds_neg[:8]])
                    print(f"y true neg validation at epoch {epoch}  ", [i[:3] for i in val_y_neg[:10]])
            
            # Filter out empty lists for metrics calculation
            pos_pairs = zip(val_y_pos, val_y_preds_pos)
            filtered_pos_pairs = [(y_true, y_pred) for y_true, y_pred in pos_pairs if y_true]
            if filtered_pos_pairs:
                filtered_val_y_pos, filtered_val_y_preds_pos = map(list, zip(*filtered_pos_pairs))
            else:
                filtered_val_y_pos, filtered_val_y_preds_pos = [], []

            neg_pairs = zip(val_y_neg, val_y_preds_neg)
            filtered_neg_pairs = [(y_true, y_pred) for y_true, y_pred in neg_pairs if y_true]
            if filtered_neg_pairs:
                filtered_val_y_neg, filtered_val_y_preds_neg = map(list, zip(*filtered_neg_pairs))
            else:
                filtered_val_y_neg, filtered_val_y_preds_neg = [], []

            accuracy_pos = np.mean([np.max([(c in filtered_val_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) if filtered_val_y_pos else 0.0
            macro_recall_pos = np.mean([np.mean([(c in filtered_val_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) if filtered_val_y_pos else 0.0
            micro_recall_pos = np.sum([np.sum([(c in filtered_val_y_preds_pos[i]) for c in filtered_val_y_pos[i]]) for i in range(len(filtered_val_y_pos))]) / np.sum([len(a) for a in filtered_val_y_pos]) if filtered_val_y_pos and np.sum([len(a) for a in filtered_val_y_pos]) > 0 else 0.0

            accuracy_neg = np.mean([np.max([(c in filtered_val_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) if filtered_val_y_neg else 0.0
            macro_recall_neg = np.mean([np.mean([(c in filtered_val_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) if filtered_val_y_neg else 0.0
            micro_recall_neg = np.sum([np.sum([(c in filtered_val_y_preds_neg[i]) for c in filtered_val_y_neg[i]]) for i in range(len(filtered_val_y_neg))]) / np.sum([len(a) for a in filtered_val_y_neg]) if filtered_val_y_neg and np.sum([len(a) for a in filtered_val_y_neg]) > 0 else 0.0

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
                print(
                    f'--- validation at epoch {epoch}, '
                    f'ACC (+) {accuracy_pos:.3f}/1.000, '
                    f'macR (+) {macro_recall_pos:.3f}/1.000, '
                    f'micR (+) {micro_recall_pos:.3f}/1.000, '
                    f'ACC (-) {accuracy_neg:.3f}/1.000, '
                    f'macR (-) {macro_recall_neg:.3f}/1.000, '
                    f'micR (-) {micro_recall_neg:.3f}/1.000, '
                    f'time elapsed(min) {(time.time() - start_time) / 60:.2f}'
                )

                if self.config["wandb"]:
                    # Log metrics to wandb
                    wandb.log({
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "validation_loss": val_loss,
                        "accuracy_pos": accuracy_pos,
                        "macro_recall_pos": macro_recall_pos,
                        "micro_recall_pos": micro_recall_pos,
                        "accuracy_neg": accuracy_neg,
                        "macro_recall_neg": macro_recall_neg,
                        "micro_recall_neg": micro_recall_neg
                    })

            else:
                print(
                    f'--- validation at epoch {epoch}, '
                    f'ACC (+) {accuracy_pos:.3f}/1.000, '
                    f'macR (+) {macro_recall_pos:.3f}/1.000, '
                    f'micR (+) {micro_recall_pos:.3f}/1.000, '
                    f'time elapsed(min) {(time.time() - start_time) / 60:.2f}'
                )
                if self.config["wandb"]:
                    wandb.log({
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "validation_loss": val_loss,
                        "accuracy_pos": accuracy_pos,
                        "macro_recall_pos": macro_recall_pos,
                        "micro_recall_pos": micro_recall_pos,
                    })

            if np.argmin(val_log[:epoch + 1]) == epoch:
                torch.save(self.net.state_dict(), self.model_path) 
                print(f'--- model at epoch {epoch} saved to {self.model_path}')
                
                # Save model to WandB
                if self.config.get("wandb_save_model", True) and self.config["wandb"] and wandb.run:
                    wandb.save(self.model_path)
                    print(f'--- model saved to WandB: {self.model_path}')
            
            # elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            #     break
    
        print('training terminated at epoch %d' %epoch)
        print(f'best model saved at epoch {np.argmin(val_log)}')
        self.load()
        
    
    def inference(self, tst_loader, n_sampling = 100, temperature=1.0):
        """
        Performs inference on the test set, generating predictions for both positive and negative samples.
        Uses the sampling method which already makes the choices for each component.
        
        Returns:
            tuple: (tst_y_preds_pos, tst_y_preds_neg) where each is a list of length batch_size,
                  and each inner list contains n_sampling predictions, where each prediction
                  is a list of indices for the selected components.
                  
        Example structure:
        tst_y_preds_pos = [
            [  # First sample in batch
                [2, 5, 3, 1, 4],  # First sampling prediction
                [2, 5, 3, 1, 4],  # Second sampling prediction
                ...  # n_sampling predictions
            ],
            [  # Second sample in batch
                [1, 3, 2, 0, 5],  # First sampling prediction
                [1, 3, 2, 0, 5],  # Second sampling prediction
                ...  # n_sampling predictions
            ],
            ...  # batch_size samples
        ]
        """
        def forward(tst_loader, pos_neg_flag=1, config=self.config):
            """
            Helper function to generate predictions for a single forward pass.
            
            Returns:
                list: List of predictions for each sample in the batch, where each prediction
                      is a list of indices for the selected components.
            """
            tst_y_preds = []
            with torch.no_grad():
                for batchidx, batchdata in enumerate(tst_loader):
                    if not batchdata:
                        print("WARNING: Empty batch detected. Skipping this batch.")
                        continue
                    inputs_rmol = [b.to(self.cuda) for b in batchdata[:2]]
                    inputs_pmol = [b.to(self.cuda) for b in batchdata[2:3]]
                    
                    # Get batch size from the first graph's batch_size attribute
                    batch_size = inputs_rmol[0].batch_size
                    
                    # Create pos_neg_sample tensor with correct batch size
                    pos_neg_sample = torch.ones((batch_size, 1), device=self.cuda) if pos_neg_flag == 1 else torch.zeros((batch_size, 1), device=self.cuda)

                    # Get predictions from sampling (already makes choices)
                    preds = self.net.sampling(inputs_rmol, inputs_pmol, pos_neg_sample, config=config, temperature=temperature)
                    
                    # Convert predictions to indices for each sample in the batch
                    for i in range(batch_size):
                        # Get the indices where the prediction is 1 (selected components)
                        pred_indices = torch.nonzero(preds[i]).squeeze().cpu().numpy()
                        # Ensure that pred_indices is always a list, even for a single prediction
                        tst_y_preds.append(np.atleast_1d(pred_indices).tolist())
            
            if not tst_y_preds:
                print("WARNING: No predictions were made. Returning empty lists.")
                return []
            return tst_y_preds

        self.net.eval()   
        
        # Generate predictions for positive samples
        tst_y_preds_pos = []
        for _ in range(n_sampling):
            batch_preds = forward(tst_loader, 1)
            if not tst_y_preds_pos:  # First sampling
                if not batch_preds:
                    print("WARNING: No positive predictions were made. Returning empty list.")
                    break
                tst_y_preds_pos = [[pred] for pred in batch_preds]
            else:  # Subsequent samplings
                for i, pred in enumerate(batch_preds):
                    tst_y_preds_pos[i].append(pred)

        # Generate predictions for negative samples
        tst_y_preds_neg = []
        for _ in range(n_sampling):
            batch_preds = forward(tst_loader, 0)
            if not tst_y_preds_neg:  # First sampling
                if not batch_preds:
                    print("WARNING: No negative predictions were made. Returning empty list.")
                    break
                tst_y_preds_neg = [[pred] for pred in batch_preds]
            else:  # Subsequent samplings
                for i, pred in enumerate(batch_preds):
                    tst_y_preds_neg[i].append(pred)

        return tst_y_preds_pos, tst_y_preds_neg
