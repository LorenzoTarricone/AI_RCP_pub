import torch
import dgl
import numpy as np
import warnings
   
def collate_reaction_graphs_old(batch):

    batchdata = list(map(list, zip(*batch)))
    gs = [dgl.batch(s) for s in batchdata[:-1]]
    labels = torch.FloatTensor(np.array(batchdata[-1]))
    
    return *gs, labels

def collate_reaction_graphs(batch):
    """
    Efficient collate function for reaction graphs.
    Handles graph batching and label processing, similar to collate_graphs_and_embeddings.
    
    Args:
        batch (list): A list of tuples, where each tuple contains:
                     (g_r1, g_r2, g_p, label, pos_neg_sample)
    
    Returns:
        tuple: (*batched_graphs, processed_labels, pos_neg_sample)
    """
    if not batch:
        return None

    # Unzip the batch
    unzipped_batch = list(zip(*batch))
    num_components = len(unzipped_batch)
    
    if num_components < 3:  # Need at least graphs, labels, and pos_neg_sample
        raise ValueError("Batch items have fewer than 3 elements. Expected graphs/labels/pos_neg_flag.")

    # Process pos_neg_sample flag (last element)
    raw_flag = unzipped_batch[-1]
    if raw_flag[0] is not None:
        if not all(isinstance(flag, torch.Tensor) for flag in raw_flag):
            raise TypeError("Pos_neg_sample flag is not a tensor.")
        # Stack the tensors and ensure they have shape [batch_size, 1]
        flag_tensor = torch.stack(raw_flag)
        if flag_tensor.dim() == 1:
            flag_tensor = flag_tensor.unsqueeze(1)
        elif flag_tensor.dim() > 2:
            # If somehow we have more than 2 dimensions, flatten to [batch_size, 1]
            flag_tensor = flag_tensor.view(flag_tensor.size(0), -1)

    else:
        flag_tensor = None

    # Process labels (second to last element)
    labels_raw = unzipped_batch[-2]
    processed_labels = None
    first_label_item = labels_raw[0]

    if isinstance(first_label_item, tuple):  # Expanded labels
        # Process each component of the expanded labels in parallel
        transposed_labels = list(map(list, zip(*labels_raw)))
        processed_labels_list = []
        
        # Process all components at once using numpy for efficiency
        try:
            stacked_components = [np.array(component_list) for component_list in transposed_labels]
            processed_labels_list = [torch.FloatTensor(component) for component in stacked_components]
            processed_labels = tuple(processed_labels_list)
        except ValueError as e:
            print(f"Error stacking expanded label components: {e}")
            raise e
    elif isinstance(first_label_item, np.ndarray):  # Simple labels
        try:
            stacked_labels = np.array(labels_raw)
            processed_labels = torch.FloatTensor(stacked_labels)
        except ValueError as e:
            print(f"Error stacking simple numpy labels: {e}")
            raise e
    elif isinstance(first_label_item, list):  # Val/Test labels
        processed_labels = list(labels_raw)
    else:  # Fallback for other types
        try:
            processed_labels = torch.tensor(labels_raw, dtype=torch.long)
        except Exception as e:
            warnings.warn(f"Could not convert raw labels of type {type(first_label_item)} to tensor: {e}")
            processed_labels = list(labels_raw)

    # Process graphs (all elements before labels)
    num_graph_components = num_components - 2  # Exclude labels and pos_neg_sample
    batched_graphs = []
    
    # Process all graph components in parallel
    for i in range(num_graph_components):
        current_component_graphs = unzipped_batch[i]
        # Filter out None values and validate graph types
        graphs_to_batch = [g for g in current_component_graphs if g is not None]
        
        if graphs_to_batch:
            if not all(isinstance(g, dgl.DGLGraph) for g in graphs_to_batch):
                raise TypeError(f"Component {i} contains non-DGL graph objects.")
            try:
                # Batch only valid graphs
                batched_graphs.append(dgl.batch(graphs_to_batch))
            except Exception as e:
                print(f"Error batching graph component {i}: {e}")
                raise e
        else:
            batched_graphs.append(None)

    return (*batched_graphs, processed_labels, flag_tensor)


# --- NEW Simplified Collate Function ---
def collate_graphs_and_embeddings(batch):
    """
    Simplified collate function assuming each batch item contains:
    (*graphs, labels, pos_neg_sample, embeddings_tensor).

    Handles graph batching, label processing (simple array or tuple),
    and stacks embeddings.

    Args:
        batch (list): A list of tuples, where each tuple is the output
                      of GraphDataset.__getitem__ in graph+embedding mode.
                      Expected format: (g_r1, g_r2, g_p, label, pos_neg_sample, embedding)

    Returns:
        tuple: (*batched_graphs, processed_labels, embeddings_tensor)
               'processed_labels' might be a tensor or tuple depending on expand_data.
    """
    if not batch:
        return None

    # --- Unzip the batch ---
    # Assumes the structure: graphs, labels, pos_neg_sample, embedding
    unzipped_batch = list(zip(*batch))

    num_components = len(unzipped_batch)
    if num_components < 3: # Need at least labels and embeddings
        raise ValueError("Batch items have fewer than 3 elements. Expected graphs/labels/pos_neg_flag/embeddings.")

    # --- Process Embeddings (Last element) ---
    raw_embeddings = unzipped_batch[-1]
    if not all(isinstance(emb, torch.Tensor) and emb.ndim == 1 for emb in raw_embeddings):
        raise TypeError("Last element in batch items is not a 1D embedding tensor.")
    embeddings_tensor = torch.stack(raw_embeddings)
    # print("embedding tensor shape", embeddings_tensor.shape)

    # --- Process Positive-negative sample flag (Second to last element) ---
    raw_flag = unzipped_batch[-2]
    #This should be a 0 dimensional tensor for training and a 1 dimensional tensor for validation/testing (?)
    if raw_flag[0] is not None: #This would happen in the training where each entry of raw_flag is just a scalar with dimension 0
        if not all(isinstance(flag, torch.Tensor) for flag in raw_flag):
            raise TypeError("Second to last element in batch items is not a tensor.")
        # Stack the tensors and ensure they have shape [batch_size, 1]
        flag_tensor = torch.stack(raw_flag)
        if flag_tensor.dim() == 1:
            flag_tensor = flag_tensor.unsqueeze(1)
        elif flag_tensor.dim() > 2:
            # If somehow we have more than 2 dimensions, flatten to [batch_size, 1]
            flag_tensor = flag_tensor.view(flag_tensor.size(0), -1)

    else:
        flag_tensor = None #During validation and test we give the conditioning flag directly when iterating through batches


    # print("flag tensor shape ", flag_tensor.shape)


    # --- Process Labels (Third to last element) ---
    labels_raw = unzipped_batch[-3]
    processed_labels = None
    first_label_item = labels_raw[0]

    if isinstance(first_label_item, tuple): # Expanded labels (tuple of numpy arrays)
        transposed_labels = list(map(list, zip(*labels_raw)))
        processed_labels_list = []
        for component_list in transposed_labels:
            try:
                stacked_component = np.array(component_list)
                processed_labels_list.append(torch.FloatTensor(stacked_component))
            except ValueError as e:
                print(f"Error stacking expanded label component: {e}")
                raise e
        processed_labels = tuple(processed_labels_list)
    elif isinstance(first_label_item, np.ndarray): # Simple labels (numpy array)
        try:
            stacked_labels = np.array(labels_raw)
            processed_labels = torch.FloatTensor(stacked_labels)
        except ValueError as e:
            print(f"Error stacking simple numpy labels: {e}")
            raise e
    elif isinstance(first_label_item, list): # Val/Test labels (list of indices)
        processed_labels = list(labels_raw)
    else: # Fallback for other types (e.g., scalar indices)
        try:
             processed_labels = torch.tensor(labels_raw, dtype=torch.long)
        except Exception as e:
             warnings.warn(f"Could not convert raw labels of type {type(first_label_item)} to tensor: {e}")
             processed_labels = list(labels_raw)

    # --- Process Graphs (All elements before labels) ---
    num_graph_components = 3 # Exclude labels, pos_neg_sample indicator and embeddings
    batched_graphs = []
    if num_graph_components > 0:
        for i in range(num_graph_components):
            current_component_graphs = unzipped_batch[i]
            graphs_to_batch = [g for g in current_component_graphs if g is not None]
            if graphs_to_batch:
                if not all(isinstance(g, dgl.DGLGraph) for g in graphs_to_batch):
                     raise TypeError(f"Component {i} contains non-DGL graph objects.")
                try:
                    batched_graphs.append(dgl.batch(graphs_to_batch))
                except Exception as e:
                    print(f"Error batching graph component {i}: {e}")
                    raise e
            else:
                batched_graphs.append(None) # Placeholder if no valid graphs
    elif num_graph_components < 0:
         raise ValueError("Detected fewer than 2 elements per batch item.")
    # If num_graph_components is 0, batched_graphs remains empty []

    # --- Return the structured batch ---
    return (*batched_graphs, processed_labels, flag_tensor, embeddings_tensor)



def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass