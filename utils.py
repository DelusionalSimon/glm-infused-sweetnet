"""
Utility functions for the GLM-Infused SweetNet project.

Functions included:
    Data Loading Functions
        - build_multilabel_dataset
        - get_embeddings_from_state_dict
        - pickle_loader
    Preparation Functions
        - multilabel_split
        - prep_infused_sweetnet
        - prepare_tsne_data
    Utility Functions   
        - seed_everything

This file contains general utilities as well as functions for data loading and preparation of data and models. 
"""

# Standard library imports
from typing import List, Tuple, Union, Dict, Optional, Literal, Any 
from collections import Counter
import random
import pickle
import os 

# Third-party library imports
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
try:
    import torch
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
except ImportError:
  raise ImportError("<torch or torch_geometric missing; did you do 'pip install glycowork[ml]'?>")

# Glycowork dependencies
from glycowork.glycan_data.loader import build_custom_df, df_glycan, lib
from glycowork.ml.train_test_split import prepare_multilabel
from glycowork.ml.models import SweetNet, init_weights


# --- Data Loading Functions ---

def build_multilabel_dataset(glycan_dataset: str = 'df_species',
                          glycan_class: str = 'Kingdom',
                          min_class_size: int = 6,
                          silent: bool = False) -> Tuple[List[str], List[List[float]], List[str]]:
    """
    Loads glycan data, prepares it for multi-label classification, and filters it.

    Removes glycans with rare label combinations and filters out individual
    labels that have no positive examples in the remaining glycans.

    Parameters
    ----------
    glycan_dataset : str, optional, default = 'df_species'
        The glycowork dataset to use. Options include:
        - 'df_species'
        - 'df_tissue'
        - 'df_disease'
    glycan_class : str, optional, default = 'Kingdom'
        The class to predict from the chosen dataset. Options depend on
        `glycan_dataset`:
        - 'df_species': 'Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain', 'ref'
        - 'df_tissue': 'tissue_sample', 'tissue_species', 'tissue_id', 'tissue_ref'
        - 'df_disease': 'disease_association', 'disease_sample', 'disease_direction', 'disease_species', 'disease_id', 'disease_ref'
    min_class_size : int, optional, default = 6
        Minimum number of samples required for a specific multi-label combination
        to be included. Set to 1 to include all combinations.
    silent : bool, optional, default = False
        If True, suppresses print statements.

    Returns
    -------
    Tuple[List[str], List[List[float]], List[str]]
        A tuple containing:
        - glycan_sequences (List[str]): List of glycan strings after filtering.
        - binary_labels_filtered (List[List[float]]): List of corresponding
          multi-label binary vectors with columns for inactive labels removed.
        - label_names_filtered (List[str]): The ordered list of names for each
          position in the binary vectors, containing only labels with at
          least one positive example.

    """
    # Load data
    all_glycan_data = df_glycan

    # Build custom dataframe
    custom_glycan_df = build_custom_df(all_glycan_data, glycan_dataset)

    # Extract the list of unique individual labels from the chosen class from the custom_glycan_df
    # These are used to dechipher the labels when the model is used for prediction
    all_possible_label_names = sorted(list(custom_glycan_df[glycan_class].unique()))
    if not silent:
        print(f"Found {len(all_possible_label_names)} unique individual classes/labels.")

    # Populates the the label_names so that they are there even when filtering is disabled
    label_names = all_possible_label_names


    # Prepare for multi-label prediction
    glycans, labels = prepare_multilabel(custom_glycan_df, glycan_class)

    # if needed, removes classes with fewer than min_class_size samples.
    if(min_class_size > 1):

        # Convert labels to string representation for counting
        label_strings = [''.join(map(str, label)) for label in labels]

        # Count occurrences of each label combination
        label_counts = Counter(label_strings)

        # Filter glycans and labels based on class size
        glycan_sequences = [glycans[i] for i, label_str in enumerate(label_strings) if label_counts[label_str] >= min_class_size]
        binary_labels_unfiltered = [labels[i] for i, label_str in enumerate(label_strings) if label_counts[label_str] >= min_class_size]
        if not silent:
            print(f"Number of unique glycans left after filtering rare classes (size >= {min_class_size}): {len(glycan_sequences)}/{len(glycans)}")
        
        # Filter out individual labels with no positive examples after glycan filtering

        # Convert binary_labels to numpy array for easier column manipulation
        binary_labels_np = np.array(binary_labels_unfiltered)

        # Find indices of labels with at least one positive example
        # Sum across rows (axis=0) to get count for each label
        label_sums = binary_labels_np.sum(axis=0)
        active_label_indices = np.where(label_sums > 0)[0]

        # Create the final list of label names using the active indices
        # Use the initially generated sorted list (all_possible_label_names)
        # because its order matches the columns of binary_labels after prepare_multilabel
        label_names = [all_possible_label_names[i] for i in active_label_indices]

        # Create the final filtered binary label vectors, keeping only the active columns
        binary_labels = binary_labels_np[:, active_label_indices].tolist() # Convert back to list of lists

        if not silent:
            print(f"Number of unique labels left after filtering: {len(binary_labels[0])}")

    else:
        glycan_sequences = glycans
        binary_labels = labels
        if not silent:
            print(f"Number of unique glycans: {len(glycan_sequences)}")

    return glycan_sequences, binary_labels, label_names


def get_embeddings_from_state_dict(model_state_path: str) -> np.ndarray:
    """
    Loads a model's state dictionary and directly extracts its item_embedding weights
    without instantiating the full model.

    Parameters
    ----------
    model_state_path : str
        Path to the saved model state dictionary (.pth file).

    Returns
    -------
    np.ndarray
        The extracted item_embedding weights (as a NumPy array).
    
    Raises
    ------
        KeyError 
            - If the model state dictionary does not contain 'item_embedding.weight'.
    """
    state_dict = torch.load(model_state_path, map_location=torch.device('cpu'))
    
    # Directly access the embedding layer's weights by its key in the state_dict
    # This key must be exactly 'item_embedding.weight'
    if 'item_embedding.weight' in state_dict:
        embeddings_tensor = state_dict['item_embedding.weight']
        embeddings_np = embeddings_tensor.cpu().numpy()
        return embeddings_np
    else:
        raise KeyError(f"'{model_state_path}' does not contain 'item_embedding.weight'. Check model architecture.")
    

def pickle_loader(pickle_file_path: str, silent: bool = False) -> Any:
    """
    Load the contents of a pickle file.
    
    Parameters
    -------
    pickle_file_path : str 
        Path to the pickle file containing embeddings.
    silent : bool, optional
        If True, suppresses print statements. Default is False.    

    Returns:
    -------
    Any
        The object loaded from the pickle file.

    Raises:
    FileNotFoundError: 
        - If the specified file does not exist.
    Exception: 
        - If there is an error during loading the pickle file.
    """
    if os.path.exists(pickle_file_path):
        if not silent: 
            print(f"Loading object from: {pickle_file_path}")

        try:
            # Open the file in binary read mode ('rb')
            with open(pickle_file_path, 'rb') as file_handle:
                # Load the object(s) from the pickle file
                object = pickle.load(file_handle)

            if not silent: 
                print("Object loaded successfully!")        
            return object
        except Exception as e:
            raise Exception(f"An error occurred while loading the pickle file: {e}")
    else:
        raise FileNotFoundError(f"Error: File not found at '{pickle_file_path}'. Please check the filename and path.")



# --- Preparation Functions ---

def multilabel_split(glycans: List[str], # list of IUPAC-condensed glycans
                 labels: List[Union[float, int, str]], # list of prediction labels
                 train_size: float = 0.7, # size of train set, the rest is split into validation and test sets
                 random_state: int = 42, # random state for reproducibility
                 no_test: bool = False, # if True, only train and validation sets are returned
                 silent: bool = False # if True, suppresses print statements
                )-> Tuple[List[str], List[str], List[str], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Splits the data into training, validation, and testing sets using StratifiedShuffleSplit.

    Parameters
    ----------
    glycans : List[str]
        List of glycan strings (IUPAC-condensed).
    labels : List[Union[float, int, str]]
        List of label vectors or single labels for stratification. 
    train_size : float, optional, default = 0.7
        Proportion of the dataset to include in the training split.
        If no_test is True, the remaining data is the validation set
        Otherwise the remaining data is split equally into validation and test sets        
    random_state : int, optional, default = 42
        Controls the randomness of the split for reproducibility. 
    no_test : bool, optional, default = False
        If True, only the training and validation sets are created, and the test set is omitted.
    silent : bool, optional, default = False
        If True, suppresses print statements.
        If False, prints the sizes of the training, validation, and test sets.
    Returns
    -------
    Tuple[List[str], List[str], List[str], List[List[float]], List[List[float]], List[List[float]]]
        A tuple containing:
        - train_glycans (List[str]): Glycans for the training set.
        - val_glycans (List[str]): Glycans for the validation set.
        - test_glycans (List[str]): Glycans for the testing set.
        - train_labels (List[List[float]]): Labels for the training set.
        - val_labels (List[List[float]]): Labels for the validation set.
        - test_labels (List[List[float]]): Labels for the testing set.

    """
    
    # Convert labels to a suitable format for stratification (string representation)
    label_strings = [''.join(map(str, label)) for label in labels]

    # Initializing empty test sets for no_test = True
    test_glycans = []
    test_labels = []
    temp_glycans = glycans
    temp_labels = labels

    # calculating split ratios
    # I used to just split out the train set and then split the rest into val and test sets
    # but the sklearn StratifiedShuffleSplit that I use down the line requires a higher min_class_size then
    # for some arcane reason splits below 0.593 require a much higher min_class_size
    if not no_test:
        test_raio = (1 - train_size)/2
        train_ratio =(1 - train_size) / (1 + train_size)
    else: # If no_test is True, we only need to split into train and validation sets
        # If no_test is False, we need to split into train, validation, and test sets
        train_ratio = 1 - train_size
    # If no_test is False, we need to split out the test set first
    if not no_test:
        # Initial split for train+val vs. test
        sss = StratifiedShuffleSplit(n_splits = 1, train_size = test_raio, random_state = random_state)
        test_index, temp_index = next(sss.split(glycans, label_strings))
        test_glycans = [glycans[i] for i in test_index]
        test_labels = [labels[i] for i in test_index]
        temp_glycans = [glycans[i] for i in temp_index]
        temp_labels = [labels[i] for i in temp_index]
        label_strings = [''.join(map(str, label)) for label in temp_labels]

    # Split the remaining train+val into validation and train sets
    sss_val_test = StratifiedShuffleSplit(n_splits = 1, train_size = train_ratio, random_state = random_state)
    val_index, train_index = next(sss_val_test.split(temp_glycans, label_strings))
    val_glycans = [temp_glycans[i] for i in val_index]
    val_labels = [temp_labels[i] for i in val_index]
    train_glycans = [temp_glycans[i] for i in train_index]
    train_labels = [temp_labels[i] for i in train_index]
    if not silent:
        print("Split complete!")
        print(f"Train set size: {len(train_glycans)}")
        print(f"Validation set size: {len(val_glycans)}")
        print(f"Test set size: {len(test_glycans)}")
        
    return train_glycans, val_glycans, test_glycans, train_labels, val_labels, test_labels


def prep_infused_sweetnet(num_classes: int, # number of unique classes for classification
                           embeddings_dict: Optional[Dict[str, np.ndarray]] = None, # embeddings for 'external' method
                           initialization_method: Literal['external', 'random', 'one_hot'] = 'external', # specifies initialization method
                           trainable_embeddings: bool = True, # whether the external embeddings should be trainable
                           hidden_dim: int = 320, # hidden dimension for the model (be sure to match dimension of embeddings)  
                           libr: Optional[Dict[str, int]] = None, # dictionary of form glycoletter:index
                           silent: bool = False # if True, suppresses print statements
                          ) -> torch.nn.Module:
    """
    Instantiates and prepares a SweetNet model with specified embedding initialization.

    Parameters
    ----------
    num_classes : int
        Number of unique classes for classification. (REQUIRED)
    embeddings_dict : Optional[Dict[str, np.ndarray]], optional, default = None
        The loaded external embeddings dictionary {glycan_word: embedding_vector}.
        Required if initialization_method is 'external'.
    initialization_method : {'external', 'random', 'one_hot'}, optional, default = 'external'
        The method to initialize the embedding layer:
        - 'external': Initialize with embeddings from embeddings_dict.
        - 'random': Randomly initialized embeddings (train from scratch).
        - 'one_hot': Initialize with one-hot encoded vectors. (not implemented yet).
    trainable_embeddings : bool, optional, default = True
        Whether the embedding layer should be trainable during training.
    hidden_dim : int, optional, default = 320
        Dimension of hidden layers. Must match the dimension of the embeddings
        used if initialization_method is 'external'.
    libr : Optional[Dict[str, int]], optional, default = None
        Dictionary of form glycoletter:index.
        If None, the standard glycowork library is used. 
    silent : bool, optional, default = False
        If True, suppresses print statements.

    Returns
    -------
    torch.nn.Module
        An initialized PyTorch model (SweetNet).

    Raises
    ------
    ValueError
        - If initialization_method is 'external' but embeddings_dict is None.
        - If initialization_method is 'external' and embedding dimension does not match hidden_dim.
        - If initialization_method is 'one_hot' and hidden_dim does not match library size.
        - If initialization_method is 'random' or 'one_hot' and libr is None.
        - If an unknown initialization_method is provided.

    """

    #  Check if libr is provided, if not, use the default library
    if libr is None:
        libr = lib

    # Check if the required components are available in the current context
    if 'SweetNet' not in globals() or not callable(globals()['SweetNet']) or \
       'init_weights' not in globals() or not callable(globals()['init_weights']):
         raise ValueError("Required glycowork components (SweetNet, init_weights) not available or not callable. Please ensure they are imported correctly.")

    # Instantiate the SweetNet model
    model = SweetNet(lib_size=len(libr), num_classes=num_classes, hidden_dim=hidden_dim)
    if not silent:
        print(f"SweetNet model instantiated with lib_size={len(libr)}, num_classes={num_classes}, hidden_dim={hidden_dim}.")


    # Apply initial weights to all layers (embedding and non-embedding)
    model = model.apply(lambda module: init_weights(module, mode = 'sparse')) # Experiment with 'sparse', 'xavier', or 'kaiming'
    
    # move model to the device (CPU or GPU)
    model = model.to(device)
    
    if initialization_method == 'external':
        if not silent:
            print("Handling 'external' initialization method.")
        
        # Check if embeddings_dict is provided
        # If not, raise an error (this is a required parameter for 'external' method)
        if embeddings_dict is None:
            raise ValueError("embeddings_dict must be provided when initialization_method is 'external'.")
        
        # Check that the dimension of the embeddings_dict matches hidden_dim
        embedding_key = next(iter(embeddings_dict))
        external_embedding_dim = embeddings_dict[embedding_key].shape[0]
        if external_embedding_dim != hidden_dim:
             raise ValueError(f"External embedding dimension ({external_embedding_dim}) must match model's hidden_dim ({hidden_dim}).")
        
        # Get the tensor of the embedding layer's weights. It already has initial random values.
        embedding_tensor_to_populate = model.item_embedding.weight.data

        #
        with torch.no_grad():
                 # Iterate through the library (which gives us the index for each glycan word)
                 for glycan_word, index in libr.items():
                    if glycan_word in embeddings_dict:

                        # Get the embedding vector from the dictionary
                        embedding_vector = embeddings_dict[glycan_word] 

                        # Convert to tensor, ensure correct dtype, and move to the same device
                        embedding_vector_tensor = torch.tensor(embedding_vector, dtype=torch.float32).to(embedding_tensor_to_populate.device)

                        # Copy the vector into the correct row of the model's embedding tensor
                        # Relying on index from libr being valid for embedding_tensor_to_populate size
                        embedding_tensor_to_populate[index].copy_(embedding_vector_tensor)

                    else:
                        # If a glycan word in libr is NOT in embeddings_dict, its initial random value is preserved (for smaller dictionaries).
                        if not silent:
                            print(f"{glycan_word} is not in library, keeping its initial random value.")
                        pass # Explicitly do nothing, keeping the initial random value
         

    elif initialization_method == 'random':
        if not silent:
            print("Handling 'random' initialization method (training from scratch).")
        
        # The item_embedding layer was already initialized randomly by the standard initialization loop above.

        pass 

        
    elif initialization_method == 'one_hot':
        if not silent:
            print(" 'one_hot' initialization method not implemented yet")
        
    # either I need the hidden_dim to be the same as the number of glycoletters in the library
    # or I need to find a way to reduce the dimensionality of the one-hot encoding to match the hidden_dim
    #or do what Roman suggested and reduce the dictionary to the 319 most common glycowords
    # Let's tackle this later given time 


        # Determine the required embedding dimension for one-hot encoding
        #required_hidden_dim = len(libr) + 1

        # Create the one-hot embedding matrix (identity matrix)
        #one_hot_matrix = torch.eye(required_hidden_dim, dtype=torch.float32)

        # Copy one_hot_matrix to model.item_embedding.weight.data
        #one_hot_matrix = one_hot_matrix.to(model.item_embedding.weight.device)

        # Copy the one-hot matrix into the model's item_embedding.weight.data
        #with torch.no_grad():
           # model.item_embedding.weight.copy_(one_hot_matrix)
            
        pass

    else:
        # This case should ideally be caught by the Literal type hint and docstring,
        # but adding a runtime check is robust.
        raise ValueError(f"Unknown initialization_method: {initialization_method}")


    # Set trainability based on trainable_embeddings flag (outside the branches)
    # This happens AFTER the initialization logic in the branches above.
    # The logic for setting requires_grad is the same regardless of initialization method.
    model.item_embedding.weight.requires_grad = trainable_embeddings
    if not silent:
        print(f"SweetNet item_embedding layer set to trainable: {trainable_embeddings}.")

    
    # Return the model
    return model


def prepare_tsne_data(embedding_arrays: List[np.ndarray], 
                      embedding_names: List[str],
                      normalize: bool = True
                      ) -> Tuple[np.ndarray, List[str]]:
    """
    Prepares embedding data for t-SNE visualization by normalizing and concatenating.

    Parameters
    ----------
    embedding_arrays : List[np.ndarray]
        A list of NumPy arrays, where each array contains a set of embeddings
        (e.g., [baseline_embs, raw_glm_embs, infused_embs]).
        All arrays in the list must have the same number of rows (glycowords)
        and same number of columns (embedding dimensions).
    embedding_names : List[str]
        A list of string names corresponding to each array in `embedding_arrays`.
        These names will be used as labels in the t-SNE plot legend.
    normalize : bool, optional
        If True, each embedding array will be normalized before concatenation.
        Default is True.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A tuple containing:
        - tsne_embeddings (np.ndarray): All input embeddings, normalized and vertically stacked.
        - tsne_labels (List[str]): Corresponding labels for each row in all_embs_for_tsne.

    Raises
    ------
    ValueError
        If the number of embedding arrays does not match the number of names,
        are empty, don't have the same number of rows,
        or if the arrays have inconsistent shapes.
    Exception
        If an error occurs during normalization.
    """
    if len(embedding_arrays) != len(embedding_names):
        raise ValueError("Number of embedding arrays must match number of embedding names.")
    if not embedding_arrays: # Handle empty input list
        raise ValueError("No embedding arrays provided.")

    # Get the number of glycowords (rows) from the first embedding array
    num_glycowords = embedding_arrays[0].shape[0]

    
    # Initialize a list to hold the arrays that will be concatenated
    arrays_to_concatenate = []

    # Normalize each embedding array and collect them if that flag is set
    for arr in embedding_arrays:
        if arr.shape[0] != num_glycowords:
            raise ValueError("All embedding arrays must have the same number of rows (glycowords).")
        
        if normalize:
            try:
                # Normalization logic
                normalized_arr = arr / np.max(np.linalg.norm(arr, axis=1, keepdims=True))
                arrays_to_concatenate.append(normalized_arr)
            except Exception as e:
                raise Exception(f"Error normalizing array: {e}")
        else:
            arrays_to_concatenate.append(arr) # If not normalizing, just add the original array
            
    # Concatenate all (potentially normalized) arrays vertically
    tsne_embeddings = np.concatenate(arrays_to_concatenate, axis=0)
    
    # Create the combined list of labels
    tsne_labels = []
    for name in embedding_names:
        tsne_labels.extend([name] * num_glycowords) # Extend with 'num_glycowords' repetitions of each name

    return tsne_embeddings, tsne_labels

# --- Utility Functions ---

def seed_everything(seed: int,  silent: int =  False, full_reproducibility: bool = True) -> None:   
    """
    Set all random seeds for reproducibility.

    Ensures that operations involving randomness (e.g., data splitting,
    model weight initialization, some PyTorch/NumPy operations)
    produce the same results across different runs when the same
    seed is provided.

    Parameters
    ----------
    seed : int
        The seed value to use for all random number generators.
    silent : bool, optional, default = False
        If True, suppresses print statements.
        If False, prints the seed value being set.
    full_reproducibility : bool, optional, default = True
        If True, sets additional PyTorch settings for full reproducibility.
        This may affect performance but ensures that results are consistent
        across different runs and hardware configurations.
    
    """
      
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # These settings are not recommended for performance, but are necessary for true reproducibility.
        if full_reproducibility:
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    if not silent:
        print(f"All random seeds set to: {seed}")



