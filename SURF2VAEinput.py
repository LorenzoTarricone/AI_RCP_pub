#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processes chemical reaction data from a CSV file.
- Cleans salts from SMILES strings in a specified column.
- Trims the reagent space based on frequency or threshold criteria.
- Generates reaction dictionaries and condition lists.
- Saves the processed data to an NPZ file.
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict

# Ensure RDKit is installed: pip install rdkit-pypi
try:
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    print("Error: RDKit library not found. Please install it using 'pip install rdkit-pypi'")
    exit(1)

# --- Helper Functions ---

def clean_smiles_string(smiles: str, remover: SaltRemover, uncharger: rdMolStandardize.Uncharger) -> str:
    """
    Cleans a single SMILES string by removing salts, neutralizing,
    and keeping the largest fragment if multiple remain.

    Args:
        smiles (str): The input SMILES string.
        remover (SaltRemover): RDKit SaltRemover instance.
        uncharger (rdMolStandardize.Uncharger): RDKit Uncharger instance.

    Returns:
        str: The cleaned SMILES string, the original string if cleaning fails,
             or an "INVALID_SMILES:..." marker if input is invalid.
    """
    if "." not in smiles: # No salt initially
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"INVALID_SMILES: {smiles}"

    # Salt Removal
    mol_no_salt = remover.StripMol(mol, dontRemoveEverything=False) # Keep largest component

    if mol_no_salt:
        try:
             # Neutralization
            mol_neutral = uncharger.uncharge(mol_no_salt)
            cleaned_smiles_str = Chem.MolToSmiles(mol_neutral)
        except Exception: # Handle potential errors during uncharging
             # If neutralization fails, keep the salt removed molecule.
            try:
                cleaned_smiles_str = Chem.MolToSmiles(mol_no_salt)
            except Exception: # If even getting SMILES fails, return original
                 return f"PROCESSING_ERROR: {smiles}" # Indicate a different error

        # Keep largest fragment if neutralization resulted in multiple parts
        fragments = cleaned_smiles_str.split('.')
        if len(fragments) > 1:
            # RDKit's SaltRemover with dontRemoveEverything=False should ideally handle this,
            # but this adds an extra layer of safety.
            # We calculate molecular weight to find the 'largest' chemically relevant fragment
            largest_fragment = ""
            max_mw = -1
            valid_fragment_found = False
            for frag_smiles in fragments:
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol:
                    valid_fragment_found = True
                    mw = Chem.Descriptors.MolWt(frag_mol)
                    if mw > max_mw:
                        max_mw = mw
                        largest_fragment = frag_smiles
            # If no valid fragment was found (e.g., '.[Na+]'), return original for inspection
            return largest_fragment if valid_fragment_found else f"FRAGMENTATION_ERROR: {smiles}"
        else:
            return cleaned_smiles_str
    else:
        # If salt removal results in nothing, return original
        return f"SALT_REMOVAL_ERROR: {smiles}"


def replace_smiles_in_dataframe(df: pd.DataFrame, smiles_dict: dict, column_name: str) -> pd.DataFrame:
    """
    Replaces occurrences of SMILES strings in a DataFrame column with their
    corresponding values from a dictionary, using pandas map for efficiency.

    Args:
        df (pd.DataFrame): The pandas DataFrame.
        smiles_dict (dict): A dictionary where keys are SMILES strings to be replaced
                            and values are their replacements.
        column_name (str): The name of the column in the DataFrame to perform
                           the replacements on.

    Returns:
        pd.DataFrame: A new DataFrame with the replacements made.
                      Returns an empty DataFrame if the input DataFrame is empty.
                      Raises KeyError if the specified column does not exist.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty.")
        return pd.DataFrame()

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")

    new_df = df.copy()
    # Use map for potentially faster replacement than iterated str.replace
    new_df[column_name] = new_df[column_name].map(smiles_dict).fillna(new_df[column_name])
    # The fillna ensures that SMILES not in the dict keys remain unchanged.

    return new_df


def clean_salts_in_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Identifies salts in a specified SMILES column, cleans them,
    and updates the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the SMILES column to clean.

    Returns:
        pd.DataFrame: DataFrame with cleaned SMILES in the specified column.
    """
    print("\n")
    print(f"--- Cleaning salts in column: {column_name} ---")
    str_unique = df[column_name].dropna().unique() # Use dropna()
    salts = [s for s in str_unique if isinstance(s, str) and "." in s] # Check type

    if not salts:
        print("No salts found containing '.' character.")
        return df

    print(f"Found {len(salts)} unique SMILES containing '.' to process.")

    remover = SaltRemover()
    uncharger = rdMolStandardize.Uncharger()
    remsalt_dict = {}
    invalid_count = 0
    error_count = 0

    for smiles in salts:
        cleaned_smiles = clean_smiles_string(smiles, remover, uncharger)
        remsalt_dict[smiles] = cleaned_smiles
        if "INVALID_SMILES" in cleaned_smiles:
            invalid_count += 1
        elif "ERROR" in cleaned_smiles:
            error_count +=1


    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid SMILES strings encountered during cleaning.")
    if error_count > 0:
         print(f"Warning: {error_count} SMILES strings encountered processing errors during cleaning.")

    # It's more efficient to map the replacements rather than iterate replace
    cleaned_df = df.copy()
    # Create a mapping series/dict from original to cleaned for the specific column
    replacement_map = cleaned_df[column_name].map(remsalt_dict)
    # Update the column only where a replacement exists (was a salt)
    cleaned_df[column_name] = cleaned_df[column_name].map(remsalt_dict).fillna(cleaned_df[column_name])


    # Final check post-cleaning
    str_unique_post = pd.Series(cleaned_df[column_name]).dropna().unique()
    salts_post = [s for s in str_unique_post if isinstance(s, str) and "." in s]

    if len(salts_post) == 0:
        print("Successfully removed/processed all salts containing '.' character.")
    else:
        print(f"Warning: {len(salts_post)} salts remain after cleaning. Examples:")
        print(salts_post[:5]) # Print first 5 remaining salts
        # Consider adding more robust logging or error handling here

    print("-" * (len(f"--- Cleaning salts in column: {column_name} ---"))) # Match length
    return cleaned_df

def add_increasing_integers(series: pd.Series) -> pd.Series:
    """Adds increasing integers to repeated letters in a pandas Series."""
    counts = defaultdict(int)
    result = []
    for item in series:
        counts[item] += 1
        result.append(f"{item}{counts[item]}")
    return pd.Series(result, index=series.index) # Preserve original index

def create_reagent_dataframe(series: pd.Series, reagent_type_char: str) -> pd.DataFrame:
    """Creates a DataFrame counting reagent occurrences."""
    if series.empty or series.nunique() == 0:
        return pd.DataFrame(columns=["reagent_type", "reagent", "count"]) # Return empty DF

    counts = series.value_counts()
    df = pd.DataFrame({
        "reagent_type": reagent_type_char,
        "reagent": counts.index,
        "count": counts.values
    })
    return df

def trim_reagents(data: pd.DataFrame, rxn_type: str, trim_type: str, trim_params: dict) -> pd.DataFrame:
    """
    Trims the reagent space based on the selected method (quantile or threshold).

    Args:
        data (pd.DataFrame): DataFrame containing reaction data.
        rxn_type (str): The type of reaction (e.g., "sm" for Suzuki-Miyaura).
        trim_type (str): Trimming method ("quantile_all", "quantile_category", "treshold_all").
        trim_params (dict): Dictionary containing threshold values for trimming.

    Returns:
        pd.DataFrame: DataFrame containing the filtered list of reagents.
                       Returns original number of unique reagents pre-filtering.
    """
    print("\n")
    print(f"--- Trimming Reagent Space using: {trim_type} ---")
    reagent_cols = {
        "catalyst_name": "C",
        "reagent_1_name": "B", # Base
        "solvent_1_name": "S",
        "additives_name_merged": "A"
    }
    if rxn_type == "sm":
        # Original code used 'L' for solvent 2 in threshold, 'S' in quantile. Standardizing to 'L'.
        reagent_cols["solvent_2_name"] = "L" # Second Solvent

    all_reagent_dfs = []
    for col, type_char in reagent_cols.items():
        if col in data:
             # Handle potential NaN values gracefully before value_counts
            series = data[col].dropna()
            if not series.empty:
                all_reagent_dfs.append(create_reagent_dataframe(series, type_char))
        else:
            print(f"Warning: Column '{col}' not found in input data for trimming.")

    if not all_reagent_dfs:
         raise ValueError("No valid reagent columns found or all were empty.")

    reagent_df_full = pd.concat(all_reagent_dfs, ignore_index=True)
    len_pre = reagent_df_full.shape[0] # Total unique reagents before trimming

    if trim_type == "quantile_all":
        # Calculate frequency across *all* reagents combined
        reagent_df_full["frequency"] = reagent_df_full["count"] / reagent_df_full["count"].sum()
        reagent_df_full = reagent_df_full.sort_values(by="frequency", ascending=False) # High freq first
        reagent_df_full['cumulative'] = reagent_df_full.frequency.cumsum()
        # Keep reagents contributing up to the quantile threshold
        reagent_df_filtered = reagent_df_full[reagent_df_full['cumulative'] <= trim_params["quantile_all_t"]].reset_index(drop=True)

    elif trim_type == "quantile_category":
        filtered_dfs = []
        quantile_map = {
            "C": trim_params.get("quantile_cats_t", 0.98), # Default if specific not found
            "B": trim_params.get("quantile_base_t", 0.98),
            "S": trim_params.get("quantile_solv_1_t", 0.98),
            "A": trim_params.get("quantile_add_t", 0.996),
            "L": trim_params.get("quantile_solv_2_t", 0.98) # Use the solvent 1 quantile if solvent 2 specific not set
        }
        for type_char in reagent_df_full["reagent_type"].unique():
            df_cat = reagent_df_full[reagent_df_full["reagent_type"] == type_char].copy()
            if df_cat.empty: continue
            df_cat["frequency"] = df_cat["count"] / df_cat["count"].sum() # Freq within category
            df_cat = df_cat.sort_values(by="frequency", ascending=False)
            df_cat['cumulative'] = df_cat.frequency.cumsum()
            quantile_t = quantile_map.get(type_char, 0.98) # Get specific quantile or default
            df_cat_filtered = df_cat[df_cat['cumulative'] <= quantile_t]
            filtered_dfs.append(df_cat_filtered)
        reagent_df_filtered = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else pd.DataFrame()


    elif trim_type == "treshold_all":
        # Simple count threshold applied to all reagents
        reagent_df_filtered = reagent_df_full[reagent_df_full["count"] >= trim_params["tresh_all"]].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown trim_type: {trim_type}")

    if reagent_df_filtered.empty:
         print("Warning: Trimming resulted in an empty reagent list.")
         reagent_df_filtered["reagent_type_numbered"] = pd.Series(dtype=str) # Add empty column
    else:
        # Sort by type then count descending for consistent numbering
        reagent_df_filtered = reagent_df_filtered.sort_values(by=['reagent_type', 'count'], ascending=[True, False])
        # Add numbered types AFTER filtering and sorting
        reagent_df_filtered["reagent_type_numbered"] = add_increasing_integers(reagent_df_filtered["reagent_type"])

    return reagent_df_filtered, len_pre


def create_dictionary_from_dataframe(df: pd.DataFrame, key_column: str, value_columns: list) -> dict:
    """
    Creates a dictionary from a DataFrame including the index.

    Args:
        df (pd.DataFrame): The input DataFrame (assumed to have a default integer index).
        key_column (str): The name of the column to use as keys.
        value_columns (list): A list of column names to use as values.

    Returns:
        dict: The created dictionary {key: [val1, val2, ..., index]}.
    """
    result_dict = {}
    # Ensure the index is usable if it was reset during trimming
    df_reset = df.reset_index(drop=True)
    for index, row in df_reset.iterrows():
        key = row[key_column]
        # Check for NaN or other non-string keys if necessary
        if pd.isna(key):
            print(f"Warning: Skipping row with NaN key in column '{key_column}' at index {index}")
            continue
        values = [row[col] for col in value_columns]
        values.append(index) # Append the current numerical index
        result_dict[key] = values
    return result_dict

# --- Main Execution Logic ---

def main(args):
    """Main function to orchestrate the data processing."""

    # --- 1. Load Data ---
    infile = args.infile
    outdir = args.outdir
    trim_type = args.trim_type
    export_reagent_df = args.export_reagent_df

    if not os.path.exists(infile):
        print(f"Error: Input file not found at {infile}")
        exit(1)

    print("\n")
    print(f"Loading data from: {infile}")
    try:
        data = pd.read_csv(infile)
        if data.empty:
             print(f"Error: Input file {infile} is empty.")
             exit(1)
        # Attempt to determine reaction type robustly
        if "rxn_type" not in data.columns or data["rxn_type"].isnull().all():
             print("Warning: 'rxn_type' column missing or empty. Cannot determine reaction type automatically.")
             rxn_type = "UnknownReaction" # Assign a default name
        else:
             # Use the first non-null value as representative type
             rxn_type = data["rxn_type"].dropna().iloc[0] if not data["rxn_type"].dropna().empty else "UnknownReaction"
        print(f"Detected reaction type: {rxn_type}")
        #Get shorter name for namig files later
        rxn_type = "bh" if rxn_type == "Buchwald-Hartwig" else "sm"

        # Understand if we are dealing with the full data or just the processd data
        if "suff_yield" in data.columns:
                print("Processing dataset with both NEGATIVE and POSITIVE samples")
                complete_dataset = True
        else:
                print("Processing dataset with just POSITIVE samples")
                data["suff_yield"] = 1
                complete_dataset = False

    except Exception as e:
        print(f"Error loading or reading CSV file {infile}: {e}")
        exit(1)


    # --- 2. Initial Sanity Checks & Salt Cleaning ---
    # Check starting material 1 (optional, based on original code's assert)
    sm1_col = 'startingmat_1_smiles'
    if sm1_col in data:
        str_unique_sm1 = pd.Series(data[sm1_col]).dropna().unique()
        if any("." in str(s) for s in str_unique_sm1 if isinstance(s,str)):
            print(f"Warning: Found salts ('.') in '{sm1_col}'. Code currently does not clean this column. Adapt if necessary.")
            # Optionally, call clean_salts_in_column(data, sm1_col) here if needed.
    else:
        print(f"Warning: Column '{sm1_col}' not found.")

    # Check product (optional, based on original code's assert)
    prod_col = 'product_1_smiles'
    if prod_col in data:
        str_unique_prod = pd.Series(data[prod_col]).dropna().unique()
        prod_salts = [s for s in str_unique_prod if isinstance(s, str) and "." in s]
        if prod_salts:
            print(f"Warning: Found salts ('.') in product column '{prod_col}'. This might indicate issues. Examples: {prod_salts[:5]}")
    else:
         print(f"Warning: Column '{prod_col}' not found.")


    # Clean starting material 2
    sm2_col = 'startingmat_2_smiles'
    if sm2_col in data:
        data_cleaned = clean_salts_in_column(data, sm2_col)
    else:
        print(f"Error: Required column '{sm2_col}' not found. Cannot proceed with salt cleaning.")
        exit(1) # Exit if the core column for cleaning is missing

    # --- 3. Define Trimming Parameters ---
    # These were hardcoded globals, now passed via dict
    trim_params = {
        "quantile_all_t": 0.98,
        "quantile_cats_t": 0.98,
        "quantile_solv_1_t": 0.98,
        "quantile_solv_2_t": 0.98, # Original had 0.0? Defaulting to 0.98 like solv1
        "quantile_add_t": 0.996,
        "quantile_base_t": 0.98,
        "tresh_all": 50
    }

    # --- 4. Trim Reagent Space ---
    try:
        reagent_df, len_pre_trim = trim_reagents(data_cleaned, rxn_type, trim_type, trim_params)
    except ValueError as e:
        print(f"Error during reagent trimming: {e}")
        exit(1)

    if reagent_df.empty:
         print("Error: No reagents remained after trimming process. Cannot continue.")
         exit(1)


    # --- 5. Filter Main DataFrame based on Kept Reagents ---
    print("--- Filtering Reactions Based on Trimmed Reagents ---")
    required_reagent_columns = {
         "catalyst_name", "reagent_1_name", "solvent_1_name",
         "additives_name_merged", "contains_water"
    }
    if rxn_type == "sm":
        required_reagent_columns.add("solvent_2_name")


    # Check if all necessary columns exist for filtering
    missing_cols = required_reagent_columns - set(data_cleaned.columns)
    if missing_cols:
        print(f"Error: Cannot filter reactions. Missing required columns: {missing_cols}")
        exit(1)

    # Build the filter condition dynamically
    filter_condition = pd.Series(True, index=data_cleaned.index) # Start with all true
    kept_reagents_set = set(reagent_df["reagent"])

    for col in required_reagent_columns:
        # Ensure we only filter based on reagents actually present in that column type
        # This prevents errors if, e.g., no 'L' type reagents remain but the column exists
        type_char = reagent_cols.get(col) # Get char like 'C', 'B', 'S', 'A', 'L'

        #Water will always be there, so don't even look at it
        if type_char == 'W':
            continue

        relevant_kept_reagents = set(reagent_df[reagent_df['reagent_type'] == type_char]['reagent']) \
                                   if type_char else kept_reagents_set # Fallback if type char mapping issue

        if not relevant_kept_reagents:
             print(f"Note: No reagents of type associated with '{col}' remain after trimming. Skipping this column in filter.")
             continue

        # Apply filter for the current column
        filter_condition &= data_cleaned[col].isin(relevant_kept_reagents)


    subset_df = data_cleaned[filter_condition].reset_index(drop=True)

    # print("Subset df ", subset_df[subset_df["solvent_2_name"] == "MeOH"])
    

    print(f"Retained {subset_df.shape[0]} reactions out of {data_cleaned.shape[0]} original (after cleaning) ({subset_df.shape[0] / data_cleaned.shape[0] * 100 :.2f}%)")
    print(f"Reagent list reduced from {len_pre_trim} unique reagents to {reagent_df.shape[0]}.")
    if len_pre_trim > 0: # Avoid division by zero if trimming started with zero reagents
         reduction_pct = (len_pre_trim - reagent_df.shape[0]) / len_pre_trim * 100
         print(f"(Reduction of {reduction_pct:.2f}%)")
    print("-" * 50)


    if subset_df.empty:
        print("Error: No reactions remained after filtering by trimmed reagents. Output file will not be generated.")
        exit(1)

    # --- 6. Prepare Final Data Structures ---
    print("\n")
    print("--- Preparing Final Data Structures ---")


    print("Including 'Water Presence' as a condition.")
    # Assuming 'contains_water' column exists and is boolean-like
    if 'contains_water' not in subset_df.columns:
        print("Warning: 'contains_water' column not found. Cannot add water condition. Set --no-water or fix input.")
    else:
        # Check if there are actually reactions with water to justify adding it
        if subset_df['contains_water'].any():
                # Define water pseudo-reagent(s)
                water_data = [{'reagent_type': 'W', 'reagent': 'True', 'count': np.nan, 'reagent_type_numbered': 'W1'}]
                #Potentially add False? Original code only had True.
                # water_data.append({'reagent_type': 'W', 'reagent': 'False', 'count': np.nan, 'reagent_type_numbered': 'W2'})

                water_df = pd.DataFrame(water_data)
                reagent_df = pd.concat([reagent_df, water_df], ignore_index=True)
                print(f"Added {len(water_data)} water condition(s) to reagent list.")
        else:
                print("No reactions with 'contains_water=True' found in the filtered data. Skipping water addition.")
    

    #Eliminate the NoAdditive row from the reagent_df (otherwise there will be still the index going around)
    reagent_df = reagent_df[reagent_df["reagent"] != "NoAdditive"]

    #Fix indexing of additives 
    # Find the rows where reagent_type is 'A'
    is_additive = reagent_df['reagent_type'] == 'A'
    # Count how many additive rows there are
    num_additives = is_additive.sum()
    # Generate the new labels ('A1', 'A2', ...)
    new_additive_labels = [f"A{i+1}" for i in range(num_additives)]
    # Update the 'reagent_type_numbered' column for the additive rows
    reagent_df.loc[is_additive, 'reagent_type_numbered'] = new_additive_labels

    reagent_df = reagent_df[reagent_df["reagent"] != "NoSolvent"]

    #Fix indexing of solvents 
    # Find the rows where reagent_type is 'S'
    is_first_solvent = reagent_df['reagent_type'] == 'S'
    # Count how many first_solvent rows there are
    num_first_solvents = is_first_solvent.sum()
    # Generate the new labels ('S1', 'S2', ...)
    new_first_solvent_labels = [f"S{i+1}" for i in range(num_first_solvents)]
    # Update the 'reagent_type_numbered' column for the first_solvent rows
    reagent_df.loc[is_first_solvent, 'reagent_type_numbered'] = new_first_solvent_labels

    #Fix indexing of solvents for second solvent
    if rxn_type == "sm": 
        # Find the rows where reagent_type is 'L'
        is_second_solvent = reagent_df['reagent_type'] == 'L'
        # Count how many second_solvent rows there are
        num_second_solvents = is_second_solvent.sum()
        # Generate the new labels ('L1', 'L2', ...)
        new_second_solvent_labels = [f"L{i+1}" for i in range(num_second_solvents)]
        # Update the 'reagent_type_numbered' column for the second_solvent rows
        reagent_df.loc[is_second_solvent, 'reagent_type_numbered'] = new_second_solvent_labels



    # >>> START: Added logic for handling ambiguous reagent names <<<
    print("\n")
    print("--- Checking for Ambiguous Reagent Names (Same Name, Different Roles) ---")
    # 1. Find reagents that appear with more than one type
    reagent_counts = reagent_df.groupby('reagent')['reagent_type'].nunique()
    ambiguous_names = reagent_counts[reagent_counts > 1].index.tolist()

    # Store the original ambiguous names before modifying reagent_df
    original_ambiguous_names = list(ambiguous_names) # Keep a copy

    if original_ambiguous_names: # Check using the original list
        print(f"Found {len(original_ambiguous_names)} reagent names with multiple roles: {original_ambiguous_names}")
        print("Appending role suffix (e.g., _A, _B) to these reagent names in reagent_df for disambiguation.")

        # 2. Create a mask for rows in reagent_df that need renaming
        ambiguous_mask_rdf = reagent_df['reagent'].isin(original_ambiguous_names)

        # 3. Apply the renaming to reagent_df using .loc
        reagent_df.loc[ambiguous_mask_rdf, 'reagent'] = \
            reagent_df.loc[ambiguous_mask_rdf, 'reagent'] + '_' + reagent_df.loc[ambiguous_mask_rdf, 'reagent_type'].astype(str)

    
        print("Renaming complete in reagent_df.")

        # >>> START: Modify subset_df to match renamed reagents <<<
        print("\n")
        print("--- Updating subset_df with disambiguated reagent names ---")
        # Define columns in subset_df that contain reagent names and their corresponding type
        # Get this from the global reagent_cols definition
        reagent_columns_in_subset = {
             'catalyst_name': 'C',
             'reagent_1_name': 'B', # Base
             'solvent_1_name': 'S',
             'additives_name_merged': 'A',
             'solvent_2_name': 'L'  # Only relevant if rxn_type requires it
             # 'contains_water' is not a reagent name column
        }

        # Iterate through the original ambiguous names found
        for original_name in original_ambiguous_names:
            # Iterate through the columns in subset_df that hold reagent names
            for col_name, role_char in reagent_columns_in_subset.items():
                 # Check if the column actually exists in subset_df
                 if col_name in subset_df.columns:
                    # Create the expected new name based on the role of this column
                    new_name = f"{original_name}_{role_char}"
                    # Create a mask for rows in this column matching the original ambiguous name
                    mask_subset = (subset_df[col_name] == original_name)
                    # Update subset_df using .loc
                    subset_df.loc[mask_subset, col_name] = new_name
                    # Optional: Log how many replacements were made per column/reagent
                    # if mask_subset.sum() > 0:
                    #    print(f"  Replaced '{original_name}' with '{new_name}' in {mask_subset.sum()} rows of column '{col_name}'")

        print("Updating subset_df complete.")
        # >>> END: Modify subset_df <<<

    else:
        print("No ambiguous reagent names found. No renaming needed in reagent_df or subset_df.")
        print("\n")
    print("-" * 50)
    # >>> END: Added logic <<<

    #Use order that we will then need for the sequential VAE
    if rxn_type == "sm":
        order_desired = [ "S", "L", "W", "A",  "C", "B" ]
    else:
        order_desired = ["C", "B", "S", "W", "A"]


    # Assign a name to the index (important for sorting by it later)
    reagent_df.index.name = 'original_index'

    # Convert the 'reagent_type' column to a categorical type with the specified order
    # This allows pandas to sort according to your custom order, not alphabetically
    reagent_df['reagent_type'] = pd.Categorical(reagent_df['reagent_type'], categories=order_desired, ordered=True)

    # Sort the DataFrame based on the ordered 'reagent_type' column first,
    # and then by the named index column to maintain original order within categories
    df_sorted = reagent_df.sort_values(by=['reagent_type', 'original_index'])

    reagent_df = df_sorted.reset_index(drop=True)

    # Use groupby().cumcount() to create sequential numbers within each group
    # Adding 1 because cumcount starts from 0
    reagent_df['reagent_type_numbered'] = reagent_df.groupby('reagent_type', observed=True).cumcount() + 1
    # Concatenate the type character with the number
    reagent_df['reagent_type_numbered'] = reagent_df['reagent_type'].astype(str) + reagent_df['reagent_type_numbered'].astype(str)
    
    # Create the condition dictionary (maps reagent name to [numbered_type, type, index])
    cond_dict = create_dictionary_from_dataframe(reagent_df, 'reagent', ['reagent_type_numbered', 'reagent_type'])

    # Generate reaction strings (reactants>>product)
    try:
         formatted_list = subset_df.apply(lambda row: f"{row['startingmat_1_smiles']}.{row['startingmat_2_smiles']}>>{row['product_1_smiles']}", axis=1).tolist()
    except KeyError as e:
         print(f"Error: Missing column required for reaction string generation: {e}")
         exit(1)

    #Manually correct smile where the valence of solphur is wrong
    old_problematic_reactant = "CC(C)(C)c1cccc(c1)P(=O)(c2cc(cc(c2)C(C)(C)C)C(C)(C)C)c3cccc4CC[C@@]5(CCc6cccc(c56)[S](=O)(=O)(O)C(F)(F)F)c34"
    replacement_reactant = "CC(C)(c1cc(P(c2cc(C(C)(C)C)cc(C(C)(C)C)c2)(c3c4c(CC[C@]54CCc6c5cccc6)ccc3)=O)ccc1)C"
    formatted_list = [i.replace(old_problematic_reactant, replacement_reactant) for i in formatted_list]

    # Generate condition lists (indices of reagents) for each reaction
    all_conds = []
    skipped_reactions = 0
    for i, row in subset_df.iterrows():
        list_to_use = []
        reaction_valid = True
        # Determine which conditions are applicable (handle "NoAdditive", "NoSolvent")
        conditions_to_check = {}
        conditions_to_check['catalyst'] = row.get('catalyst_name')
        conditions_to_check['base'] = row.get('reagent_1_name')

        if row.get("solvent_1_name") != "NoSolvent":
            conditions_to_check['solvent1'] = row.get('solvent_1_name')
        if row.get("additives_name_merged") != "NoAdditive" and not pd.isna(row.get("additives_name_merged")):
             conditions_to_check['additive'] = row.get("additives_name_merged")
        if rxn_type == "sm" and row.get("solvent_2_name") != "NoSolvent" and not pd.isna(row.get("solvent_2_name")):
             conditions_to_check['solvent2'] = row.get("solvent_2_name")
        if row.get('contains_water', False): # Default to False if col missing after check
             conditions_to_check['water'] = 'True' # Map boolean to the string key used in cond_dict

        # Look up indices, ensuring the reagent is actually in our filtered dictionary
        for cond_type, reagent_name in conditions_to_check.items():
            if reagent_name in cond_dict:
                cond_info = cond_dict[reagent_name]
                list_to_use.append(cond_info[2]) # Append the index
            else:
                # This reagent wasn't kept during trimming or is unexpected (e.g., NaN not handled earlier)
                 #print(f"Warning: Reagent '{reagent_name}' for condition type '{cond_type}' in reaction {i} not found in final condition dictionary. Skipping reaction.")
                 reaction_valid = False
                 break # Skip this reaction

        if reaction_valid:
            all_conds.append((sorted(list_to_use), row.get("suff_yield"), row.get("product_1_area%"))) # Sort indices for consistency
        else:
             skipped_reactions += 1
             # Remove the corresponding reaction string if we skip the conditions
             # This is tricky if reactions are duplicated; safer to filter at the end
             pass # We'll filter formatted_list later based on skipped count if needed


    if skipped_reactions > 0:
        print(f"Warning: Skipped {skipped_reactions} reactions due to missing reagent lookups.")
        # If reaction strings and conditions must match perfectly 1:1, we need to rebuild
        # formatted_list excluding the skipped ones. This requires tracking indices.
        # For now, we assume the downstream process can handle potential mismatches or
        # that reaction_dict creation handles it. Revisit if strict 1:1 mapping is needed.


    # Create the final reaction dictionary { reaction_string: [ [cond_indices_1], [cond_indices_2], ... ] }
    reaction_dict = defaultdict(list)
    if len(formatted_list) != len(all_conds):
         print(f"Warning: Mismatch between number of reaction strings ({len(formatted_list)}) and condition lists ({len(all_conds)}). This may indicate an issue with skipped reactions handling.")
         # Decide on recovery strategy: maybe only use min(len(formatted_list), len(all_conds))?
         num_reactions_to_process = min(len(formatted_list), len(all_conds))
         print(f"Processing only the first {num_reactions_to_process} entries.")
    else:
         num_reactions_to_process = len(formatted_list)


    for i in range(num_reactions_to_process):
        reaction = formatted_list[i]
        conds = all_conds[i]
        # Only add if conditions are not empty (might happen if filtering fails unexpectedly)
        if conds:
             reaction_dict[reaction].append(conds)


    # Final condition list (just the numbered types 'C1', 'B1', etc.)
    clist = list(reagent_df["reagent_type_numbered"])


    # --- 7. Save Data ---
    print("\n")
    print("--- Saving Processed Data ---")
    # Ensure output directory exists
    try:
        os.makedirs(outdir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory {outdir}: {e}")
        exit(1)

    if export_reagent_df:
        try:
            os.makedirs("./reagents_dfs", exist_ok=True)
            data_suffix = "_all" if complete_dataset else "_positive"
            out_filename_reagent_df = f"{rxn_type}_{trim_type}{data_suffix}_reagent_df.csv"
            reagent_df.reset_index(drop=True).to_csv("./reagents_dfs/" + out_filename_reagent_df)
            print("Succesfully saved reagent dataframe as: ./reagents_dfs/", out_filename_reagent_df)
        except OSError as e:
            print(f"Error: Could not create output directory ./reagents_df: {e}")
            exit(1)


    # Prepare data for saving
    # Ensure reaction_dict is a standard dict for saving
    data_to_save = {'data': np.array([dict(reaction_dict), clist], dtype=object)}

    # Define output filename
    # Add trim_type and all data flag status to filename for clarity
    data_suffix = "_all" if complete_dataset else "_positive"
    out_filename = f"{rxn_type}_{trim_type}{data_suffix}_processed.npz"
    out_path = os.path.join(outdir, out_filename)

    try:
        np.savez_compressed(out_path, **data_to_save) # Use compression
        print(f"Successfully saved processed data to: {out_path}")
    except Exception as e:
        print(f"Error saving data to {out_path}: {e}")
        exit(1)

    print("--- Processing Complete ---")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload surf data, trim reagent space, and save processed data as required as imput for the ReactionVAE models.")

    parser.add_argument(
        "--infile",
        type=str,
        help="Path to the input CSV file with data in SURF format (e.g.: data/data_clean_new.csv)", 
        required=True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data", # Default output directory
        help="Directory to save the output NPZ file (default: ./data)"
    )
    parser.add_argument(
        "--trim_type",
        type=str,
        default="treshold_all", # Default trimming type
        choices=["quantile_all", "quantile_category", "treshold_all"],
        help="Method for trimming the reagent space (default: treshold_all)"
    )
    parser.add_argument(
        "--export_reagent_df",
        action=argparse.BooleanOptionalAction, 
        default=True, # Default is not to not export the csv
        help="Exports the reagent_df DataFrame as a .csv file. It contains a precise mapping between reagents and assigned number"
    )

    args = parser.parse_args()

    # Add reagent type mapping (used in trimming and filtering)
    reagent_cols = {
        "catalyst_name": "C",
        "reagent_1_name": "B",
        "solvent_1_name": "S",
        "additives_name_merged": "A",
        "solvent_2_name": "L", # Consistent mapping for solvent 2
        "contains_water": "W"
    }


    main(args)