import pandas as pd
import json
from pathlib import Path
import warnings
from numpy import int64
#loading and saving data

def get_base_path():

    base_path = Path(__file__).resolve().parent.parent
    return base_path

def data_load(path, sep=";", header=0):

    base_path = get_base_path()
    df = pd.read_csv(base_path/path, sep=sep, header=header, low_memory=False)
    print("Data loaded!")
    return df

def save_df(dataframe, path, sep=";"):

    base_path = get_base_path()
    dataframe.to_csv(base_path/path, sep = ";", index = False)
    print("Data saved!")

# missing and wrong values

def normalize_missing_values(dataframe):
    dataframe = dataframe.copy()
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].replace(["", " ", "\xa0", "\t"], pd.NA)
    return dataframe

def delete_na(dataframe):
    dataframe = dataframe.copy()
    print(f"Before deletion: {dataframe.isna().sum()}")
    dataframe = dataframe.dropna()
    print(f"after: {dataframe.isna().sum()}")
    return dataframe

def filter_income(dataframe, wrong_values):
    dataframe = dataframe.copy()
    print(f"Value count: {len(dataframe)}")
    mask = dataframe["income"].isin(wrong_values)
    dataframe = dataframe[~mask]
    print(f"Value count: {len(dataframe)}")
    print(f"Unique Values: {dataframe['income'].unique()}")
    print(f"Removed {mask.sum()} rows with income in {wrong_values}")
    return dataframe

# renaming and labeling

def select_rename(mapping_dict_path, dataframe):
    dataframe = dataframe.copy()
    base_path = get_base_path()
    with open(base_path/mapping_dict_path, "r") as f:
        mappings = json.load(f)
    if "features" not in mappings:
        warnings.warn("features not found in mappings")
    selected_features = mappings["features"]
    dataframe = dataframe[selected_features].copy()
    dataframe.rename(columns=mappings["feature_names"], inplace=True)
    print(f"selected_features: {dataframe.columns}")

    return dataframe

def apply_label_mappings_string(dataframe, mapping_dict_path):
    base_path = get_base_path()
    dataframe = dataframe.copy()
    with open(base_path/mapping_dict_path, "r") as f:
        mappings = json.load(f)
    contained_columns = set()
    for column in dataframe.columns:

        if column in mappings:
            unique_values = set(dataframe[column].dropna().unique())
            mapping_keys = set(mappings[column].keys())
            not_mapped = unique_values - mapping_keys
            if not_mapped:
                print(f"Column '{column}': not mapped: {not_mapped}")
            contained_columns.add(column)
            dataframe[column] = dataframe[column].map(mappings[column])
        
    print(f"changed columns: {contained_columns}")

    return dataframe

def apply_label_mappings_int(dataframe, mapping_dict_path):
    base_path = get_base_path()
    dataframe = dataframe.copy()
    with open(base_path/mapping_dict_path, "r") as f:
        mappings = json.load(f)

    mappings = {    col: {int64(k): v for k, v in col_map.items()}
    for col, col_map in mappings.items()
    }
    contained_columns = set()
    for column in dataframe.columns:

        if column in mappings:
            unique_values = set(dataframe[column].dropna().unique())
            mapping_keys = set(mappings[column].keys())
            not_mapped = unique_values - mapping_keys
            if not_mapped:
                print(f"Column '{column}': not mapped: {not_mapped}")
            contained_columns.add(column)
            dataframe[column] = dataframe[column].map(mappings[column])
        
    print(f"changed columns: {contained_columns}")

    return dataframe


# save dictionary

def save_dictionary(dictionary, path):
    base_path = get_base_path()
    with open(base_path/path, "w") as f:
        json.dump(dictionary, f, indent=2)
    print("Dictionary Saved!")