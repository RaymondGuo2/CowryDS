# Package for cleaning data in Task1

import pandas as pd

# Load the data to ensure dataframe is well-formatted
def load_data(file_path):
    # Load singular header for ease of processing
    df = pd.read_excel(file_path, "Data", skiprows=1)
    cols = list(df.columns)
    df.rename(columns={cols[0]:"Batch Group", cols[1]:"Gender", cols[2]:"Age", cols[13]:"Preparedness Levels"}, inplace=True)

    # Split the batch groups into their respective countries and industries to enable modelling of country-level and industry-level differences
    split_cols = df["Batch Group"].str.split(' ', n=1, expand=True)
    split_cols.columns = ["Country", "Industry"]
    for i, col_name in enumerate(split_cols.columns):
        df.insert(i+1, col_name, split_cols[col_name]) 
    return df

# Re-label latent and explicit factors for cleanliness with data labels
def relabel_factors(df):
    for col in df.columns:
        if col.startswith("Emotional Likert.") or col.startswith("Emotional Statements.") or col.startswith("Functional Likert.") or col.startswith("Functional Statements."):
            # Extract the factor code and match the provided keys
            split_text = col.split(".", 1)[1]
            factor = split_text.replace(" ","")
            factor = factor.lower()
            df.rename(columns={col: factor}, inplace=True)
    # Handle edge case with wrong spelling
    df.rename(columns={"lcarpromd": "lcarprom"}, inplace=True)    
    return df

# Combine the explicit and implicit factors together
def create_combined_factors(df):
    emotional_explicit_index = df.iloc[:, 17:32]
    emotional_implicit_index = df.iloc[:, 43:58] 
    workplace_explicit_index = df.iloc[:, 32:43]
    workplace_implicit_index = df.iloc[:, 58:69]

    if not (emotional_explicit_index.shape[1] == emotional_implicit_index.shape[1] and 
            workplace_explicit_index.shape[1] == workplace_implicit_index.shape[1]):
        raise ValueError("Explicit and implicit ranges must have the same number of columns.")

    df_new = df.copy()

    for c1, c2 in zip(emotional_explicit_index.columns, emotional_implicit_index.columns):
        new_name = c1[1:]
        df_new[new_name] = (emotional_explicit_index[c1] + emotional_implicit_index[c2]) / 2

    # Combine workplace factors
    for c1, c2 in zip(workplace_explicit_index.columns, workplace_implicit_index.columns):
        new_name = c1[1:]
        df_new[new_name] = (workplace_explicit_index[c1] + workplace_implicit_index[c2]) / 2

    return df_new



