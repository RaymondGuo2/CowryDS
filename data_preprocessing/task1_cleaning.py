import pandas as pd

# Load the data to ensure dataframe is well-formatted
def load_data(file_path):
    # Load singular header for ease of processing
    df = pd.read_excel(file_path, "Data", skiprows=1)
    cols = list(df.columns)
    df.rename(columns={cols[0]:"Batch Group", cols[1]:"Gender", cols[2]:"Age"}, inplace=True)

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



