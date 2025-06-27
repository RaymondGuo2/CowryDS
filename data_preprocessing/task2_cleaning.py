
# Function to remove all values of February
def exclude_february(df):
    filtered_df = df[df['TO_CHAR'].dt.month != 2]
    return filtered_df

# Function to segment the dataset to return VOLT only, non-VOLT only, and control and treatment groups within each (6 total subgroups)
def data_segmentation(df):
    df_v = df[df['VOLT_FLAG'] == 'yes']
    df_nv = df[df['VOLT_FLAG'] != 'yes']
    df_v_control = df[(df['VOLT_FLAG'] == 'yes') & (df['COLUMN_4'] == 'control')]
    df_nv_control = df[(df['VOLT_FLAG'] != 'yes') & (df['COLUMN_4'] == 'control')]
    df_v_treatment = df[(df['VOLT_FLAG'] == 'yes') & (df['COLUMN_4'] == 'pilot')]
    df_nv_treatment = df[(df['VOLT_FLAG'] != 'yes') & (df['COLUMN_4'] == 'pilot')]
    return df_v, df_nv, df_v_control, df_nv_control, df_v_treatment, df_nv_treatment


