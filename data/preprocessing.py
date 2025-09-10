def clean_data(df):
    df_clean = df[['ru', 'ko']].copy()
    df_clean.columns = ['source_text', 'target_text']
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean['source_text'].str.len() > 5]
    df_clean = df_clean[df_clean['target_text'].str.len() > 5]
    return df_clean
