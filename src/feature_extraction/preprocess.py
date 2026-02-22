import pandas as pd

def merge_features(metadata_path: str, features_csv: str) -> pd.DataFrame:
    #Merge metadata with stylometric features.
    metadata = pd.read_csv(metadata_path)
    features = pd.read_csv(features_csv)
    # Ensure features has doc_id
    if 'doc_id' not in features.columns:
        features = features.rename(columns={features.columns[0]: 'doc_id'})
    # Clean both doc_id columns to ensure a perfect match (removes .txt and hidden spaces)
    metadata['doc_id'] = metadata['doc_id'].astype(str).str.replace('.txt', '', regex=False).str.strip()
    features['doc_id'] = features['doc_id'].astype(str).str.replace('.txt', '', regex=False).str.strip()
    
    merged = pd.merge(metadata, features, on='doc_id', how='inner')
    return merged