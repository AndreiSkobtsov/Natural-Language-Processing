import os
import pandas as pd
from typing import List, Dict

def save_corpus(corpus: List[Dict], base_path: str, metadata_path: str) -> None:
    #Save each text as a separate .txt file and write metadata CSV.
    os.makedirs(base_path, exist_ok=True)
    metadata_rows = []
    for i, entry in enumerate(corpus):
        filename = f"{entry['model']}_{entry['genre']}_{i:03d}.txt"
        filepath = os.path.join(base_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(entry['text'])
        metadata_rows.append({
            'doc_id': filename,
            'model': entry['model'],
            'genre': entry['genre'],
            'prompt': entry['prompt']
        })
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    print(f"Saved {len(corpus)} documents to {base_path}")