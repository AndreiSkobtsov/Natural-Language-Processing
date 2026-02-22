import os
import pandas as pd
import stylo_metrix as sm

class StyloMetrixExtractor:
    def __init__(self, lang: str = 'en'):
        self.lang = lang
    
    def extract(self, input_folder: str, output_csv: str) -> pd.DataFrame:
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Input folder {input_folder} not found.")
        
        print(f"Initializing StyloMetrix for language '{self.lang}'...")
        stylo = sm.StyloMetrix(self.lang)
        
        texts = []
        doc_ids = []
        
        #reads all text files in the directory
        for file in os.listdir(input_folder):
            if file.endswith('.txt'):
                with open(os.path.join(input_folder, file), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                # Save the filename (without .txt) to match the metadata (double security)
                doc_ids.append(file.replace('.txt', ''))
        
        if not texts:
            raise ValueError(f"No .txt files found in {input_folder}")
            
        print(f"Extracting features for {len(texts)} documents. This might take a minute...")
        
        #Extract features using the native Python method
        #Basically we run stylometrix on all the .txt files
        df = stylo.transform(texts)
        
        # Insert doc_id as the very first column
        df.insert(0, 'doc_id', doc_ids)
        
        #saves to CSV
        df.to_csv(output_csv, index=False)
        return df