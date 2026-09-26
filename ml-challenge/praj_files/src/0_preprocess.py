try:
    import cudf as pd
    print("✅ Using GPU-accelerated cuDF for blazing fast preprocessing!")
except ImportError:
    import pandas as pd
    print("⚠️ cuDF not found. Falling back to CPU-based Pandas.")
    
import os
import re
import time
from tqdm import tqdm

def clean_text_column(series):
    """
    Applies standard cleaning to a text column in a pandas Series.
    """
    # 1. Lowercase and fill missing values
    s = series.fillna('').str.lower()
    
    # 2. Handle specific symbols before stripping punctuation
    s = s.str.replace(r'&', ' and ', regex=True)
    s = s.str.replace(r'@', ' at ', regex=True)
    s = s.str.replace(r'\bc/o\b', ' care of ', regex=True)
    s = s.str.replace(r'\bs/o\b', ' son of ', regex=True)
    s = s.str.replace(r'\bd/o\b', ' daughter of ', regex=True)
    s = s.str.replace(r'\bw/o\b', ' wife of ', regex=True)
    s = s.str.replace(r'\bopp\b\.?', ' opposite ', regex=True)
    
    # 3. Remove all punctuation except spaces (replace with space)
    # This turns "<< Team Ecole" into "   team ecole"
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    
    # 4. Standardize common abbreviations using word boundaries (\b)
    # We standardize to the shortest common form
    replacements = {
        r'\bprivate\b': 'pvt',
        r'\blimited\b': 'ltd',
        r'\bcorporation\b': 'corp',
        r'\bcompany\b': 'co',
        r'\bincorporated\b': 'inc',
        r'\broad\b': 'rd',
        r'\bstreet\b': 'st',
        r'\bavenue\b': 'ave',
        r'\blane\b': 'ln',
        r'\bfloor\b': 'fl',
        r'\bapartment\b': 'apt',
        r'\bbuilding\b': 'bldg',
        r'\bnr\s': 'near ', # nr. jhanvi -> near jhanvi
    }
    
    for pattern, replacement in replacements.items():
        s = s.str.replace(pattern, replacement, regex=True)
        
    # 4. Strip extra whitespace (replace multiple spaces with a single space)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return s

def process_and_save_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    print(f"\nProcessing {filename}...")
    start_time = time.time()
    
    import csv
    # Read the TSV (quoting=csv.QUOTE_NONE prevents pandas from getting confused by stray quotes)
    df = pd.read_csv(input_path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE)
    
    # Clean Business Name
    if 'business_name' in df.columns:
        df['business_name_clean'] = clean_text_column(df['business_name'])
        
    # Clean Business Address
    if 'business_address' in df.columns:
        df['business_address_clean'] = clean_text_column(df['business_address'])
        
    # Create the combined text for future TF-IDF or Embedding generation
    if 'business_name_clean' in df.columns and 'business_address_clean' in df.columns:
        df['combined_text_clean'] = df['business_name_clean'] + " " + df['business_address_clean']
    
    # Save as Parquet (highly compressed, super fast loading)
    output_filename = filename.replace('.tsv', '.parquet')
    output_path = os.path.join(output_dir, output_filename)
    
    df.to_parquet(output_path, engine='pyarrow', index=False)
    
    print(f"Saved to {output_path} in {time.time() - start_time:.2f} seconds.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input directories
    train_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "business_entity_resolution", "challenge-dataset", "dataset", "train"))
    test_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "business_entity_resolution", "challenge-dataset", "dataset", "test"))
    
    # Output directory
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "datasets", "preprocessed"))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input Train Dir: {train_dir}")
    print(f"Input Test Dir:  {test_dir}")
    print(f"Output Dir:      {output_dir}")
    
    # Files to process
    files_to_process = [
        os.path.join(train_dir, "train_source1.tsv"),
        os.path.join(train_dir, "train_source2.tsv"),
        os.path.join(train_dir, "train_source3.tsv"),
        os.path.join(train_dir, "train_ground_truth.tsv"), # For ground truth, we just convert to parquet for speed
        os.path.join(test_dir, "test_source1.tsv"),
        os.path.join(test_dir, "test_source2.tsv"),
        os.path.join(test_dir, "test_source3.tsv"),
    ]
    
    for file_path in tqdm(files_to_process, desc="Processing Files"):
        if os.path.exists(file_path):
            process_and_save_file(file_path, output_dir)
        else:
            print(f"\n[WARNING] File not found: {file_path}")

    print("\nAll preprocessing complete! You can now load these .parquet files instantly in your future scripts.")

if __name__ == "__main__":
    main()
