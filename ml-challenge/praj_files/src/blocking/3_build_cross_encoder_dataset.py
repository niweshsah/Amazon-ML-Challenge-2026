import pandas as pd
import os
import random
from tqdm import tqdm

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "business_entity_resolution", "challenge-dataset", "dataset", "train"))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    
    # We load the ORIGINAL text (not preprocessed) because Cross-Encoders are smart
    # and benefit from natural language capitalization and punctuation.
    print("Loading Original Source 1...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str)
    df_s1['text'] = df_s1['business_name'].fillna('') + " " + df_s1['business_address'].fillna('') + " " + df_s1['country'].fillna('')
    s1_dict = df_s1.set_index('entity_id')['text'].to_dict()
    del df_s1
    
    print("Loading Original Source 2 & 3...")
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str)
    df_s2['text'] = df_s2['business_name'].fillna('') + " " + df_s2['business_address'].fillna('') + " " + df_s2['country'].fillna('')
    s2_dict = df_s2.set_index('entity_id')['text'].to_dict()
    del df_s2
    
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", dtype=str)
    df_s3['text'] = df_s3['business_name'].fillna('') + " " + df_s3['business_address'].fillna('') + " " + df_s3['country'].fillna('')
    s3_dict = df_s3.set_index('entity_id')['text'].to_dict()
    del df_s3
    
    # Combine S2 and S3 into one giant target dictionary
    target_dict = {**s2_dict, **s3_dict}
    del s2_dict, s3_dict
    
    print("Loading Ground Truth...")
    # Assume ground truth has columns: source1_entity_id, candidate_entity_ids (or similar)
    # We will need to check exact column names, but usually it maps s1 -> s2/s3
    gt_df = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)
    
    # Build positive set
    gt_positives = set()
    for _, row in gt_df.iterrows():
        s1_id = row.iloc[0]
        # The truth might be comma separated
        matches = str(row.iloc[1]).split(',')
        for match in matches:
            match = match.strip()
            if match:
                gt_positives.add((s1_id, match))
                
    print(f"Total Ground Truth Positives: {len(gt_positives)}")
    
    print("Loading KNN Candidates to create Hard Negatives...")
    knn_df = pd.read_csv(os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", dtype=str)
    
    training_data = []
    
    print("Building Cross-Encoder Training Dataset...")
    # We will take all true positives, and 3 hard negatives per query
    for _, row in tqdm(knn_df.iterrows(), total=len(knn_df)):
        s1_id = str(row['source1_entity_id'])
        if pd.isna(row['candidate_entity_ids']):
            continue
            
        candidates = str(row['candidate_entity_ids']).split(',')
        
        negatives_added = 0
        for cand_id in candidates:
            cand_id = cand_id.strip()
            if not cand_id: continue
            
            # Check if this pair is a true positive
            is_positive = (s1_id, cand_id) in gt_positives
            
            if is_positive:
                if s1_id in s1_dict and cand_id in target_dict:
                    training_data.append({
                        'text_a': s1_dict[s1_id],
                        'text_b': target_dict[cand_id],
                        'label': 1
                    })
            else:
                # Add up to 3 hard negatives per s1_id
                if negatives_added < 3:
                    if s1_id in s1_dict and cand_id in target_dict:
                        training_data.append({
                            'text_a': s1_dict[s1_id],
                            'text_b': target_dict[cand_id],
                            'label': 0
                        })
                        negatives_added += 1
                        
    out_path = os.path.join(output_dir, "cross_encoder_train.csv")
    pd.DataFrame(training_data).to_csv(out_path, index=False)
    print(f"\nSUCCESS! Saved Cross-Encoder dataset to {out_path} with {len(training_data)} rows.")

if __name__ == "__main__":
    main()
