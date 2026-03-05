import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'origin_dataset.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

def clean_and_split():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    print(f"Loading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # Normalize: lowercase
    df['word'] = df['word'].str.lower()
    
    # Remove duplicates
    original_len = len(df)
    df.drop_duplicates(subset=['word'], inplace=True)
    print(f"Removed {original_len - len(df)} duplicates (case-insensitive).")
    
    # Filter: strict a-z only?
    # Some words might have accents. 'etymwn' has many languages. English words usually are ascii.
    # Let's keep it simple for now, but maybe remove words with numbers or strict non-alpha?
    df = df[df['word'].str.isalpha()]
    print(f"Filtered to {len(df)} alpha-only words.")
    
    # Stats
    print("Class distribution:")
    print(df['origin'].value_counts())
    
    # Split
    # Stratified split to maintain class ratios
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['origin'])
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42, stratify=test_val_df['origin'])
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    print("Saved splits to dataset directory.")

if __name__ == '__main__':
    clean_and_split()
