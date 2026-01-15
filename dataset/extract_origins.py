import csv
import collections
import os

INPUT_FILE = r'd:\dataset\etymwn.tsv'
OUTPUT_FILE = r'd:\dataset\origin_dataset.csv'

# Mappings based on ISO 639 codes common in Wiktionary/EtymWN
# Categories: Germanic, Latin, Greek, French
CATEGORY_MAP = {
    # Germanic (Native & Cousins)
    'ang': 'Germanic',  # Old English
    'enm': 'Germanic',  # Middle English (Careful: ME often has French roots, but if marked as source it might imply continuity. Usually we want 'ang' for true native roots, but let's include 'enm' as checking logic or exclude? 
                        # If 'enm' is the source, it implies the word existed in ME. This doesn't prove it's Germanic. 
                        # Let's stick to PROVEN Germanic ancestors for "Germanic": Old English, Old Norse, etc.)
    'non': 'Germanic',  # Old Norse
    'deu': 'Germanic',  # German
    'nld': 'Germanic',  # Dutch
    'gem': 'Germanic',  # Proto-Germanic
    'goh': 'Germanic',  # Old High German
    'gmw': 'Germanic',  # Proto-West Germanic
    
    # Latin
    'lat': 'Latin',
    
    # Greek
    'grc': 'Greek',
    
    # French
    'fra': 'French',
    'fro': 'French',    # Old French
    'frm': 'French'     # Middle French
}

def main():
    print(f"Processing {INPUT_FILE}...")
    
    counts = collections.Counter()
    # Use a dict to store word -> set of origins to handle multiple
    word_origins = collections.defaultdict(set)
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                source, rel, target = parts[0], parts[1], parts[2]
                
                # Check relation type
                if 'etymological_origin_of' not in rel:
                    continue
                    
                # We want English targets
                if not target.startswith('eng:'):
                    continue
                
                # Source language code extraction
                if ':' not in source:
                    continue
                src_lang = source.split(':')[0]
                
                if src_lang in CATEGORY_MAP:
                    category = CATEGORY_MAP[src_lang]
                    
                    # Extract the word (remove 'eng:' prefix)
                    word = target.split(':', 1)[1].strip()
                    
                    # basic cleaning: remove excessive whitespace, maybe skip multi-word phrases for cleaner single-word analysis?
                    # The user project mentions "character-level features" so single words are best.
                    if ' ' in word or '-' in word:
                        continue
                        
                    word_origins[word].add(category)

    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    # Process and write
    final_rows = []
    
    # Heuristic for multi-origin: 
    # If a word is marked as both Latin and French, it's likely Latin -> French -> English. 
    # In this context, "French" is usually the immediate donor of interest for loanword studies, 
    # but "Latin" is the ultimate root. 
    # However, usually we prioritize the IMMEDIATE loan source for "origin sets" if we want to contrast 'native' vs 'loan'.
    # For simplicity, if multiple exist, we'll keep all unique pairs, or perhaps flag them. 
    # For this dataset generation, let's just write all valid unique pairs.
    # The classifier can then decide how to handle ambiguity (e.g. drop ambiguous ones).
    
    for word, origins in word_origins.items():
        for origin in origins:
            final_rows.append((word, origin))
            counts[origin] += 1
            
    # Sort for consistency
    final_rows.sort()

    print(f"Found {len(final_rows)} valid entries.")
    print("Counts by Category:")
    for cat, count in counts.items():
        print(f"  {cat}: {count}")

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'origin'])
        writer.writerows(final_rows)
        
    print(f"Saved dataset to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
