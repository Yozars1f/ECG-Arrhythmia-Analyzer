import pandas as pd
import numpy as np

def balance_dataset():
    input_file = 'data/mitbih_train.csv'
    output_file = 'data/mitbih_balanced_train.csv'

    print(f"Loading {input_file}...")
    try:
        df = pd.read_csv(input_file, header=None)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # Use integer index for column 187 since header is None
    # Rename for clarity temporarily
    df.rename(columns={187: 'category'}, inplace=True)

    print("Target distribution before balancing:")
    print(df['category'].value_counts().sort_index())

    # Find the count of the smallest class
    min_class_count = df['category'].value_counts().min()
    print(f"\nSmallest class count: {min_class_count}")

    # Undersample each class
    balanced_df = pd.DataFrame()
    
    # Iterate through classes 0 to 4
    for i in range(5):
        class_df = df[df['category'] == i]
        # Sample N rows
        if len(class_df) >= min_class_count:
            resampled_class = class_df.sample(n=min_class_count, random_state=42)
            balanced_df = pd.concat([balanced_df, resampled_class], axis=0)
    
    # Shuffle the final dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Rename column back to original index or keep it? 
    # The requirement says "No header", so we should save without header.
    # The column name 'category' won't be saved because header=False.

    print("\nTarget distribution after balancing:")
    print(balanced_df['category'].value_counts().sort_index())

    # Save to CSV
    balanced_df.to_csv(output_file, index=False, header=False)
    print(f"\nBalanced dataset saved to {output_file}")

if __name__ == "__main__":
    balance_dataset()
