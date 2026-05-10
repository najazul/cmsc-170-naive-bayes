import pandas as pd
from datasets import load_dataset

print("Loading existing cleaned dataset...")
df_existing = pd.read_csv("sleep_stage_sample_cleaned.csv")
existing_count = len(df_existing)
rows_needed = 500000 - existing_count

if rows_needed <= 0:
    print(f"Dataset already has {existing_count} rows. No rows needed.")
    exit(0)

print(f"Need {rows_needed} additional rows.")

print("Loading additional data from Hugging Face...")
ds_extra = load_dataset("abmallick/heart-breath-sleep-stage-dataset", split="train[500000:501000]")
df_extra = ds_extra.to_pandas()

# Clean extra data to match format
df_extra_clean = df_extra[df_extra['sleep_stage'] != 9].copy()
cols_to_drop = [c for c in ['time_seconds', 'heart_rate', 'respiratory_rate'] if c in df_extra_clean.columns]
df_extra_clean = df_extra_clean.drop(columns=cols_to_drop)

stage_mapping = {0: 'Wake', 1: 'N1_Light', 2: 'N2_Light', 3: 'N3_Deep', 5: 'REM'}
df_extra_clean['stage_name'] = df_extra_clean['sleep_stage'].map(stage_mapping)

# Ensure columns are in the same order
df_extra_clean = df_extra_clean[df_existing.columns]

# Avoid duplicates
merged = df_extra_clean.merge(df_existing, how='left', indicator=True)
df_new = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

# Drop duplicates within new rows
df_new = df_new.drop_duplicates()

if len(df_new) >= rows_needed:
    df_append = df_new.head(rows_needed)
    print(f"Appending {rows_needed} rows...")
    df_append.to_csv("sleep_stage_sample_cleaned.csv", mode='a', header=False, index=False)
    print("Done!")
    
    # Verify
    df_final = pd.read_csv("sleep_stage_sample_cleaned.csv")
    print(f"Final row count: {len(df_final)}")
else:
    print("Not enough unique rows found in the chunk.")
