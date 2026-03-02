import pandas as pd

# Check studies table
studies = pd.read_parquet('data/processed/studies.parquet')
print("=== STUDIES ===")
print("enrollment_type unique values:", studies['enrollment_type'].unique())
print("enrollment sample:", studies[['nct_id','enrollment','enrollment_type']].head(10))
print("enrollment non-null:", studies['enrollment'].notna().sum())

# Check calculated_values table  
cv = pd.read_parquet('data/processed/calculated_values.parquet')
print("\n=== CALCULATED_VALUES columns ===")
print(cv.columns.tolist())
print(cv.head(3))