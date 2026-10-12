import pandas as pd
from utils.preprocessing import preprocess_pipeline

# File paths
INPUT_FILE = r"C:\Users\Kamalika\Documents\DisasterAssist_NLP\data\emergency_messages.csv"
OUTPUT_FILE = r"C:\Users\Kamalika\Documents\DisasterAssist_NLP\data\preprocessed_messages.csv"

print("Starting ETL process...")

# =========================
# 1. EXTRACT
# =========================
print("\n[EXTRACT]")
df = pd.read_csv(INPUT_FILE)

print("Original dataset loaded successfully.")
print("Original shape:", df.shape)

# =========================
# 2. TRANSFORM
# =========================
print("\n[TRANSFORM]")
print("Applying NLP preprocessing to all messages...")

# Make sure message_en exists
if "message_en" not in df.columns:
    raise ValueError("Column 'message_en' was not found in the dataset.")

df["processed_message"] = (
    df["message_en"]
    .fillna("")
    .astype(str)
    .apply(preprocess_pipeline)
)

print("Text preprocessing completed.")
print("Steps: Cleaning → Tokenization → Stop-word Removal → Lemmatization")

# =========================
# HANDLE MISSING VALUES
# =========================

if "person" in df.columns:
    df["person"] = df["person"].fillna("Unknown")

if "location" in df.columns:
    df["location"] = df["location"].fillna("Unknown")

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# =========================
# 3. LOAD
# =========================
print("\n[LOAD]")

df.to_csv(OUTPUT_FILE, index=False)

print("Preprocessed dataset saved successfully.")
print("Output file:")
print(OUTPUT_FILE)

print("\nFinal dataset shape:", df.shape)

# Show sample results
print("\nSample Before and After:")
print(df[["message_en", "processed_message"]].head(5).to_string(index=False))

print("\nETL PROCESS COMPLETED SUCCESSFULLY!")