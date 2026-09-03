import pandas as pd

print("Reading Online Retail II dataset...")

file_path = "data/retail_raw/online_retail_II.xlsx"

# Read both sheets
df1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
df2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")

# Combine both years
df = pd.concat([df1, df2], ignore_index=True)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Remove rows without product descriptions
df = df.dropna(subset=["Description"])

# Remove cancelled invoices
df = df[~df["Invoice"].astype(str).str.startswith("C")]

# Keep valid transactions
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# Select exactly 1,000 rows
df = df.sample(n=1000, random_state=42)

# Save our project dataset
df.to_csv("data/online_retail.csv", index=False)

print("Final shape:", df.shape)
print("Dataset saved successfully!")

print("\nFirst 5 rows:")
print(df.head())