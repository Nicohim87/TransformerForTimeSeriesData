import pandas as pd
import unicodedata
import joblib
import json
import re
import os

with open("parameters.json", 'r') as f:
    params = json.load(f)

# ---- Data Preparation ----

# Read air quality dataset
df = pd.read_csv("./data/global_air_quality_2014_2025.csv")
df = df.drop(columns=["State", "AQI_Bucket"])

# Read world city dataset
city = pd.read_csv("./data/worldcities.csv")
city["country"] = city["country"].replace("Korea, North", "North Korea")
city["country"] = city["country"].replace("Korea, South", "South Korea")

# Read world city supplementary dataset
supplement = pd.read_csv("./data/worldcities_supplement.csv")

# City dataset preparation
city = city.rename(columns={"lng": "lon"})
city = city[["city", "city_ascii", "country", "lat", "lon"]]

# Concat city and its supplementary dataset
city = pd.concat([city, supplement], ignore_index=True)
city["id"] = city["city_ascii"] + " | " + city["country"]

# City dataset cleaning
city = city[["id", "lat", "lon"]].drop_duplicates(subset="id")
city = city.rename(columns={"lat": "Lat", "lon": "Lon"}).set_index("id")

# Air quality dataset cleaning
def normalize_name(text):
    if not isinstance(text, str):
        return text

    # Remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

    # Remove anything inside parentheses (including the parentheses)
    text = re.sub(r'\s*\([^)]*\)', '', text)

    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["City"] = df["City"].apply(normalize_name)

df["City_id"] = df["City"] + " | " + df["Country"]

# Joining air quality and city dataset
df = df.join(city, how="left", on="City_id", validate="m:1")

# Separate date into year and month
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

df = df.drop(columns="Date")

# Drop missing values
df = df.dropna()

# Drop duplicate columns
df = df.drop_duplicates(subset=["City_id", "Year", "Month"])

# Sequence id (For data splitting)
df["Seq_id"] = df["Year"]*12 + df["Month"]

# Reorder Columns
first_cols = ['City_id', 'Seq_id', 'Year', 'Month', 'Lat', 'Lon', 'Population_Density_per_SqKm']

df = df[first_cols + [c for c in df.columns if c not in first_cols]]

# ---- Save Processed Raw DS ----
df.to_csv("./data/preprocessed_data.csv", index=False)



# ---- Train Test Splitting ----
import numpy as np

SEQ_LEN = params["seq_len"]

entities = df["City_id"].unique()

# Filter test only entities by predetermined country names
test_countries = params["test_countries"]
selector = df["Country"].isin(test_countries)

test_df  = df[df["Country"].isin(test_countries)]
df = df[~df["Country"].isin(test_countries)]

test_df = test_df.drop(columns=["City", "Country"])
df = df.drop(columns=["City", "Country"])

# Get sequence starts
entity = df["City_id"].unique()
seq = df["Seq_id"].unique()
seq_start = np.arange(seq.min(), seq.max() - SEQ_LEN + 1, 1)

# Train, val, test split
train_seq = seq_start[:- params["val_time_len"]]
val_seq = seq_start[- params["val_time_len"]:]



# ---- Normalization ----
from sklearn.preprocessing import StandardScaler

# Initiate scaler and scaler df
scaler = StandardScaler()
scaler_seq = set(seq[:- params["val_time_len"]])
scaler_df = df[df["Seq_id"].isin(scaler_seq)].drop(columns=["City_id", "Seq_id"])

# Fit scaler
scaler.fit(scaler_df)

# Scale the whole ds
df[df.columns[2:]] = scaler.transform(df[df.columns[2:]])
test_df[test_df.columns[2:]] = scaler.transform(test_df[test_df.columns[2:]])

# ---- Entity - Seq Pair ----
def generate_pair(seq, entity=entity):
    pairs = []

    for s in seq:
        for e in entity:
            pairs.append((e, s))

    return pairs

ds_train = generate_pair(train_seq)
ds_val = generate_pair(val_seq)

ds_test = generate_pair(seq_start, test_df["City_id"].unique())
print(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

# ---- Save scaled ds, scaler, and pairs ----
df.to_csv("./data/scaled_data.csv", index=False)
test_df.to_csv("./data/scaled_test_data.csv", index=False)

with open("./data/entity_seq_pair.json", "w") as f:
    json.dump({
        "metadata": {
            "seq_starts": seq_start.tolist(),
            "seq_len": SEQ_LEN,
            "index": df.columns[:2].tolist(),
            "features": df.columns[6:].tolist(),
            "input_only_features": df.columns[2:6].tolist(),
            "test_countries": test_countries,
            "test_city_id": test_df["City_id"].unique().tolist()
        },
        "train": ds_train,
        "val": ds_val,
        "test": ds_test
    }, f, indent=4, default=int)

os.makedirs("./model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
