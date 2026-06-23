import pandas as pd

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
df["City_id"] = df["City"] + " | " + df["Country"]
df = df.drop(columns=["City", "Country"])

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

# Save
df.to_csv("./data/preprocessed_data.csv")