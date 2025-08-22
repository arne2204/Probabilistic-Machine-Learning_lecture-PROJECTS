import pandas as pd

# read csv
df = pd.read_csv(r"C:\Users\lukaspasold\colors\colors_rgb.csv")

# extract date
df["date"] = pd.to_datetime(df["filename"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0])

# convert date to unix time
df["feature_12"] = df["date"].astype("int64") // 10**9  # seconds since 1970-01-01

df.to_csv(r"C:\Users\lukaspasold\colors\colors_rgb_with_dates.csv", index=False)