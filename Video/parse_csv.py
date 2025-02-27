import pandas as pd

csv_file = "annotations.csv"

df = pd.read_csv(csv_file)
print(df.head())  # prints the first few rows
