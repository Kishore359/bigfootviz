import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import os

# Download necessary NLTK data if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Load the dataset
# df = pd.read_csv('bigfoot_reports.csv') # Old way
import zipfile
archive_path = 'reports.csv.zip'
csv_filename_in_zip = 'reports.csv' # Assuming this is the name inside the zip

# Open the zip file and then the CSV file within it
with zipfile.ZipFile(archive_path, 'r') as zf:
    with zf.open(csv_filename_in_zip) as f:
        df = pd.read_csv(f)
print(f"Initial df shape: {df.shape}")

# Compute 'year'
df['year'] = pd.to_numeric(df['Year'], errors='coerce')
print(f"Shape after year computation: {df.shape}, NaNs in year: {df['year'].isna().sum()}")

# Compute 'delay'
df['Submitted Date'] = pd.to_datetime(df['Submitted Date'], errors='coerce')
# Explicitly handle cases where year might be NaN before constructing date string
df.dropna(subset=['year'], inplace=True) # Remove rows where year could not be parsed
df['year'] = df['year'].astype(int) # Ensure year is integer for date string construction
df['event_july1'] = pd.to_datetime(df['year'].astype(str) + '-07-01', format='%Y-%m-%d', errors='coerce')
df['delay'] = (df['Submitted Date'] - df['event_july1']).dt.days
print(f"Shape after delay computation: {df.shape}, NaNs in Submitted Date: {df['Submitted Date'].isna().sum()}, NaNs in event_july1: {df['event_july1'].isna().sum()}, NaNs in delay: {df['delay'].isna().sum()}")

# Compute 'narrative'
df['narrative'] = df['Observed'].astype(str).apply(len)

# Compute 'sentiment'
analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['Observed'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
print(f"Shape after sentiment computation: {df.shape}, NaNs in sentiment: {df['sentiment'].isna().sum()}")

# Select numeric columns for outlier removal
numeric_cols = ['year', 'delay', 'narrative', 'sentiment']
df = df.dropna(subset=numeric_cols) # Drop rows where any of these are NaN
print(f"Shape after dropping NaNs from numeric_cols ({', '.join(numeric_cols)}): {df.shape}")

# Outlier rule
for col in numeric_cols:
    # Add check for empty dataframe before quantile
    if df.empty:
        print(f"DataFrame is empty before processing outlier rule for column {col}. Skipping.")
        break
    lower_bound = df[col].quantile(0.005)
    upper_bound = df[col].quantile(0.995)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Shape after outlier rule for {col}: {df.shape}")

# Randomly sample max 5,000 rows
if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)
print(f"Shape after sampling: {df.shape}")

# Prepare data for JSON output
# Ensure columns exist before selecting, in case df is empty
if not df.empty:
    output_df = df[['year', 'delay', 'narrative', 'sentiment']]
else:
    output_df = pd.DataFrame(columns=['year', 'delay', 'narrative', 'sentiment']) # empty df with correct columns
output_data = output_df.to_dict(orient='records')

# Dump to JSON
with open('scatter_data.json', 'w') as f:
    json.dump(output_data, f)

# Print final byte size
file_size = os.path.getsize('scatter_data.json')
print(f"Final JSON file size: {file_size} bytes")

if file_size > 2 * 1024 * 1024:
    print("Warning: JSON file size exceeds 2 MB limit.")
