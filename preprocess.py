# Bigfoot Data Preprocessing Script
# This script will read the bigfoot_reports.csv, process it,
# and output points.json and agg.json for visualization.

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
import time # For potential API rate limiting

# spaCy and NLTK imports will be added as those functionalities are implemented
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk # For stopwords

# --- Configuration ---
INPUT_CSV = "bigfoot_reports.csv"
ELEVATION_CACHE_FILE = "elevation_cache.json"
OUTPUT_POINTS_JSON = "points.json"
OUTPUT_AGG_JSON = "agg.json"
RANDOM_SEED = 42
MAX_ROWS_POINTS_JSON = 5000

# --- Helper Functions ---
def load_elevation_cache():
    if os.path.exists(ELEVATION_CACHE_FILE):
        with open(ELEVATION_CACHE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_elevation_cache(cache):
    with open(ELEVATION_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# --- Main Preprocessing Logic ---
def main():
    print("Preprocessing script started...")

    # Load data
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print("Initial columns:", df.columns.tolist())
    print("Initial dtypes:\n", df.dtypes)
    print("Missing values per column:\n", df.isnull().sum())

    df_processed = df.copy()

    # --- Feature Engineering Functions ---

    MONTH_TO_NUM = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
        'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }

    def get_month(row):
        # Try parsing from 'Month' column
        if pd.notna(row['Month']):
            month_str = str(row['Month']).lower().strip()
            if month_str in MONTH_TO_NUM:
                return MONTH_TO_NUM[month_str]

        # Fallback to 'Submitted Date'
        if pd.notna(row['Submitted Date']):
            try:
                # Attempt to parse various common date formats
                # pd.to_datetime is quite flexible
                dt = pd.to_datetime(row['Submitted Date'], errors='coerce')
                if pd.notna(dt):
                    return dt.month
            except Exception: # Broad exception for various parsing issues
                pass
        return np.nan

    # Month to Season mapping (Northern Hemisphere)
    # Winter: Dec, Jan, Feb (12, 1, 2)
    # Spring: Mar, Apr, May (3, 4, 5)
    # Summer: Jun, Jul, Aug (6, 7, 8)
    # Fall: Sep, Oct, Nov (9, 10, 11)
    def month_to_season(month_num):
        if month_num in [12, 1, 2]:
            return 'Winter'
        elif month_num in [3, 4, 5]:
            return 'Spring'
        elif month_num in [6, 7, 8]:
            return 'Summer'
        elif month_num in [9, 10, 11]:
            return 'Fall'
        return np.nan

    def get_season(row, derived_month_col):
        # Use existing 'Season' column if available and valid
        if pd.notna(row['Season']):
            season_str = str(row['Season']).strip().capitalize()
            if season_str in ['Winter', 'Spring', 'Summer', 'Fall', 'Autumn']: # Autumn is alias for Fall
                return 'Fall' if season_str == 'Autumn' else season_str

        # Fallback to derived_month_col
        if pd.notna(row[derived_month_col]):
            return month_to_season(int(row[derived_month_col]))
        return np.nan

    # --- Apply Feature Engineering ---
    print("\nStarting feature engineering...")

    # Month and Season
    df_processed['derived_month'] = df_processed.apply(get_month, axis=1)
    df_processed['derived_season'] = df_processed.apply(lambda row: get_season(row, 'derived_month'), axis=1)

    # Year
    def get_year(report_year):
        if pd.notna(report_year):
            try:
                # Extract first 4 digits if it's a string like "2023 (summer)" or just "2023"
                year_str = str(report_year)
                match = re.search(r'\b(\d{4})\b', year_str)
                if match:
                    return int(match.group(1))
            except ValueError:
                pass # Could not convert to int
        return np.nan
    df_processed['derived_year'] = df_processed['year'].apply(get_year) # 'year' is a column in dummy CSV

    # Submitted Date (parsed)
    def parse_submitted_date(date_str):
        if pd.notna(date_str):
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except Exception:
                return pd.NaT
        return pd.NaT
    df_processed['derived_submitted_date'] = df_processed['Submitted Date'].apply(parse_submitted_date)

    # Day/Night from Time And Conditions
    def get_day_night(time_cond_str):
        if pd.isna(time_cond_str):
            return np.nan

        text = str(time_cond_str).lower()

        # Check for explicit "night", "dusk", "dawn", "evening", "morning"
        if any(term in text for term in ['night', 'dusk', 'dark']):
            return 'Night'
        if any(term in text for term in ['dawn', 'sunrise', 'daybreak', 'morning']): # Morning can be ambiguous but often < 18:00
            return 'Day'
        if 'evening' in text: # Evening is typically >= 18:00
             return 'Night'
        if 'afternoon' in text or 'mid-day' in text or 'daytime' in text or 'noon' in text:
            return 'Day'

        # Regex for HH:MM or H:MM format (24-hour or am/pm)
        # This regex is a bit simplified, real-world time extraction can be complex
        match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            ampm = match.group(3)

            if ampm:
                if ampm == 'pm' and hour != 12:
                    hour += 12
                elif ampm == 'am' and hour == 12: # Midnight case
                    hour = 0

            # Assuming hour is now in 24-hour format (0-23)
            if 0 <= hour < 24: # Basic validation
                if hour < 6 or hour >= 18:
                    return 'Night'
                else:
                    return 'Day'
        return np.nan # Default if no specific time info found

    df_processed['derived_day_night'] = df_processed['Time And Conditions'].apply(get_day_night)

    # Delay
    def get_delay(row):
        if pd.notna(row['derived_submitted_date']) and pd.notna(row['derived_year']):
            try:
                # Create a datetime for July 1st of the observation year
                obs_year_july1st = datetime(int(row['derived_year']), 7, 1)
                # Calculate delta
                delta = row['derived_submitted_date'] - obs_year_july1st
                return delta.days
            except ValueError: # Handles cases like year 0 or invalid date components
                return np.nan
        return np.nan
    df_processed['derived_delay_days'] = df_processed.apply(get_delay, axis=1)

    # Witness Group
    df_processed['derived_witness_grp'] = df_processed['Other Witnesses'].notna() & (df_processed['Other Witnesses'].str.strip() != '')


    # Class (using 'classification' column from dummy CSV)
    # Assuming 'Class A', 'Class B', 'Class C' are the main values.
    # For simplicity, direct copy if valid, else NaN. Could be expanded.
    valid_classes = ['Class A', 'Class B', 'Class C']
    df_processed['derived_class'] = df_processed['classification'].apply(
        lambda x: x if pd.notna(x) and x in valid_classes else np.nan
    )

    # Environment
    # Simple copy for now, can be cleaned/standardized later if needed
    df_processed['derived_environment'] = df_processed['Environment']

    # --- NLP Feature Engineering ---
    # Load NLP models (once)
    nlp_spacy = None
    try:
        nlp_spacy = spacy.load('en_core_web_sm')
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Please run setup_env.sh or download it.")
        # Potentially exit or handle gracefully if these features are critical

    sid = SentimentIntensityAnalyzer()

    # Download stopwords if not already present (first time NLTK usage)
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')

    def get_lemmatized_verbs(text):
        if pd.isna(text) or not nlp_spacy:
            return []
        doc = nlp_spacy(str(text))
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        return list(set(verbs)) # Unique verbs

    def get_cleaned_words(text):
        if pd.isna(text):
            return []
        # Remove punctuation (simple regex) and lowercase
        text_no_punct = re.sub(r'[^\w\s]', '', str(text).lower())
        words = text_no_punct.split()
        # Remove stopwords
        return [word for word in words if word not in stopwords and word.isalpha()]


    def get_vader_sentiment(text):
        if pd.isna(text):
            return np.nan
        return sid.polarity_scores(str(text))['compound']

    if nlp_spacy: # Only apply if spaCy model loaded
        df_processed['derived_verbs'] = df_processed['observed'].apply(get_lemmatized_verbs)
    else:
        df_processed['derived_verbs'] = [[] for _ in range(len(df_processed))]

    df_processed['derived_words'] = df_processed['observed'].apply(get_cleaned_words)
    df_processed['derived_sent'] = df_processed['observed'].apply(get_vader_sentiment)

    # Narrative Length
    df_processed['derived_narr_len'] = df_processed['observed'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

    # Road Type
    def get_road_type(road_str):
        if pd.isna(road_str):
            return np.nan
        road_str = str(road_str).lower()
        if re.search(r'\bi-\d+\b|interstate', road_str): # I-5, interstate 90
            return 'Interstate'
        elif re.search(r'\bhwy\b|highway|us-\d+|route \d+', road_str): # hwy 26, highway 101, US-101, Route 6
             # Avoid matching "State Route" as just "Route" if "State" is also there
            if 'state route' in road_str or 'sr-' in road_str:
                 return 'State' # Handled by next condition more specifically
            return 'Hwy'
        if re.search(r'\bstate (route|rd|hwy)\b|sr-\d+|rte-\d+', road_str): # State Route 6, SR-14
            return 'State'
        if re.search(r'\bcounty (rd|road|hwy)\b|co rd|cr-\d+', road_str): # County Road 388, CR-23
            return 'County'
        if pd.notna(road_str): # If it's not NaN and not matched above, assume local
            return 'Local'
        return np.nan # Should be caught by pd.isna earlier, but as a fallback

    df_processed['derived_road_type'] = df_processed['Nearest Road'].apply(get_road_type)

    # Elevation
    # Ensure imports are available in this scope if not globally resolved as expected
    import requests
    import json # for json.JSONDecodeError
    import time # for time.sleep

    elevation_cache = load_elevation_cache() # Load cache at start of main() would be better

    def parse_coordinates(details_str):
        if pd.isna(details_str):
            return None
        # Simple regex for "lat: XX.XXX, lon: YY.YYY" or "XX.XXX, YY.YYY"
        # More robust parsing might be needed for real data
        # Order: Latitude then Longitude
        patterns = [
            r'lat(?:itude)?:\s*(-?\d+\.?\d*)[,\s]+lon(?:gitude)?:\s*(-?\d+\.?\d*)',
            r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
        ]
        for pattern in patterns:
            match = re.search(pattern, str(details_str), re.IGNORECASE)
            if match:
                try:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    # Basic validation for lat/lon ranges
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        return (lat, lon)
                except ValueError:
                    continue # Could not convert to float
        return None

    API_REQUEST_DELAY = 1 # seconds, to be polite

    def get_elevation(coords):
        if coords is None:
            return np.nan

        coord_key = f"{coords[0]:.5f},{coords[1]:.5f}" # Cache key with fixed precision
        if coord_key in elevation_cache:
            return elevation_cache[coord_key]

        print(f"Querying Open-Elevation API for: {coords}")
        try:
            # Using the public API endpoint
            response = requests.get(f"https://api.open-elevation.com/api/v1/lookup?locations={coords[0]},{coords[1]}")
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
            data = response.json()
            if data['results'] and 'elevation' in data['results'][0]:
                elev = data['results'][0]['elevation']
                elevation_cache[coord_key] = elev
                time.sleep(API_REQUEST_DELAY) # Wait after a successful request
                return elev
        except requests.exceptions.RequestException as e:
            print(f"API request failed for {coords}: {e}")
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response for {coords}")
        except Exception as e:
            print(f"An unexpected error occurred during elevation lookup for {coords}: {e}")

        time.sleep(API_REQUEST_DELAY) # Also wait after a failed request to avoid hammering
        return np.nan

    # First extract all coordinates
    df_processed['coords'] = df_processed['location_details'].apply(parse_coordinates)

    # Then get elevations, using the cache
    # Note: This will make API calls if coords are not in cache.
    # For a large dataset, this step can be slow.
    df_processed['derived_elev'] = df_processed['coords'].apply(get_elevation)

    save_elevation_cache(elevation_cache) # Save cache after processing all rows

    # Pronoun detection
    def get_dominant_pronoun(row):
        text_observed = str(row['observed']) if pd.notna(row['observed']) else ""
        text_noticed = str(row['Also Noticed']) if pd.notna(row['Also Noticed']) else ""
        full_text = (text_observed + " " + text_noticed).lower()

        if not full_text.strip():
            return np.nan

        # More specific pronoun forms can be added (him, her, his, hers etc.)
        # but sticking to "he", "she", "it" as per spec for dominance.
        # Using regex to count whole words to avoid matching "them" with "he".
        counts = {
            "he": len(re.findall(r'\bhe\b', full_text)),
            "she": len(re.findall(r'\bshe\b', full_text)),
            "it": len(re.findall(r'\bit\b', full_text)),
        }

        # Find the dominant pronoun
        max_count = 0
        dominant_pronoun = np.nan
        # Ensure deterministic choice in case of ties, e.g., by order: he, she, it
        for pronoun in ["he", "she", "it"]:
            if counts[pronoun] > max_count:
                max_count = counts[pronoun]
                dominant_pronoun = pronoun
            # If counts[pronoun] == max_count and max_count > 0, it's a tie.
            # Current logic keeps the first one encountered in ["he", "she", "it"] order.
            # If no pronouns are found, max_count remains 0, dominant_pronoun remains np.nan.

        return dominant_pronoun if max_count > 0 else np.nan

    df_processed['derived_pronoun'] = df_processed.apply(get_dominant_pronoun, axis=1)

    # Weather Term
    WEATHER_TERMS = ['rain', 'snow', 'fog', 'clear', 'cloud', 'wind', 'sunny']
    def get_weather_term(time_cond_str):
        if pd.isna(time_cond_str):
            return np.nan
        text = str(time_cond_str).lower()
        for term in WEATHER_TERMS:
            if term in text:
                return term
        return np.nan

    df_processed['derived_weather_term'] = df_processed['Time And Conditions'].apply(get_weather_term)

    # --- FIPS and Population Data ---
    FIPS_URL = "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv"
    POP_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020.csv"

    try:
        print(f"\nLoading FIPS data from {FIPS_URL}...")
        df_fips = pd.read_csv(FIPS_URL, dtype={'fips': str}) # Keep FIPS as string
        # Filter out state summaries, keep only county-level FIPS (typically 5 digits)
        df_fips = df_fips[df_fips['fips'].str.len() == 5]
        df_fips = df_fips.rename(columns={'fips': 'county_fips', 'name': 'fips_county_name', 'state': 'fips_state_abbr'})
        # Clean county name (remove " County", " Parish", etc.)
        df_fips['fips_county_name'] = df_fips['fips_county_name'].str.replace(r'\s+(County|Parish|Borough|Municipality|Census Area|City and Borough)$', '', regex=True)
        print(f"Loaded {len(df_fips)} county FIPS codes.")

        print(f"\nLoading Population data from {POP_URL}...")
        # The population CSV is Latin-1 encoded
        df_pop = pd.read_csv(POP_URL, encoding='latin1', dtype={'STATE': str, 'COUNTY': str})
        df_pop = df_pop[df_pop['SUMLEV'] == 50] # Filter for county-level data
        df_pop['county_fips'] = df_pop['STATE'] + df_pop['COUNTY']
        df_pop = df_pop[['county_fips', 'POPESTIMATE2020']]
        df_pop = df_pop.rename(columns={'POPESTIMATE2020': 'pop_2020'})
        print(f"Loaded {len(df_pop)} county population estimates for 2020.")

        # Prepare main df for merge

        STATE_NAME_TO_ABBR = {
            'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
            'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
            'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
            'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
            'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
            'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH',
            'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC',
            'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA',
            'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN',
            'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA',
            'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
            'DISTRICT OF COLUMBIA': 'DC', 'AMERICAN SAMOA': 'AS', 'GUAM': 'GU',
            'NORTHERN MARIANA ISLANDS': 'MP', 'PUERTO RICO': 'PR', 'VIRGIN ISLANDS': 'VI'
        }

        # Clean county names in df_processed: lowercase, remove " county"
        df_processed['clean_county_for_merge'] = df_processed['county'].astype(str).str.lower().str.replace(r'\s+county$', '', regex=True).str.strip()
        df_fips['clean_county_for_merge'] = df_fips['fips_county_name'].str.lower().str.strip()

        # Standardize state abbreviations
        df_processed['state_abbr_for_merge'] = df_processed['state'].astype(str).str.upper().map(STATE_NAME_TO_ABBR).fillna(df_processed['state'].astype(str).str.upper())
        df_fips['state_abbr_for_merge'] = df_fips['fips_state_abbr'].str.upper().str.strip()

        # Merge for FIPS
        print("Merging FIPS codes...")
        df_processed = pd.merge(df_processed,
                                df_fips[['county_fips', 'clean_county_for_merge', 'state_abbr_for_merge']],
                                on=['clean_county_for_merge', 'state_abbr_for_merge'],
                                how='left')

        # Merge for Population
        print("Merging population data...")
        if 'county_fips' in df_processed.columns:
            df_processed = pd.merge(df_processed, df_pop, on='county_fips', how='left')
        else:
            print("Warning: 'county_fips' column not created, skipping population merge.")
            df_processed['pop_2020'] = np.nan

        # Drop temporary merge columns
        df_processed.drop(columns=['clean_county_for_merge', 'state_abbr_for_merge'], inplace=True, errors='ignore')

    except Exception as e:
        print(f"Error loading or merging FIPS/Population data: {e}")
        df_processed['county_fips'] = np.nan
        df_processed['pop_2020'] = np.nan


    print("\nSample of processed data with latest features (incl. FIPS, Pop):")
    cols_to_show = ['county', 'state', 'county_fips', 'pop_2020', 'derived_pronoun', 'derived_weather_term']
    print(df_processed[cols_to_show].head(10))

    print("\nMissing values in latest features (incl. FIPS, Pop):")
    print(df_processed[['derived_month', 'derived_season', 'derived_year',
                        'derived_submitted_date', 'derived_day_night', 'derived_delay_days',
                        'derived_witness_grp', 'derived_class', 'derived_environment',
                        'derived_verbs', 'derived_words', 'derived_sent',
                        'derived_narr_len', 'derived_road_type', 'derived_elev',
                        'derived_pronoun', 'derived_weather_term', 'county_fips', 'pop_2020'
                        ]].isnull().sum())

    # --- Robust Numeric Trimming ---
    print(f"\nShape before trimming: {df_processed.shape}")
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()

    # Remove FIPS codes from trimming if they are numeric but shouldn't be trimmed by value
    # county_fips is string, but if any other ID-like numeric columns exist, exclude them here.
    # For now, all derived numeric columns seem like actual measurements or counts.
    # derived_year, derived_month, derived_delay_days, derived_narr_len, derived_elev, pop_2020, derived_sent
    # Latitude and longitude from original data might also be present if not dropped.
    # The problem says "every numeric column in the table" - this implies the final table.

    cols_to_trim = [col for col in numeric_cols if col not in ['latitude', 'longitude']] # Example: keep lat/lon as is, or handle separately.
                                                                                       # For now, let's assume all np.number columns are candidates.

    print(f"Numeric columns identified for potential trimming: {numeric_cols}")

    # Create a boolean mask, True for rows to keep, initialized to all True
    keep_mask = pd.Series([True] * len(df_processed), index=df_processed.index)

    for col in numeric_cols:
        # Skip columns with all NaNs or very few unique values if percentile calculation is problematic
        if df_processed[col].nunique(dropna=True) < 2: # Need at least 2 unique non-NaN values for percentile
            print(f"Skipping trimming for column '{col}' due to insufficient unique values.")
            continue

        q_low = df_processed[col].quantile(0.005)
        q_high = df_processed[col].quantile(0.995)

        print(f"Trimming column '{col}': Low={q_low}, High={q_high}")

        # Update mask: row must be > q_low AND < q_high. NaNs will not satisfy this.
        # Rows with NaN in the current numeric column will be effectively dropped by this condition
        # unless explicitly handled to keep them. The problem implies keeping rows *within* percentiles.

        # Current row's value for this column must be within bounds
        # NaNs in 'col' will result in False for both comparisons, so they are effectively excluded
        # by this logic unless q_low or q_high is also NaN (which shouldn't happen with dropna=True in nunique check).
        col_mask = (df_processed[col] > q_low) & (df_processed[col] < q_high)

        # For rows where the column is NaN, they should not fail the trim for *this* column.
        # They might fail for other numeric columns where they have non-NaN values outside bounds.
        # The requirement: "keep rows within the 0.5 – 99.5 percentile" for *every* numeric column.
        # This implies that if a value is NaN, it's not "within" the percentiles of actual numbers.
        # So, NaNs in numeric columns effectively lead to row removal if that column is used for trimming.
        # This seems like a reasonable interpretation.

        keep_mask = keep_mask & col_mask

    df_trimmed = df_processed[keep_mask].copy()
    print(f"Shape after trimming: {df_trimmed.shape}")
    print(f"Number of rows trimmed: {len(df_processed) - len(df_trimmed)}")

    # --- Generate points.json ---
    df_for_points = df_trimmed.copy() # Start with the trimmed data

    if len(df_for_points) > MAX_ROWS_POINTS_JSON:
        print(f"Sampling {MAX_ROWS_POINTS_JSON} rows for points.json from {len(df_for_points)} trimmed rows.")
        df_for_points = df_for_points.sample(n=MAX_ROWS_POINTS_JSON, random_state=RANDOM_SEED)
    else:
        print(f"Using all {len(df_for_points)} trimmed rows for points.json (less than or eq to {MAX_ROWS_POINTS_JSON}).")

    # Define all per-row fields required (as per problem description and derived columns)
    # This list should include all columns mentioned in the table in the problem description
    # plus any original useful ones like 'observed' text, lat/lon if they were preserved.
    # I'll list all derived columns and some key original ones.
    # The problem implies these are the columns created:
    # month, season, day_night, year, submitted, delay, witness_grp, class, environment,
    # verbs, words, elev, sent, road_type, narr_len, pronoun, weather_term, county_fips, pop_2020

    # Let's make sure to use the 'derived_' prefixed names for clarity, or final names.
    # The plan implies the 'output' columns are these derived ones.
    point_columns = [
        'derived_month', 'derived_season', 'derived_day_night', 'derived_year',
        'derived_submitted_date', 'derived_delay_days', 'derived_witness_grp',
        'derived_class', 'derived_environment', 'derived_verbs', 'derived_words',
        'derived_elev', 'derived_sent', 'derived_road_type', 'derived_narr_len',
        'derived_pronoun', 'derived_weather_term', 'county_fips', 'pop_2020',
        # Also include original potentially useful fields for tooltips/display if not already derived:
        'observed', # Original text
        'location_details', # For context, if needed
        'latitude', 'longitude', # Original coordinates
        'title', # Report title
        'Time And Conditions', # Original for context
        'Nearest Road',
        'Other Witnesses',
        'Also Noticed',
        'classification', # Original class
        'Season', # Original season
        'Month', # Original month
        'year' # Original year
    ]

    # Filter df_for_points to only include existing columns from point_columns
    existing_point_columns = [col for col in point_columns if col in df_for_points.columns]
    df_points_json = df_for_points[existing_point_columns]

    # Convert datetime objects to ISO format string for JSON serialization
    if 'derived_submitted_date' in df_points_json.columns:
        # NaT values in 'derived_submitted_date' will become NaN after strftime,
        # which to_json handles by converting to null.
        df_points_json.loc[:, 'derived_submitted_date'] = df_points_json['derived_submitted_date'].dt.strftime('%Y-%m-%dT%H:%M:%S')


    print(f"\nGenerating {OUTPUT_POINTS_JSON} with {len(df_points_json)} rows and {len(existing_point_columns)} columns.")
    try:
        df_points_json.to_json(OUTPUT_POINTS_JSON, orient='records', indent=2)
        print(f"{OUTPUT_POINTS_JSON} generated successfully.")
    except Exception as e:
        print(f"Error generating {OUTPUT_POINTS_JSON}: {e}")


    # TODO: Generate agg.json

    # --- Generate agg.json (Aggregations) ---
    # Use df_trimmed for aggregations as per problem spec
    # If df_trimmed is empty, many aggregates will be empty or might error.
    # Add checks for empty dataframes where necessary.

    print(f"\nGenerating aggregations for {OUTPUT_AGG_JSON} from df_trimmed ({len(df_trimmed)} rows)...")
    aggregates = {}

    # Placeholder for movie release dates and internet cutoff
    BIGFOOT_MOVIE_RELEASES = { # year: event_name
        1972: "The Legend of Boggy Creek",
        1987: "Harry and the Hendersons",
        2008: "Minerva Monster"
    }
    INTERNET_ERA_CUTOFF_YEAR = 2000

    # Ensure df_trimmed has the necessary columns, even if empty
    # This helps prevent KeyErrors if df_trimmed is empty but columns are expected
    required_agg_cols = ['derived_month', 'derived_day_night', 'derived_year', 'derived_submitted_date',
                         'derived_witness_grp', 'derived_class', 'derived_environment', 'derived_verbs',
                         'derived_words', 'derived_season', 'derived_elev', 'derived_sent',
                         'derived_delay_days', 'derived_road_type', 'derived_pronoun', 'derived_narr_len',
                         'derived_weather_term', 'county_fips', 'pop_2020']

    df_agg_source = df_trimmed.copy() # Use df_trimmed
    # If df_agg_source is empty, we can still try to define structures, but counts will be 0.
    # Alternatively, for testing with dummy data, one might use df_processed if df_trimmed is empty.
    # For now, strictly follow "after trimming".

    # 1. Month counts
    if not df_agg_source.empty and 'derived_month' in df_agg_source:
        aggregates['month_counts'] = df_agg_source['derived_month'].value_counts().sort_index().to_dict()
    else:
        aggregates['month_counts'] = {}

    # 2. Day/night counts
    if not df_agg_source.empty and 'derived_day_night' in df_agg_source:
        aggregates['day_night_counts'] = df_agg_source['derived_day_night'].value_counts().to_dict()
    else:
        aggregates['day_night_counts'] = {}

    # 3. Media timeline (monthly sightings)
    # For this, we need a reliable date for each sighting, derived_submitted_date or derived_year/month
    if not df_agg_source.empty and 'derived_submitted_date' in df_agg_source and pd.api.types.is_datetime64_any_dtype(df_agg_source['derived_submitted_date']):
        # Ensure derived_submitted_date is datetime before resampling
        temp_df = df_agg_source.copy()
        temp_df['derived_submitted_date'] = pd.to_datetime(temp_df['derived_submitted_date'], errors='coerce')
        temp_df.dropna(subset=['derived_submitted_date'], inplace=True)
        if not temp_df.empty:
            monthly_sightings = temp_df.set_index('derived_submitted_date').resample('M').size()
            aggregates['media_timeline_monthly'] = {date.strftime('%Y-%m'): count for date, count in monthly_sightings.items()}
        else:
            aggregates['media_timeline_monthly'] = {}
    else: # Fallback or if derived_submitted_date is not suitable
        aggregates['media_timeline_monthly'] = {}
    aggregates['media_timeline_releases'] = BIGFOOT_MOVIE_RELEASES


    # 4. Witness x class crosstab
    if not df_agg_source.empty and 'derived_witness_grp' in df_agg_source and 'derived_class' in df_agg_source:
        aggregates['witness_class_crosstab'] = pd.crosstab(df_agg_source['derived_witness_grp'], df_agg_source['derived_class']).to_dict()
    else:
        aggregates['witness_class_crosstab'] = {}

    # 5. Environment-verb edge list (complex, placeholder for now if empty)
    # This requires 'derived_environment' and 'derived_verbs'
    # For each environment, find top N most frequent co-occurring verbs
    # Example: [{source: environment_A, target: verb_X, value: frequency}, ...]
    aggregates['environment_verb_edges'] = [] # Placeholder
    if not df_agg_source.empty and 'derived_environment' in df_agg_source and 'derived_verbs' in df_agg_source:
        env_verb_pairs = []
        df_agg_source_nlp = df_agg_source.dropna(subset=['derived_environment', 'derived_verbs'])
        for _, row in df_agg_source_nlp.iterrows():
            env = row['derived_environment']
            if pd.notna(env) and isinstance(row['derived_verbs'], list):
                for verb in row['derived_verbs']:
                    env_verb_pairs.append((env, verb))

        if env_verb_pairs:
            df_env_verb = pd.DataFrame(env_verb_pairs, columns=['environment', 'verb'])
            edge_counts = df_env_verb.groupby(['environment', 'verb']).size().reset_index(name='value')
            # Optional: filter for top N verbs per environment or overall top edges
            aggregates['environment_verb_edges'] = edge_counts.to_dict(orient='records')


    # 6. Season word-freq (dictionary of {season: {word: freq}})
    aggregates['season_word_freq'] = {}
    if not df_agg_source.empty and 'derived_season' in df_agg_source and 'derived_words' in df_agg_source:
        df_agg_source_words = df_agg_source.dropna(subset=['derived_season', 'derived_words'])
        for season, group in df_agg_source_words.groupby('derived_season'):
            all_words_in_season = [word for sublist in group['derived_words'] for word in sublist]
            if all_words_in_season:
                freq_dist = nltk.FreqDist(all_words_in_season)
                aggregates['season_word_freq'][season] = dict(freq_dist.most_common(30)) # Top 30 words


    # 7. Altitude-sent scatter (sub-sampled list of [alt, sent] pairs)
    aggregates['altitude_sentiment_scatter'] = []
    if not df_agg_source.empty and 'derived_elev' in df_agg_source and 'derived_sent' in df_agg_source:
        scatter_data = df_agg_source[['derived_elev', 'derived_sent']].dropna()
        if len(scatter_data) > 1000: # Subsample if large
            scatter_data = scatter_data.sample(n=1000, random_state=RANDOM_SEED)
        aggregates['altitude_sentiment_scatter'] = scatter_data.values.tolist()


    # 8. Delay x road_type summary (for box plots: min, q1, median, q3, max)
    aggregates['delay_road_type_summary'] = {}
    if not df_agg_source.empty and 'derived_delay_days' in df_agg_source and 'derived_road_type' in df_agg_source:
        df_delay_road = df_agg_source.dropna(subset=['derived_delay_days', 'derived_road_type'])
        if not df_delay_road.empty:
            # Using describe() to get quantiles
            summary_stats = df_delay_road.groupby('derived_road_type')['derived_delay_days'].describe(
                percentiles=[.25, .5, .75]
            )
            # Convert to a more JSON-friendly dict format
            for road_type, stats in summary_stats.iterrows():
                aggregates['delay_road_type_summary'][road_type] = {
                    'min': stats['min'],
                    'q1': stats['25%'],
                    'median': stats['50%'],
                    'q3': stats['75%'],
                    'max': stats['max'],
                    'count': int(stats['count'])
                }


    # 9. Pronoun counts
    if not df_agg_source.empty and 'derived_pronoun' in df_agg_source:
        aggregates['pronoun_counts'] = df_agg_source['derived_pronoun'].value_counts().to_dict()
    else:
        aggregates['pronoun_counts'] = {}


    # 10. Pre/post internet summaries
    aggregates['pre_post_internet'] = {'class_summary': {}, 'narr_len_summary': {}}
    if not df_agg_source.empty and 'derived_year' in df_agg_source:
        df_internet = df_agg_source.dropna(subset=['derived_year'])
        if not df_internet.empty:
            df_internet['era'] = df_internet['derived_year'].apply(lambda y: 'post_internet' if y >= INTERNET_ERA_CUTOFF_YEAR else 'pre_internet')

            # Class summary pre/post
            if 'derived_class' in df_internet:
                class_summary = pd.crosstab(df_internet['era'], df_internet['derived_class'])
                aggregates['pre_post_internet']['class_summary'] = class_summary.to_dict()

            # Narrative length summary pre/post (for violin plots)
            if 'derived_narr_len' in df_internet:
                narr_len_summary = df_internet.groupby('era')['derived_narr_len'].agg(
                    ['min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max', 'mean', 'std', 'count']
                ).rename(columns={'<lambda_0>': 'q1', '<lambda_1>': 'q3'})
                aggregates['pre_post_internet']['narr_len_summary'] = narr_len_summary.to_dict(orient='index')


    # 11. Season x weather matrix
    if not df_agg_source.empty and 'derived_season' in df_agg_source and 'derived_weather_term' in df_agg_source:
        aggregates['season_weather_matrix'] = pd.crosstab(df_agg_source['derived_season'], df_agg_source['derived_weather_term']).to_dict()
    else:
        aggregates['season_weather_matrix'] = {}


    # 12. County sightings + population
    aggregates['county_sightings_population'] = []
    if not df_agg_source.empty and 'county_fips' in df_agg_source and 'pop_2020' in df_agg_source:
        # Count sightings per county
        county_sightings = df_agg_source['county_fips'].value_counts().reset_index()
        county_sightings.columns = ['county_fips', 'sightings']

        # Get unique population per county (assuming pop_2020 is already correctly mapped)
        county_pop = df_agg_source[['county_fips', 'pop_2020']].drop_duplicates('county_fips').dropna()

        if not county_sightings.empty and not county_pop.empty:
            county_summary = pd.merge(county_sightings, county_pop, on='county_fips', how='left')
            aggregates['county_sightings_population'] = county_summary.to_dict(orient='records')


    # Save aggregates to JSON
    print(f"\nSaving aggregates to {OUTPUT_AGG_JSON}...")
    try:
        with open(OUTPUT_AGG_JSON, 'w') as f:
            json.dump(aggregates, f, indent=2, default=lambda x: x.isoformat() if hasattr(x, 'isoformat') else x) # Handle potential datetime objects if any slipped through
        print(f"{OUTPUT_AGG_JSON} generated successfully.")
    except Exception as e:
        print(f"Error generating {OUTPUT_AGG_JSON}: {e}")


    # --- Print JSON file sizes ---
    try:
        points_json_size = os.path.getsize(OUTPUT_POINTS_JSON)
        print(f"\nSize of {OUTPUT_POINTS_JSON}: {points_json_size} bytes")
    except OSError as e:
        print(f"Could not get size of {OUTPUT_POINTS_JSON}: {e}")

    try:
        agg_json_size = os.path.getsize(OUTPUT_AGG_JSON)
        print(f"Size of {OUTPUT_AGG_JSON}: {agg_json_size} bytes")
    except OSError as e:
        print(f"Could not get size of {OUTPUT_AGG_JSON}: {e}")

    print("\nPreprocessing script finished.")

if __name__ == "__main__":
    main()
