import os
import re
from typing import List, Tuple
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv

load_dotenv()

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ENTSOE_API_KEY")
# vectorized month map
_MONTH_MAP = {
    'Jan':  1, 'Feb':  2, 'Mar':  3, 'Apr':  4, 'May':  5, 'Jun':  6,
    'Jul':  7, 'Aug':  8, 'Sep':  9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# ENTSO-E API configuration
ENTSOE_API_BASE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

def set_entsoe_api_key(api_key: str):
    """
    Set the ENTSO-E API key for accessing consumption data.
    
    Args:
        api_key (str): Your ENTSO-E API key
    """
    global ENTSOE_API_KEY
    ENTSOE_API_KEY = api_key
    print("✅ ENTSO-E API key set successfully!")
    print("You can now use get_dataframe() for Portugal, Sweden, and Spain")

# Country bidding zone mappings for ENTSO-E API
COUNTRY_BIDDING_ZONES = {
    'portugal': '10YPT-REN------W',  # Portugal
    # 'sweden': '10X1001A1001A418',   # Sweden
    'sweden_malmo': '10Y1001A1001A47J',
    'sweden_umea': '10Y1001A1001A45N',
    'sweden_stockholm': '10Y1001A1001A46L',
    'sweden_lulea': '10Y1001A1001A44P', # Lulea
    'spain': '10YES-REE------0',      # Spain
    'austria': '10YAT-APG------L',   # Austria
}

def load_entsoe_data(country: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load data from ENTSO-E API for the specified country and date range.
    """
    if country.lower() not in COUNTRY_BIDDING_ZONES:
        raise ValueError(f"Unsupported country for ENTSO-E API: {country}")
    
    if ENTSOE_API_KEY is None:
        raise ValueError("ENTSO-E API key not set. Please use set_entsoe_api_key() first.")
    
    # API parameters
    params = {
        'documentType': 'A65', # This indicates that the system is on a System total Load (so it is consumed, passing in the transmission system, not generated)
        'processType': 'A16',
        'outBiddingZone_Domain': COUNTRY_BIDDING_ZONES[country.lower()],
        'periodStart': start_date,
        'periodEnd': end_date,
        'securityToken': ENTSOE_API_KEY
    }
    
    try:
        # Assumes you have imported requests as requests and xml.etree.ElementTree as ET
        response = requests.get(ENTSOE_API_BASE_URL, params=params)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        data_points = []
        
        # This XML parsing logic seems correct, so it is preserved
        for time_series in root.findall('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}TimeSeries'):
            time_interval = time_series.find('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}timeInterval')
            if time_interval is not None:
                start_elem = time_interval.find('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}start')
                if start_elem is None or start_elem.text is None:
                    continue  # skip this time_series if start is missing
                start_time = start_elem.text
                start_dt = datetime.strptime(start_time, '%Y-%m-%dT%H:%MZ')
                resolution_elem = time_series.find('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}resolution')
                resolution = resolution_elem.text if resolution_elem is not None else 'PT60M'
                
                for point in time_series.findall('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}Point'):
                    position_elem = point.find('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}position')
                    quantity_elem = point.find('.//{urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0}quantity')
                    if position_elem is None or position_elem.text is None or quantity_elem is None or quantity_elem.text is None:
                        continue  # skip this point if position or quantity is missing
                    position = int(position_elem.text)
                    quantity = float(quantity_elem.text)
                    
                    # Calculate timestamp based on resolution
                    if resolution == 'PT15M':
                        timestamp = start_dt + timedelta(minutes=15*(position-1))
                    elif resolution == 'PT30M':
                        timestamp = start_dt + timedelta(minutes=30*(position-1))
                    else:  # PT60M or default to hourly
                        timestamp = start_dt + timedelta(hours=position-1)
                    
                    data_points.append({'time': timestamp, 'consumption': quantity})
        
        if not data_points:
            raise ValueError("No data points found in API response")
        
        # Create DataFrame
        df = pd.DataFrame(data_points)
        df = df.sort_values('time').reset_index(drop=True)
        
        df = df.rename(columns={'consumption': 'Consumption'})
        df['Time'] = df['time']
        
        # --- START OF THE FIX ---
        # Drop the original 'time' column after creating the 'Time' column to avoid duplication
        df = df.drop(columns=['time'])
        # --- END OF THE FIX ---

        df['Year'] = df['Time'].dt.year
        
        return df.set_index('Time')
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")
    except ET.ParseError as e:
        raise RuntimeError(f"XML parsing failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

def get_entsoe_date_range(country: str) -> Tuple[List[int], List[int]]:
    """
    Generate train and test years for ENTSO-E API countries.
    Uses available data from 2019-2024 similar to Brazil.
    
    Args:
        country (str): Country name
    
    Returns:
        Tuple[List[int], List[int]]: (train_years, test_years)
    """
    # For now, use the same split as Brazil
    # In a real implementation, you might want to check API availability
    train_years = list(range(2019, 2024))
    test_years = [2024]
    
    return train_years, test_years

def load_entsoe_consumption(country: str, train_years: List[int], test_years: List[int]) -> pd.DataFrame:
    """Orchestrates loading and plotting for ENTSO-E countries."""
    all_data = []
    for year in train_years + test_years:
        start_date = f"{year}01010000"
        end_date = f"{year}12312300"
        try:
            # Assumes you have a `load_entsoe_data` function that returns a DataFrame
            # with a 'Time' index and 'Consumption' column.
            if country == 'sweden':
                # swedish_regions = ['sweden_malmo', 'sweden_umea', 'sweden_stockholm', 'sweden_lulea']
                swedish_regions = ['sweden_stockholm']
                regional_dfs_for_year = []
                
                for region in swedish_regions:
                    try:
                        # Fetch data for each region
                        region_data = load_entsoe_data(region, start_date, end_date)
                        if not region_data.empty:
                            regional_dfs_for_year.append(region_data)
                    except Exception as region_e:
                        print(f"Warning: Could not load data for {region} in {year}: {region_e}")

                if regional_dfs_for_year:
                    # Sum the consumption across all regions for the current year
                    # Group by the index (time) and sum only the 'Consumption' column
                    year_data = pd.concat(regional_dfs_for_year).groupby(level=0)['Consumption'].sum().to_frame()
                    # Correctly set the Year column after grouping
                    year_data['Year'] = year 
                    all_data.append(year_data)
                else:
                    print(f"Warning: No data could be loaded for Sweden in {year}.")
            else:
                year_data = load_entsoe_data(country, start_date, end_date)
                all_data.append(year_data)
        except Exception as e:
            print(f"Warning: Could not load ENTSO-E data for {year}: {e}")
    
    if not all_data:
        raise RuntimeError(f"No ENTSO-E data could be loaded for {country}")
        
    df = pd.concat(all_data).sort_index()
    
    # --- START OF THE FIX ---
    # Handle duplicate timestamps (e.g., from DST changes) by averaging them.
    # This ensures the index is unique before reindexing.
    if df.index.has_duplicates:
        print("Warning: Duplicate timestamps found. Aggregating by taking the mean.")
        print( df.index[df.index.duplicated()] )
        print( df[df.index.duplicated(keep=False)] )
        print("Duplicates have been aggregated by averaging their values.")
        print(df.head(10))
        df = df.groupby(df.index).mean()
    # --- END OF THE FIX ---

    # Create a full hourly date range and reindex to fill any gaps.
    # This will now work because the index is unique.
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_range)
    df['Consumption'] = df['Consumption'].interpolate(method='linear')

    df.index = pd.to_datetime(df.index)
    df.index.name = 'Time'  # Name the index
    # Recalculate Year from the index after reindexing (previous Year values may have NaN)
    df['Year'] = df.index.year

    country_names = {
        'brazil': 'Brazil',
        'sweden': 'Sweden',
        'switzerland': 'Switzerland',
        'spain': 'Spain',
        'portugal': 'Portugal',
        'austria': 'Austria'
    }
    
    # --- START OF THE FIX ---
    # Pass the last year of the training set, not the test set list.
    plot_time_series(df, country_names[country], train_years[-1])
    # --- END OF THE FIX ---

    plot_average_by_period(df, 'hour', f"Average Hourly Consumption - {country_names[country]}", "Hour of Day")
    plot_average_by_period(df, 'dow', f"Average Consumption by Day of Week - {country_names[country]}", "Day of Week")
    plot_average_by_period(df, 'month', f"Average Monthly Consumption - {country_names[country]}", "Month")

    annual = df.groupby('Year')['Consumption'].mean().reset_index()
    fig = px.bar(annual, x='Year', y='Consumption', title=f'Average Annual Consumption - {country_names[country]}', labels={'Consumption':'Avg. Consumption (MW)'}, template='plotly_white')
    fig.show()
    
    return df.reset_index().rename(columns={'Time': 'time', 'Consumption': 'value'})[['time', 'value']]

def load_consumption_data(
    files: List[str],
    time_col: str = "Time",
    demand_col: str = "Demand"
) -> pd.DataFrame:
    dfs = []
    for path in files:
        df = pd.read_excel(path, usecols=[time_col, demand_col], dtype={time_col: str})
        parts = df[time_col].str.extract(
            r'(?P<day>\d{1,2})\s+de\s+(?P<month>[A-Za-zçÇéÉ]{3})\s+de\s+'
            r'(?P<year>\d{2,4})\s*-\s*(?P<hour>\d{1,2})\s*h'
        )
        parts['month'] = parts['month'].str.capitalize().map(_MONTH_MAP)
        parts['year'] = parts['year'].astype(int)
        parts.loc[parts['year'] < 100, 'year'] += 2000
        parts['day'] = parts['day'].astype(int)
        parts['hour'] = parts['hour'].astype(int)
        df[time_col] = pd.to_datetime(parts[['year','month','day','hour']])
        df = df.rename(columns={demand_col: "Consumption"})
        df['Year'] = df[time_col].dt.year
        dfs.append(df.set_index(time_col))

    return pd.concat(dfs).sort_index()

def plot_time_series(df: pd.DataFrame, country_name: str, train_end_year: int):
    """Plots the full time series, splitting it and coloring the test set by country flag."""

    # --- START OF THE FIX ---
    # 1. Define a dictionary to map countries to their flag colors.
    #    We use lowercase keys for easy, case-insensitive matching.
    FLAG_COLORS = {
        'brazil': '#009C3B',       # Brazilian Green
        'sweden': '#006AA7',       # Swedish Blue
        'switzerland': '#DA291C',  # Swiss Red
        'spain': '#C60B1E',        # Spanish Red
        'portugal': '#006600',     # Portuguese Green
        'austria': '#ED2939',      # Austrian Red
        'centro-oeste': "#A38611", # Using a gold/yellow color for Centro-Oeste
    }

    # 2. Look up the color for the given country.
    #    .lower() makes the lookup robust against capitalization (e.g., "Brazil", "brazil").
    #    .get() provides a default color ('#009C3B') if the country is not in the map.
    test_color = FLAG_COLORS.get(country_name.lower(), '#009C3B')
    # --- END OF THE FIX ---

    df_plot = df.copy().reset_index().rename(columns={'index': 'Time'})
    division_time = pd.Timestamp(f"{train_end_year}-12-31 23:00:00")
    three_am = pd.Timestamp(f"2019-10-08 03:00:00")
    nine_am = pd.Timestamp(f"2019-10-08 09:00:00")

    df_train = df_plot[df_plot['Time'] <= division_time]
    df_test = df_plot[df_plot['Time'] > division_time]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_train['Time'], y=df_train['Consumption'], name='Train',
        line=dict(color='gray', width=2), mode='lines'
    ))
    
    # 3. Use the selected color variable for the test set line.
    fig.add_trace(go.Scatter(
        x=df_test['Time'], y=df_test['Consumption'], name='Test',
        line=dict(color=test_color, width=2), mode='lines'
    ))
    
    fig.add_shape(
        type='line', x0=division_time, x1=division_time, y0=0, y1=1,
        yref='paper', line=dict(color='black', width=2)
    )
    # fig.add_shape(
    #     type='line', x0=nine_am, x1=nine_am, y0=0, y1=1,
    #     yref='paper', line=dict(color='magenta', width=2, dash='dash')
    # )
    # fig.add_shape(
    #     type='line', x0=three_am, x1=three_am, y0=0, y1=1,
    #     yref='paper', line=dict(color='green', width=2, dash='dash')
    # )
    if country_name == str('Switzerland'):
        fig.update_layout(
            # title=f'Energy Consumption in {country_name} Over Time in 15-Minute Intervals',
            xaxis_title='Time', 
            yaxis_title='Consumption (MWh)',
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            template='plotly_white', 
            legend=dict(font=dict(size=20), yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
    else:
        fig.update_layout(
            # title=f'Hourly Energy Consumption in {country_name} Over Time',
            xaxis_title='Time', 
            yaxis_title='Consumption (MWh)',
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            template='plotly_white', 
            legend=dict(font=dict(size=20), yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
    
    fig.show()
    
    # # Alternative: Save directly as PDF using kaleido (requires: pip install kaleido)
    # # Uncomment the lines below to automatically save a PDF file:
    # try:
    #     fig.write_image(f"Not_results_plots/{country_name}_Consumption.pdf", 
    #                     width=1400, height=800, scale=2)
    #     print(f"✅ PDF saved to plots/energy_consumption_{country_name.lower()}.pdf")
    # except Exception as e:
    #     print(f"⚠️ Could not save PDF: {e}")
    #     print("Install kaleido with: pip install kaleido")


def plot_average_by_period(df: pd.DataFrame, period: str, title: str, xlabel: str):
    d = df.copy().reset_index()
    if period == 'dow':
        d['Period'] = d['Time'].dt.day_name()
        category_orders = {'Period': WEEKDAYS}
    else:
        d['Period'] = getattr(d['Time'].dt, period)
        category_orders = None

    agg = d.groupby('Period', observed=True)['Consumption'].mean().reset_index()
    fig = px.bar(
        agg, x='Period', y='Consumption',
        title=title,
        labels={'Consumption':'Avg. Consumption (MWh)', 'Period':xlabel},
        template='plotly_white',
        category_orders=category_orders
    )
    fig.show()

def load_and_plot_consumption(
    files: List[str],
    train_years: List[int],
    test_years: List[int],
    output_dir: str = "plots"
):
    os.makedirs(output_dir, exist_ok=True)
    df = load_consumption_data(files)

    plot_time_series(df, 'Brazil', train_years[-1])

    plot_average_by_period(df, 'hour', "Average Hourly Consumption", "Hour of Day")
    plot_average_by_period(df, 'dow', "Average Consumption by Day of Week", "Day of Week")
    plot_average_by_period(df, 'month', "Average Monthly Consumption", "Month")

    # --- Average Annual Consumption ---
    annual = df.groupby('Year')['Consumption'].mean().reset_index()
    fig = px.bar(
        annual, x='Year', y='Consumption',
        title='Average Annual Consumption',
        labels={'Consumption':'Avg. Consumption (MWh)'},
        template='plotly_white'
    )
    
    fig.show()

    df_clean = df.reset_index()[['Time', 'Consumption']].rename(
        columns={'Time': 'time', 'Consumption': 'value'}
        )

    return df_clean

def load_consumption_data_from_csv(file_path: str) -> pd.DataFrame:
    """Loads consumption data from a single CSV file."""
    print(f"Reading data from: {file_path}")
    df = pd.read_csv(file_path, usecols=['datetime', 'value'], parse_dates=['datetime'])
    
    # Rename columns to the standard format
    df = df.rename(columns={"datetime": "Time", "value": "Consumption"})
    
    # --- FIX ---
    # 1. Create the 'Year' column while 'Time' is still a regular column.
    #    Pylance understands this perfectly.
    df['Year'] = df['Time'].dt.year
    
    # 2. Now, set the index.
    df = df.set_index("Time")
    
    return df.sort_index()

# --- NEW FUNCTION START ---
# This orchestrator calls the new CSV loading function and then plots the data
def load_and_plot_consumption_from_csv(
    files: List[str],
    train_years: List[int],
    test_years: List[int]
):
    """Orchestrates loading, plotting, and cleaning for CSV-based data."""
    # Since we have one file with all data, we just use the first file in the list
    df = load_consumption_data_from_csv(files[0])

    # The plot functions use the last year of the training set to draw the split line
    train_end_year = train_years[-1]
    
    plot_time_series(df, 'Centro-Oeste', train_end_year)
    plot_average_by_period(df, 'hour', "Average Hourly Consumption (Centro-Oeste)", "Hour of Day")
    plot_average_by_period(df, 'dow', "Average Consumption by Day of Week (Centro-Oeste)", "Day of Week")
    plot_average_by_period(df, 'month', "Average Monthly Consumption (Centro-Oeste)", "Month")

    annual = df.groupby('Year')['Consumption'].mean().reset_index()
    fig = px.bar(
        annual, x='Year', y='Consumption',
        title='Average Annual Consumption (Centro-Oeste)',
        labels={'Consumption':'Avg. Consumption (MWh)'},
        template='plotly_white'
    )
    fig.show()
    
    # Return the dataframe in the standard format for the rest of your notebook
    df_clean = df.reset_index().rename(
        columns={'Time': 'time', 'Consumption': 'value'}
    )
    return df_clean[['time', 'value']]

def load_swiss_data(files: List[str]) -> pd.DataFrame:
    """Loads and processes multiple Swiss energy Excel files at 15-minute resolution."""
    dfs = []
    for file in files:
        try:
            year = int(os.path.basename(file).split("-")[-1].split(".")[0])
            df = pd.read_excel(
                file,
                sheet_name="Zeitreihen0h15",
                skiprows=1,
                usecols=["Temps", "Energie_kWh"]
            )
            
            df["Temps"] = pd.to_datetime(df["Temps"], dayfirst=True, errors='coerce')
            df = df.dropna(subset=["Temps"])
            df = df.rename(columns={"Temps": "Time", "Energie_kWh": "Consumption"})

            # 1. Convert consumption from kWh to MWh on the original 15-minute data
            df['Consumption'] = pd.to_numeric(df['Consumption'], errors='coerce').fillna(0) / 1000

            df = df.set_index("Time")
            df['Year'] = year
            
            # 2. Append the processed 15-minute dataframe (resampling is removed)
            dfs.append(df)
            
        except Exception as e:
            print(f"Warning: Could not process file {file}. Error: {e}")
            
    return pd.concat(dfs).sort_index()

def load_and_plot_swiss_consumption(files: List[str], train_years: List[int], test_years: List[int]) -> pd.DataFrame:
    """Orchestrates loading and plotting for Switzerland."""
    df = load_swiss_data(files)
    
    country_name = "Switzerland"
    plot_time_series(df, country_name, train_end_year=train_years[-1])
    plot_average_by_period(df, 'hour', f"Average Hourly Consumption - {country_name}", "Hour of Day")
    plot_average_by_period(df, 'dow', f"Average Consumption by Day of Week - {country_name}", "Day of Week")
    plot_average_by_period(df, 'month', f"Average Monthly Consumption - {country_name}", "Month")

    annual = df.groupby('Year')['Consumption'].mean().reset_index()
    fig = px.bar(annual, x='Year', y='Consumption', title=f'Average Annual Consumption - {country_name}', labels={'Consumption':'Avg. Consumption (MWh)'}, template='plotly_white')
    fig.show()

    # Return cleaned dataframe in the standard format
    return df.reset_index().rename(columns={'Time': 'time', 'Consumption': 'value'})[['time', 'value']]


def get_dataframe(country_name: str) -> pd.DataFrame:
    """
    Generate a dataframe for the specified country.
    
    Args:
        country_name (str): Name of the country. Supports various formats:
            - Brazil: "Brazil", "BR", "Brasil"
            - Sweden: "Sweden", "SE", "Sverige"
            - Switzerland: "Switzerland", "CH", "Svizra", "Schweiz", "Suisse", "Svizzera"
            - Portugal: "Portugal", "PT"
            - Spain: "Spain", "España", "ES"
    
    Returns:
        pd.DataFrame: Cleaned dataframe with time and value columns
    """
    # Normalize country name to lowercase for comparison
    country_lower = country_name.lower().strip()
    
    # Brazil handling
    if country_lower in ['brazil', 'br', 'brasil']:
        files = [
            "Datasets/Brazil/BR_demand_2019.xlsx",
            "Datasets/Brazil/BR_demand_2020.xlsx",
            "Datasets/Brazil/BR_demand_2021.xlsx",
            "Datasets/Brazil/BR_demand_2022.xlsx",
            "Datasets/Brazil/BR_demand_2023.xlsx",
            "Datasets/Brazil/BR_demand_2024.xlsx",
        ]
        train_years = list(range(2019, 2024))
        test_years = [2024]
        
        return load_and_plot_consumption(files, train_years, test_years)
    
    # Brazilian Centro-Oeste handling
    elif country_lower in ['centro-oeste', 'goiânia', 'pequi']:
        # Note: Using double backslashes for the path to be safe in Python
        files = [
            "Datasets\\data_total_arquivo_Yan_centro-oeste_sudeste.csv"
        ]
        # Adjust years based on your CSV's date range
        train_years = list(range(2015, 2019))
        test_years = [2019]
        
        return load_and_plot_consumption_from_csv(files, train_years, test_years)
    
    # Sweden handling using ENTSO-E API
    elif country_lower in ['sweden', 'se', 'sverige']:
        train_years, test_years = get_entsoe_date_range('sweden')
        return load_entsoe_consumption('sweden', train_years, test_years)
    
    # Switzerland handling
    elif country_lower in ['switzerland', 'ch', 'svizra', 'schweiz', 'suisse', 'svizzera']:
        files = [f"Datasets/Switzerland/EnergieUebersichtCH-{year}.xlsx" for year in range(2019, 2025)]
        train_years = list(range(2019, 2024))
        test_years = [2024]
        return load_and_plot_swiss_consumption(files, train_years, test_years)
    
    # Portugal handling using ENTSO-E API
    elif country_lower in ['portugal', 'pt']:
        train_years, test_years = get_entsoe_date_range('portugal')
        return load_entsoe_consumption('portugal', train_years, test_years)

    # Spain handling using ENTSO-E API
    elif country_lower in ['spain', 'españa', 'es']:
        train_years, test_years = get_entsoe_date_range('spain')
        return load_entsoe_consumption('spain', train_years, test_years)
    
    
    # Spain handling using ENTSO-E API
    elif country_lower in ['austria', 'österreich', 'at']:
        train_years, test_years = get_entsoe_date_range('austria')
        return load_entsoe_consumption('austria', train_years, test_years)

    else:
        raise ValueError(f"Unsupported country: {country_name}. Supported countries: Brazil, Sweden, Switzerland, Portugal, Spain")

if __name__ == "__main__":
    # Example usage for Brazil
    try:
        df = get_dataframe("Brazil")
        print("Brazil data loaded successfully!")
        print(df.tail())
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("For Portugal, Sweden, and Spain, you need to set your API key first:")
    print("set_entsoe_api_key('YOUR_ACTUAL_API_KEY')")
    print("Then you can use:")
    print("df = get_dataframe('Portugal')  # or 'Sweden', 'Spain'")
    df = get_dataframe('Portugal')
    print("="*60)

def know_processing_unit():
    """
    Detects and prints information about the CPU and available GPUs.
    """
    print("--- Processing Unit Information ---")
    
    # --- CPU Information ---
    cpu_info_str = "CPU not found"
    try:
        # Using cpuinfo for detailed CPU name
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        cpu_info_str = info.get('brand_raw', 'N/A')
    except ImportError:
        # Fallback to platform if cpuinfo is not installed
        import platform
        cpu_info_str = f"{platform.processor()} (For more details, run: pip install py-cpuinfo)"
    except Exception as e:
        # General fallback
        import platform
        cpu_info_str = f"{platform.processor()} (cpuinfo check failed: {e})"
        
    print(f"CPU: {cpu_info_str}")

    # --- GPU Information ---
    print("\n--- GPU & Accelerator Status ---")
    
    # Check with TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("✅ TensorFlow is using GPU.")
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  - Detected: {details.get('device_name', 'N/A')}")
        else:
            print("ℹ️ TensorFlow is using CPU. No compatible GPU found by TensorFlow.")
    except ImportError:
        print("⚠️ TensorFlow not found. Cannot check its GPU status.")
    except Exception as e:
        print(f"❌ Error checking TensorFlow GPU status: {e}")

    # Check with Numba for CUDA
    try:
        from numba import cuda
        if cuda.is_available():
            print("✅ Numba detected a CUDA-enabled GPU.")
            for i, gpu in enumerate(cuda.gpus):
                 print(f"  - Detected: {gpu.name.decode()}")
        else:
            print("ℹ️ Numba: No CUDA-enabled GPU detected.")
    except ImportError:
        print("⚠️ Numba not found. Cannot check its CUDA status.")
    except Exception as e:
        print(f"❌ Error checking Numba CUDA status: {e}")
        
    print("\n" + "-"*35)



