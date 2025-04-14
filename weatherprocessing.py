import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from scipy.stats import ttest_ind
import thermofeel as tf
import os

# File paths
FORECAST_WEATHER_PATH = "C:\\Users\\tzaar\\Downloads\\5CDatathon\\data_and_starter_code\\data\\forecast_weather.csv"
TRAIN_PATH = "C:\\Users\\tzaar\\Downloads\\5CDatathon\\data_and_starter_code\\data\\train.csv"
LITHUANIA_DATA_PATH = "C:\\Users\\tzaar\\Downloads\\Lithuania Data - Sheet1 (1).csv"

# Create output directory for plots
PLOTS_DIR = "weather_comparison_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Constants for solar radiation in Lithuania (kWh/m2/day)
SOLAR_RADIATION_BY_MONTH = {
    'January': 0.60,
    'February': 1.50,
    'March': 2.69,
    'April': 4.43,
    'May': 5.21,
    'June': 5.80,
    'July': 5.46,
    'August': 4.45,
    'September': 3.35,
    'October': 1.24,
    'November': 0.45,
    'December': 0.29
}

def load_estonia_data(file_path):
    """Load and preprocess Estonia weather data."""
    df = pd.read_csv(file_path)
    
    # Convert datetime
    df['origin_datetime'] = pd.to_datetime(df['origin_datetime'], format='%d-%m-%Y %H:%M')
    df['date'] = df['origin_datetime'].dt.strftime('%Y-%m-%d')
    
    # Calculate weather metrics
    df['humidity'] = calculate_rh_vectorized(df['temperature'], df['dewpoint'])
    df['wind_speed'] = np.sqrt(df['10_metre_u_wind_component']**2 + df['10_metre_v_wind_component']**2)
    df['relative_humidity'] = df.apply(lambda row: relative_humidity(row['temperature'], row['dewpoint']), axis=1)
    
    # Group by date to get daily averages
    df_grouped = df.groupby('date').agg({
        'wind_speed': 'mean',
        'cloudcover_total': 'mean',
        'dewpoint': 'mean',
        'temperature': 'mean',
        'relative_humidity': 'mean',
        'direct_solar_radiation': 'mean',
        'surface_solar_radiation_downwards': 'mean',
        'snowfall': 'mean'
    }).reset_index()
    
    # Convert date to datetime for comparison
    df_grouped['date'] = pd.to_datetime(df_grouped['date'])
    
    # Convert solar radiation to kWh/m2/day for fair comparison with Lithuania data
    df_grouped['surface_solar_radiation_downwards'] = df_grouped['surface_solar_radiation_downwards'] * (1/1000) * 24
    
    return df_grouped

def load_lithuania_data(file_path):
    """Load and preprocess Lithuania weather data."""
    df_lithuania = pd.read_csv(file_path, header=None)
    
    # Initialize tracking
    final_data = []
    current_month_year = None
    
    # Parse the unusual format
    for index, row in df_lithuania.iterrows():
        first_val = str(row[0]).strip()
        
        # Detect new month-year headers
        if re.match(r"^[A-Za-z]{3,9} \d{4}$", first_val):
            current_month_year = first_val
            continue
        
        # Skip header rows like 'Max Avg Min...'
        if first_val.lower() in ['max', 'avg', 'min'] or current_month_year is None:
            continue

        # If it's a numeric day, build a full date
        try:
            day = int(first_val)
            full_date = pd.to_datetime(f"{current_month_year} {day}")
            new_row = [full_date] + row[1:].tolist()
            final_data.append(new_row)
        except ValueError:
            continue

    # Create column headers
    columns = ["Date",
            "Temp_Max", "Temp_Avg", "Temp_Min",
            "DewPoint_Max", "DewPoint_Avg", "DewPoint_Min",
            "Humidity_Max", "Humidity_Avg", "Humidity_Min",
            "WindSpeed_Max", "WindSpeed_Avg", "WindSpeed_Min",
            "Pressure_Max", "Pressure_Avg", "Pressure_Min"]

    lithuania_df = pd.DataFrame(final_data, columns=columns)
    
    # Clean the data
    lithuania_df = clean_lithuania_data(lithuania_df)
    
    # Calculate estimated cloud cover
    lithuania_df = calculate_cloud_cover(lithuania_df)
    
    # Add month column for solar radiation mapping
    lithuania_df['month'] = lithuania_df['Date'].dt.strftime('%B')
    lithuania_df['solar_radiation_kWh/m2/day'] = lithuania_df['month'].map(SOLAR_RADIATION_BY_MONTH)
    
    return lithuania_df

def clean_lithuania_data(lithuania_df):
    """Clean the Lithuania data by handling data types and missing values."""
    # Convert numeric columns
    numeric_columns = [
        'Temp_Max', 'Temp_Avg', 'Temp_Min', 'DewPoint_Max', 'DewPoint_Avg', 
        'DewPoint_Min', 'Humidity_Max', 'Humidity_Avg', 'Humidity_Min',
        'WindSpeed_Max', 'WindSpeed_Avg', 'WindSpeed_Min', 
        'Pressure_Max', 'Pressure_Avg', 'Pressure_Min'
    ]
    
    # Convert to numeric, coercing errors to NaN
    lithuania_df[numeric_columns] = lithuania_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with the column's mean
    lithuania_df.fillna(lithuania_df.mean(), inplace=True)
    
    return lithuania_df

def calculate_cloud_cover(lithuania_df):
    """Calculate estimated cloud cover for Lithuania data."""
    # Convert humidity to fraction
    humidity = lithuania_df['Humidity_Avg'] / 100
    
    # Dew point depression
    dpd = lithuania_df['Temp_Avg'] - lithuania_df['DewPoint_Avg']
    
    # Dew point factor (clipped between 0 and 1)
    dp_factor = (1 - dpd / 25).clip(lower=0, upper=1)
    
    # Estimated cloud cover (as a fraction)
    lithuania_df['Estimated_Cloud_Cover'] = (0.4 * humidity + 0.6 * dp_factor)
    
    return lithuania_df

def calculate_rh_vectorized(temp, dewpoint):
    """Calculate relative humidity using the Magnus-Tetens formula."""
    a = 17.625
    b = 243.04
    alpha_dp = (a * dewpoint) / (b + dewpoint)
    alpha_t = (a * temp) / (b + temp)
    return 100 * (np.exp(alpha_dp) / np.exp(alpha_t))

def vapor_pressure(T):
    """Calculate vapor pressure."""
    return 6.112 * np.exp(17.67 * T / (T + 243.5))

def relative_humidity(T, T_d):
    """Calculate relative humidity from temperature and dew point."""
    e_T = vapor_pressure(T)
    e_Td = vapor_pressure(T_d)
    return (e_Td / e_T) * 100

def filter_data_by_date_range(estonia_df, lithuania_df):
    """Filter Lithuania data to match Estonia's date range."""
    estonia_start_date = estonia_df['date'].min()
    estonia_end_date = estonia_df['date'].max()
    
    lithuania_filtered = lithuania_df[
        (lithuania_df['Date'] >= estonia_start_date) & 
        (lithuania_df['Date'] <= estonia_end_date)
    ]
    
    return lithuania_filtered

def remove_outliers(df, column, mean, std, n_std=3):
    """Remove outliers based on mean and standard deviation."""
    return df[
        (df[column] >= mean - n_std * std) & 
        (df[column] <= mean + n_std * std)
    ]

def perform_statistical_tests(estonia_df, lithuania_df):
    """Perform statistical tests to compare the datasets."""
    # Wind speed comparison
    estonia_wind = estonia_df['wind_speed'].dropna()
    lithuania_wind = lithuania_df['WindSpeed_Avg'].dropna()
    
    # Get statistics for wind speed
    wind_stats = {
        'Estonia': {
            'mean': np.mean(estonia_wind),
            'std': np.std(estonia_wind, ddof=1),
            'n': len(estonia_wind)
        },
        'Lithuania': {
            'mean': np.mean(lithuania_wind),
            'std': np.std(lithuania_wind, ddof=1),
            'n': len(lithuania_wind)
        }
    }
    
    # Cloud cover comparison
    estonia_cc = estonia_df['cloudcover_total'].dropna()
    lithuania_cc = lithuania_df['Estimated_Cloud_Cover'].dropna()
    
    # Get statistics for cloud cover
    cc_stats = {
        'Estonia': {
            'mean': np.mean(estonia_cc),
            'std': np.std(estonia_cc, ddof=1),
            'n': len(estonia_cc)
        },
        'Lithuania': {
            'mean': np.mean(lithuania_cc),
            'std': np.std(lithuania_cc, ddof=1),
            'n': len(lithuania_cc)
        }
    }
    
    # Print statistics
    print(f"Wind Estonia - N: {wind_stats['Estonia']['n']}, Mean: {wind_stats['Estonia']['mean']:.3f}, Std Dev: {wind_stats['Estonia']['std']:.3f}")
    print(f"Wind Lithuania - N: {wind_stats['Lithuania']['n']}, Mean: {wind_stats['Lithuania']['mean']:.3f}, Std Dev: {wind_stats['Lithuania']['std']:.3f}")
    print(f"Cloud Cover Estonia - N: {cc_stats['Estonia']['n']}, Mean: {cc_stats['Estonia']['mean']:.3f}, Std Dev: {cc_stats['Estonia']['std']:.3f}")
    print(f"Cloud Cover Lithuania - N: {cc_stats['Lithuania']['n']}, Mean: {cc_stats['Lithuania']['mean']:.3f}, Std Dev: {cc_stats['Lithuania']['std']:.3f}")
    
    # Remove outliers
    estonia_df_clean = remove_outliers(estonia_df, 'wind_speed', wind_stats['Estonia']['mean'], wind_stats['Estonia']['std'])
    lithuania_df_clean = remove_outliers(lithuania_df, 'WindSpeed_Avg', wind_stats['Lithuania']['mean'], wind_stats['Lithuania']['std'])
    
    # T-test for wind speed
    t_stat_wind, p_value_wind = ttest_ind(
        estonia_df_clean['wind_speed'].dropna(), 
        lithuania_df_clean['WindSpeed_Avg'].dropna(), 
        equal_var=False
    )
    
    # T-test for cloud cover
    t_stat_cc, p_value_cc = ttest_ind(
        estonia_df['cloudcover_total'].dropna(), 
        lithuania_df['Estimated_Cloud_Cover'].dropna(), 
        equal_var=False
    )
    
    # Print t-test results
    print(f"\nT-statistic wind: {t_stat_wind:.4f}")
    print(f"P-value wind: {p_value_wind}")
    print(f"T-statistic cloud cover: {t_stat_cc}")
    print(f"P-value cloud cover: {p_value_cc}")
    
    return estonia_df_clean, lithuania_df_clean

def plot_solar_radiation(estonia_df, lithuania_df):
    """Plot solar radiation comparison between Estonia and Lithuania."""
    plt.figure(figsize=(10, 6))
    
    # Sort Lithuania data by date for proper plotting
    lithuania_df_sorted = lithuania_df.sort_values(by='Date')
    
    # Plot both datasets
    plt.plot(lithuania_df_sorted['Date'], lithuania_df_sorted['solar_radiation_kWh/m2/day'], 
             label='Solar Radiation (Lithuania)', marker='x', color='g')
    plt.plot(estonia_df['date'], estonia_df['surface_solar_radiation_downwards'], 
             label='Solar Radiation (Estonia)', marker='x', color='b')
    
    # Add labels and styling
    plt.xlabel('Date')
    plt.ylabel('Solar Radiation (kWh/m²/day)')
    plt.title('Solar Radiation Over Time (Estonia vs Lithuania)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOTS_DIR, 'solar_radiation_comparison.png'))
    plt.close()

def plot_wind_speed(estonia_df, lithuania_df):
    """Plot wind speed comparison between Estonia and Lithuania."""
    plt.figure(figsize=(10, 6))
    
    # Sort Lithuania data by date for proper plotting
    lithuania_df_sorted = lithuania_df.sort_values(by='Date')
    
    # Plot both datasets
    plt.plot(lithuania_df_sorted['Date'], lithuania_df_sorted['WindSpeed_Avg'], 
             label='Wind Speed (Lithuania)', marker='o', color='r', alpha=0.7)
    plt.plot(estonia_df['date'], estonia_df['wind_speed'], 
             label='Wind Speed (Estonia)', marker='o', color='b', alpha=0.7)
    
    # Add labels and styling
    plt.xlabel('Date')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed Comparison: Estonia vs Lithuania')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add text with statistical information
    estonia_mean = estonia_df['wind_speed'].mean()
    lithuania_mean = lithuania_df['WindSpeed_Avg'].mean()
    
    plt.axhline(y=estonia_mean, color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=lithuania_mean, color='r', linestyle='--', alpha=0.5)
    
    plt.text(estonia_df['date'].iloc[0], estonia_mean + 0.2, 
             f'Estonia Mean: {estonia_mean:.2f}', color='b')
    plt.text(estonia_df['date'].iloc[0], lithuania_mean - 0.3, 
             f'Lithuania Mean: {lithuania_mean:.2f}', color='r')
    
    # Save the plot
    plt.savefig(os.path.join(PLOTS_DIR, 'wind_speed_comparison.png'))
    plt.close()

def plot_cloud_cover(estonia_df, lithuania_df):
    """Plot cloud cover comparison between Estonia and Lithuania."""
    plt.figure(figsize=(10, 6))
    
    # Sort Lithuania data by date for proper plotting
    lithuania_df_sorted = lithuania_df.sort_values(by='Date')
    
    # Plot both datasets - note that Lithuania's cloud cover is between 0-1, Estonia's might be between 0-100
    # Scale them to be on the same range for comparison
    estonia_cc = estonia_df['cloudcover_total']
    lithuania_cc = lithuania_df_sorted['Estimated_Cloud_Cover']
    
    # Check if Estonia's cloud cover is on 0-100 scale, and convert to 0-1 if needed
    if estonia_cc.mean() > 1:
        estonia_cc = estonia_cc / 100
    
    plt.plot(lithuania_df_sorted['Date'], lithuania_cc, 
             label='Cloud Cover (Lithuania)', marker='o', color='g', alpha=0.7)
    plt.plot(estonia_df['date'], estonia_cc, 
             label='Cloud Cover (Estonia)', marker='o', color='purple', alpha=0.7)
    
    # Add labels and styling
    plt.xlabel('Date')
    plt.ylabel('Cloud Cover (0-1 scale)')
    plt.title('Cloud Cover Comparison: Estonia vs Lithuania')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add text with statistical information
    estonia_mean = estonia_cc.mean()
    lithuania_mean = lithuania_cc.mean()
    
    plt.axhline(y=estonia_mean, color='purple', linestyle='--', alpha=0.5)
    plt.axhline(y=lithuania_mean, color='g', linestyle='--', alpha=0.5)
    
    plt.text(estonia_df['date'].iloc[0], max(estonia_mean, lithuania_mean) + 0.05, 
             f'Estonia Mean: {estonia_mean:.2f}', color='purple')
    plt.text(estonia_df['date'].iloc[0], min(estonia_mean, lithuania_mean) - 0.05, 
             f'Lithuania Mean: {lithuania_mean:.2f}', color='g')
    
    # Save the plot
    plt.savefig(os.path.join(PLOTS_DIR, 'cloud_cover_comparison.png'))
    plt.close()

def create_monthly_comparison_plots(estonia_df, lithuania_df):
    """Create monthly comparison plots for all metrics."""
    # Add month column to Estonia data
    estonia_df['month'] = estonia_df['date'].dt.strftime('%B')
    
    # Group by month
    estonia_monthly = estonia_df.groupby('month').agg({
        'wind_speed': 'mean',
        'cloudcover_total': 'mean',
        'temperature': 'mean',
        'surface_solar_radiation_downwards': 'mean'
    }).reset_index()
    
    lithuania_monthly = lithuania_df.groupby('month').agg({
        'WindSpeed_Avg': 'mean',
        'Estimated_Cloud_Cover': 'mean',
        'Temp_Avg': 'mean',
        'solar_radiation_kWh/m2/day': 'mean'
    }).reset_index()
    
    # Order months chronologically
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    estonia_monthly['month_num'] = estonia_monthly['month'].apply(lambda x: month_order.index(x) if x in month_order else -1)
    lithuania_monthly['month_num'] = lithuania_monthly['month'].apply(lambda x: month_order.index(x) if x in month_order else -1)
    
    estonia_monthly = estonia_monthly.sort_values('month_num')
    lithuania_monthly = lithuania_monthly.sort_values('month_num')
    
    # Plot monthly wind speed separately
    plt.figure(figsize=(10, 6))
    plt.bar(estonia_monthly['month'], estonia_monthly['wind_speed'], alpha=0.6, label='Estonia', color='b')
    plt.bar(lithuania_monthly['month'], lithuania_monthly['WindSpeed_Avg'], alpha=0.6, label='Lithuania', color='r')
    plt.xlabel('Month')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Monthly Average Wind Speed')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_wind_speed.png'))
    plt.close()
    
    # Plot monthly cloud cover separately
    plt.figure(figsize=(10, 6))
    estonia_cc = estonia_monthly['cloudcover_total']
    if estonia_cc.mean() > 1:
        estonia_cc = estonia_cc / 100
    
    plt.bar(estonia_monthly['month'], estonia_cc, alpha=0.6, label='Estonia', color='purple')
    plt.bar(lithuania_monthly['month'], lithuania_monthly['Estimated_Cloud_Cover'], alpha=0.6, label='Lithuania', color='g')
    plt.xlabel('Month')
    plt.ylabel('Cloud Cover (0-1 scale)')
    plt.title('Monthly Average Cloud Cover')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_cloud_cover.png'))
    plt.close()
    
    # Plot monthly temperature separately
    plt.figure(figsize=(10, 6))
    plt.bar(estonia_monthly['month'], estonia_monthly['temperature'], alpha=0.6, label='Estonia', color='orange')
    plt.bar(lithuania_monthly['month'], lithuania_monthly['Temp_Avg'], alpha=0.6, label='Lithuania', color='darkgreen')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.title('Monthly Average Temperature')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_temperature.png'))
    plt.close()
    
    # Plot monthly solar radiation separately
    plt.figure(figsize=(10, 6))
    plt.bar(estonia_monthly['month'], estonia_monthly['surface_solar_radiation_downwards'], alpha=0.6, label='Estonia', color='cyan')
    plt.bar(lithuania_monthly['month'], lithuania_monthly['solar_radiation_kWh/m2/day'], alpha=0.6, label='Lithuania', color='magenta')
    plt.xlabel('Month')
    plt.ylabel('Solar Radiation (kWh/m²/day)')
    plt.title('Monthly Average Solar Radiation')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'monthly_solar_radiation.png'))
    plt.close()

def main():
    print("Starting weather data analysis...")
    
    # Load data
    print("Loading Estonia data...")
    estonia_df = load_estonia_data(FORECAST_WEATHER_PATH)
    
    print("Loading Lithuania data...")
    lithuania_df = load_lithuania_data(LITHUANIA_DATA_PATH)
    
    # Filter Lithuania data to match Estonia's date range
    print("Filtering Lithuania data to match Estonia's date range...")
    lithuania_filtered = filter_data_by_date_range(estonia_df, lithuania_df)
    
    # Perform statistical tests
    print("Performing statistical tests...")
    estonia_clean, lithuania_clean = perform_statistical_tests(estonia_df, lithuania_filtered)
    
    # Create plots directory if it doesn't exist
    print(f"Creating plots in directory: {PLOTS_DIR}")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Plot comparisons and save to files
    print("Creating solar radiation plot...")
    plot_solar_radiation(estonia_df, lithuania_filtered)
    
    print("Creating wind speed plot...")
    plot_wind_speed(estonia_df, lithuania_filtered)
    
    print("Creating cloud cover plot...")
    plot_cloud_cover(estonia_df, lithuania_filtered)
    
    print("Creating monthly comparison plots...")
    create_monthly_comparison_plots(estonia_df, lithuania_filtered)
    
    # Save cleaned data if needed
    print("Saving cleaned Lithuania data...")
    lithuania_df.to_csv('lithuania_weather_data_cleaned.csv', index=False)
    
    print(f"Analysis complete! All plots saved to {PLOTS_DIR} directory.")

if __name__ == "__main__":
    main()