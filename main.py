import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import os
import shutil

# Constants
GRAPH_FOLDER = 'graph_custom'

suffix = ""
csv_file = ""

# Binance Dark Theme
binance_dark = {
    "base_mpl_style": "dark_background",
    "marketcolors": {
        "candle": {"up": "#3dc985", "down": "#ef4f60"},
        "edge": {"up": "#3dc985", "down": "#ef4f60"},
        "wick": {"up": "#3dc985", "down": "#ef4f60"},
        "ohlc": {"up": "green", "down": "red"},
        "volume": {"up": "#247252", "down": "#82333f"},
        "vcedge": {"up": "green", "down": "red"},
        "vcdopcod": False,
        "alpha": 1,
    },
    "mavcolors": ("#ad7739", "#a63ab2", "#62b8ba"),
    "facecolor": "#1b1f24",
    "gridcolor": "#2c2e31",
    "gridstyle": "--",
    "y_on_right": True,
    "rc": {
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.edgecolor": "#474d56",
        "axes.titlecolor": "red",
        "figure.facecolor": "#161a1e",
        "figure.titlesize": 10,  # Reduced title size
        "figure.titleweight": "semibold",
        "axes.labelsize": 5,  # Reduced label size
        "axes.titlesize": 8,  # Reduced axes title size
        "xtick.labelsize": 5,  # Reduced x-axis tick label size
        "ytick.labelsize": 5,  # Reduced y-axis tick label size
    },
    "base_mpf_style": "binance-dark",
}

def read_csv_and_get_symbols(file_path, limit_from_top):
    """Read CSV and extract stock symbols."""
    try:
        df = pd.read_csv(file_path)
        if 'Symbol' not in df.columns:
            raise KeyError("The CSV file must contain a 'Symbol' column.")
        return df['Symbol'].head(limit_from_top).tolist()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except KeyError as e:
        print(e)
        return []

def fetch_stock_data(symbol, start_date, interval):
    """Fetch historical stock data for a given symbol."""
    try:
        symbol_with_suffix = f"{symbol}{suffix}"
        data = yf.download(symbol_with_suffix, start=start_date, interval=interval)
        if data.empty or len(data) < 2:
            print(f"Insufficient data for {symbol}.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_return(data):
    """Calculate stock return for the given data."""
    try:
        # Extract scalar values using .iloc[0].item()
        start_price = data['Close'].iloc[0].item()
        end_price = data['Close'].iloc[-1].item()
        return ((end_price - start_price) / start_price) * 100
    except Exception as e:
        print(f"Error calculating return: {e}")
        return None

def clean_and_prepare_data(data, symbol):
    """Clean and prepare the data for mplfinance."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.map('_'.join).str.strip()

        column_mapping = {
            f"Close_{symbol}{suffix}": 'Close',
            f'High_{symbol}{suffix}': 'High',
            f'Low_{symbol}{suffix}': 'Low',
            f'Open_{symbol}{suffix}': 'Open',
            f'Volume_{symbol}{suffix}': 'Volume',
        }
        data = data.rename(columns=column_mapping)

        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            raise KeyError(f"Required columns {required_columns} not found in data.")

        data = data[required_columns].apply(pd.to_numeric, errors='coerce').dropna()
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        print(f"Error cleaning and preparing data: {e}")
        return None

def save_candlestick_chart(data, symbol, rank, return_percent):
    """Save the candlestick chart."""
    try:
        data = clean_and_prepare_data(data, symbol)
        if data is None or data.empty:
            print(f"Insufficient or invalid data for {symbol}.")
            return

        file_name = os.path.join(GRAPH_FOLDER, f"{rank}.png")
        title = f"{symbol} - Return: {return_percent:.2f}%"
        mpf.plot(
            data,
            type='candle',
            style=binance_dark,
            title=title,
            ylabel='Price',
            savefig=dict(fname=file_name, dpi=300, bbox_inches='tight'),
            figratio=(20, 9),
            figscale=0.8,
        )
        print(f"Candlestick chart saved for {symbol} as {file_name}.")
    except Exception as e:
        print(f"Error saving candlestick chart for {symbol}: {e}")

def main():
    global suffix, csv_file, GRAPH_FOLDER
    """Main function to execute the script."""
    try:
        # Ask user to select the country
        country = input("Do you want to analyze stocks from 'US(us)' or 'India(india)'? ").strip().lower()
        if country not in ['us', 'india']:
            print("Invalid choice. Please enter either 'US' or 'India'.")
            return

        # Set the exchange based on the selected country
        if country == 'us':
            suffix = ""  # US stocks don't need a suffix for yfinance
            csv_file = 'csv/us.csv'  # Replace with your US stock list CSV
        elif country == 'india':
            suffix = ".NS"
            csv_file = 'csv/india.csv'

        try:
            stock_limit = int(input("Enter the number of stocks you want to analyze from the top(by mCap): Top "))
        except ValueError:
            print("Invalid number. Please enter a valid integer.")
            return

        duration_type = input("Do you want to enter the duration in 'days', 'weeks' or 'months'? ").strip().lower()
        if duration_type not in ['weeks', 'months', 'days']:
            print("Invalid choice. Please enter either 'days', 'weeks' or 'months'.")
            return

        try:
            duration = int(input(f"Enter the number of {duration_type} for historical data: "))
        except ValueError:
            print("Invalid number. Please enter a valid integer.")
            return
        
        print("Choose a candle interval:")
        print("  '1m'  - 1-minute candles")
        print("  '5m'  - 5-minutes candles")
        print("  '1h'  - 1-hour candles")
        print("  '4h'  - 5-hours candles")
        print("  '1d'  - Daily candles")
        print("  '1wk' - Weekly candles")
        print("  '1mo' - Monthly candles")
        print("Note: These are the most commonly used intervals.")
        interval = input("Enter the candle interval (1m / 5m / 1h / 4h / 1d / 1wk / 1mo): ").strip()
        if interval not in ['1m', '5m', '15m', '30m', '1h', '2h', '3h', '4h', '1d', '2d', '5d', '1wk', '2wk', '1mo', '3mo']:
            print("Invalid choice. Please enter among '1m', '5m', '15m', '30m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo'")
            return

        if duration_type == 'weeks':
            start_date = (datetime.now() - timedelta(weeks=duration)).strftime('%Y-%m-%d')
        elif duration_type == 'months':
            start_date = (datetime.now() - timedelta(days=duration * 30)).strftime('%Y-%m-%d')
        elif duration_type == 'days':
            start_date = (datetime.now() - timedelta(days=duration)).strftime('%Y-%m-%d')

        print(f"Fetching data from {start_date} with interval '{interval}'.")
    except ValueError:
        print("Invalid input. Please enter valid numbers and interval.")
        return

    GRAPH_FOLDER = os.path.join("graph_stock", f"{duration}{duration_type}{interval}")
    if os.path.exists(GRAPH_FOLDER):
        shutil.rmtree(GRAPH_FOLDER)
    os.makedirs(GRAPH_FOLDER)

    symbols = read_csv_and_get_symbols(csv_file, stock_limit)
    if not symbols:
        return

    results = []
    for symbol in symbols:
        print(f"Processing {symbol}...")
        data = fetch_stock_data(symbol, start_date, interval)
        if data is not None:
            stock_return = calculate_return(data)
            if stock_return is not None:
                results.append((symbol, stock_return, data))

    results.sort(key=lambda x: x[1], reverse=True)

    for rank, (symbol, stock_return, data) in enumerate(results, start=1):
        save_candlestick_chart(data, symbol, rank, stock_return)
        print(f"{rank}. {symbol}: {stock_return:.2f}% return")

    try:
        if os.name == 'nt':
            os.startfile(GRAPH_FOLDER)
        elif os.name == 'posix':
            os.system(f'open "{GRAPH_FOLDER}"' if 'darwin' in os.uname().sysname.lower() else f'xdg-open "{GRAPH_FOLDER}"')
    except Exception as e:
        print(f"Error opening folder: {e}")

if __name__ == "__main__":
    main()
