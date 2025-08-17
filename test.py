import os
import shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# IMPORTANT: set non-interactive backend before importing mplfinance / pyplot
import matplotlib
matplotlib.use("Agg")   # <-- prevents MacOSX NSWindow errors when plotting in threads

import pandas as pd
import yfinance as yf
import mplfinance as mpf
import multiprocessing

# ----------------- Configuration -----------------
GRAPH_ROOT = "graph_stock"
suffix = ""
csv_file = ""

# Create an mplfinance style with mpf.make_mpf_style
binance_dark = mpf.make_mpf_style(
    base_mpl_style="dark_background",
    marketcolors=mpf.make_marketcolors(
        up="#3dc985",
        down="#ef4f60",
        edge={"up": "#3dc985", "down": "#ef4f60"},
        wick={"up": "#3dc985", "down": "#ef4f60"},
        volume={"up": "#247252", "down": "#82333f"},
    ),
    facecolor="#1b1f24",
    gridcolor="#2c2e31",
    gridstyle="--",
    y_on_right=True,
    rc={
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.edgecolor": "#474d56",
        "axes.titlecolor": "red",
        "figure.facecolor": "#161a1e",
        "figure.titlesize": 10,
        "axes.labelsize": 5,
        "axes.titlesize": 8,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
    }
)
# -------------------------------------------------

def read_csv_and_get_symbols(file_path, limit_from_top):
    try:
        df = pd.read_csv(file_path)
        if 'Symbol' not in df.columns:
            raise KeyError("CSV must have a 'Symbol' column.")
        syms = df['Symbol'].astype(str).str.strip().tolist()
        # Basic cleaning: uppercase, remove duplicates while preserving order
        seen = set()
        cleaned = []
        for s in syms:
            s_up = s.upper()
            if s_up not in seen:
                cleaned.append(s_up)
                seen.add(s_up)
        return cleaned[:limit_from_top]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def fetch_all_stock_data(symbols, start_date, interval, auto_adjust=True):
    """Download all tickers in one call. Return raw yfinance object and a list of failed symbols."""
    tickers = [f"{s}{suffix}" for s in symbols]
    try:
        # Explicitly set auto_adjust to avoid yfinance message. threads=True to speed up.
        raw = yf.download(tickers, start=start_date, interval=interval, group_by="ticker", threads=True, auto_adjust=auto_adjust)
        return raw
    except Exception as e:
        print(f"Error fetching data for multiple tickers: {e}")
        return None

def get_single_symbol_df_from_raw(raw_data, symbol_with_suffix):
    """Extract symbol dataframe from yfinance multi-ticker raw download safely."""
    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            df = raw_data[symbol_with_suffix].copy()
        else:
            df = raw_data.copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        df.index = pd.to_datetime(df.index)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def calculate_return(df):
    try:
        # Ensure there are at least two data points to calculate a return
        if len(df) < 2:
            return 0.0
        start_price = float(df['Close'].iloc[0])
        end_price = float(df['Close'].iloc[-1])
        return ((end_price - start_price) / start_price) * 100
    except Exception:
        return None

def save_candlestick_chart(symbol, df, rank, return_percent, out_folder):
    """Plot and save chart using Agg backend (safe in threads)."""
    try:
        fname = os.path.join(out_folder, f"{rank:03d}_{symbol}.png")
        title = f"{symbol} - Return: {return_percent:.2f}%"
        mpf.plot(
            df,
            type='candle',
            style=binance_dark,
            title=title,
            ylabel='Price',
            savefig=dict(fname=fname, dpi=300, bbox_inches='tight'),
            figratio=(20, 9),
            figscale=0.8,
            #volume=True, # <-- Added volume to the plot
        )
        return True, fname
    except Exception as e:
        return False, str(e)

# <-- NEW: Helper function to get duration details from user
def get_duration_details(prompt_message):
    print(prompt_message)
    duration_type = input("Duration type ('days', 'weeks', 'months'): ").strip().lower()
    if duration_type not in ['days', 'weeks', 'months']:
        print("Invalid duration type. Exiting.")
        return None, None, None
    try:
        duration = int(input(f"Enter the number of {duration_type}: "))
    except ValueError:
        print("Invalid duration. Exiting.")
        return None, None, None
    
    if duration_type == 'weeks':
        start_date_dt = datetime.now() - timedelta(weeks=duration)
    elif duration_type == 'months':
        start_date_dt = datetime.now() - timedelta(days=duration * 30)
    else: # days
        start_date_dt = datetime.now() - timedelta(days=duration)
        
    return duration, duration_type, start_date_dt

def main():
    global suffix, csv_file, GRAPH_ROOT

    country = input("Do you want to analyze stocks from 'US(us)' or 'India(india)'? ").strip().lower()
    if country not in ['us', 'india']:
        print("Invalid choice. Exiting.")
        return

    suffix = "" if country == "us" else ".NS"
    csv_file = os.path.join("csv", "us.csv" if country == "us" else "india.csv")

    try:
        stock_limit = int(input("Enter number of stocks from top (by mCap): "))
    except ValueError:
        print("Invalid number. Exiting.")
        return

    # --- Get Analysis Duration ---
    # <-- MODIFIED: Get duration for % calculation and sorting
    analysis_duration, analysis_duration_type, analysis_start_date_dt = get_duration_details(
        "\n--- Enter Analysis Duration (for % return calculation) ---"
    )
    if not analysis_duration: return

    # --- Get Chart Duration ---
    # <-- NEW: Get duration for the chart plot itself
    chart_duration, chart_duration_type, chart_start_date_dt = get_duration_details(
        "\n--- Enter Chart Duration (for the plot visualization) ---"
    )
    if not chart_duration: return

    interval = input("Enter interval (e.g. '1m','5m','1h','4h','1d','1wk','1mo'): ").strip()
    allowed_intervals = ['1m','5m','15m','30m','1h','2h','3h','4h','1d','2d','5d','1wk','2wk','1mo','3mo']
    if interval not in allowed_intervals:
        print("Invalid interval. Exiting.")
        return
    
    # <-- MODIFIED: Determine the earliest start date needed to fetch all required data
    fetch_start_date = min(analysis_start_date_dt, chart_start_date_dt)
    fetch_start_date_str = fetch_start_date.strftime('%Y-%m-%d')
    
    # <-- MODIFIED: Output folder name is based on the analysis period
    out_folder = os.path.join(GRAPH_ROOT, f"Analysis_{analysis_duration}{analysis_duration_type}_{interval}")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    symbols = read_csv_and_get_symbols(csv_file, stock_limit)
    if not symbols:
        print("No symbols found. Exiting.")
        return

    print(f"\nFetching data from {fetch_start_date_str} with interval '{interval}' (auto_adjust=True).")
    raw_data = fetch_all_stock_data(symbols, fetch_start_date_str, interval, auto_adjust=True)
    if raw_data is None or raw_data.empty:
        print("No data returned by yfinance. Exiting.")
        return
    
    results = []
    failed_downloads = []
    print("Calculating returns and preparing chart data...")
    for sym in symbols:
        sym_with_suffix = f"{sym}{suffix}"
        full_df = get_single_symbol_df_from_raw(raw_data, sym_with_suffix)
        if full_df is None or full_df.empty:
            failed_downloads.append(sym_with_suffix)
            continue
        
        # <-- MODIFIED: Slice the DataFrame for analysis and for charting
        df_analysis = full_df[full_df.index >= analysis_start_date_dt]
        df_chart = full_df[full_df.index >= chart_start_date_dt]

        if df_analysis.empty or df_chart.empty:
            failed_downloads.append(sym_with_suffix)
            continue

        ret = calculate_return(df_analysis) # <-- Return is calculated on the analysis slice
        if ret is None:
            failed_downloads.append(sym_with_suffix)
            continue
        
        # <-- MODIFIED: Store the chart-specific dataframe in the results
        results.append((sym, ret, df_chart))

    results.sort(key=lambda x: x[1], reverse=True)

    cpu = multiprocessing.cpu_count() or 4
    max_workers = min(32, cpu + 4)

    print(f"Creating charts in parallel using max_workers={max_workers} ...")
    failed_charts = []
    created_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # <-- The rest of this section is unchanged, it correctly uses the df_chart stored in results
        future_to_sym = {
            ex.submit(save_candlestick_chart, sym, df, rank, ret, out_folder): (sym, rank)
            for rank, (sym, ret, df) in enumerate(results, start=1)
        }
        for fut in as_completed(future_to_sym):
            sym, rank = future_to_sym[fut]
            ok, info = fut.result()
            if not ok:
                failed_charts.append((sym, info))
            else:
                created_files.append(info)
    
    print("\nTop results (based on analysis period):")
    for rank, (sym, ret, _) in enumerate(results[:20], start=1): # Show top 20
        print(f"{rank}. {sym} - {ret:.2f}%")

    if failed_downloads:
        print("\nFailed downloads (no data from yfinance):")
        print(failed_downloads)

    if failed_charts:
        print("\nFailed to create chart for these symbols:")
        for sym, err in failed_charts:
            print(f"{sym} -> {err}")

    print(f"\nSaved {len(created_files)} chart files to: {out_folder}")

    try:
        if os.name == 'nt':
            os.startfile(out_folder)
        elif os.name == 'posix':
            uname = os.uname().sysname.lower()
            if 'darwin' in uname:
                os.system(f'open "{out_folder}"')
            else:
                os.system(f'xdg-open "{out_folder}"')
    except Exception as e:
        print("Could not open folder automatically:", e)

if __name__ == "__main__":
    main()