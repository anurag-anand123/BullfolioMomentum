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

def fetch_all_stock_data(symbols, start_date, interval, description="data"):
    """Download all tickers in one call for a specific purpose."""
    tickers = [f"{s}{suffix}" for s in symbols]
    print(f"\nFetching {description} for {len(tickers)} tickers from {start_date.strftime('%Y-%m-%d')} with interval '{interval}'...")
    try:
        raw = yf.download(tickers, start=start_date, interval=interval, group_by="ticker", threads=True, auto_adjust=True)
        return raw
    except Exception as e:
        print(f"Error fetching {description}: {e}")
        return None

def get_single_symbol_df_from_raw(raw_data, symbol_with_suffix):
    """Extract symbol dataframe from yfinance multi-ticker raw download safely."""
    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            df = raw_data[symbol_with_suffix].copy()
        else: # Handle single ticker download case
            df = raw_data.copy()

        # Check if essential columns exist before proceeding
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return None # Not a valid OHLCV dataframe

        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        df.index = pd.to_datetime(df.index)

        if df.empty:
            return None
        return df
    except (KeyError, AttributeError):
        return None # Symbol not found in raw data or other errors

def calculate_return(df):
    try:
        if len(df) < 2:
            return 0.0
        start_price = float(df['Close'].iloc[0])
        end_price = float(df['Close'].iloc[-1])
        if start_price == 0: return 0.0
        return ((end_price - start_price) / start_price) * 100
    except Exception:
        return None

def save_candlestick_chart(symbol, df, rank, return_percent, out_folder):
    """Plot and save chart using Agg backend (safe in threads)."""
    try:
        fname = os.path.join(out_folder, f"{rank:03d}_{symbol}.png")
        title = f"{symbol} - Return (from analysis period): {return_percent:.2f}%"
        mpf.plot(
            df,
            type='candle',
            style=binance_dark,
            title=title,
            ylabel='Price',
            savefig=dict(fname=fname, dpi=300, bbox_inches='tight'),
            figratio=(20, 9),
            figscale=0.8,
        )
        return True, fname
    except Exception as e:
        return False, str(e)

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
        start_date_dt = datetime.now() - timedelta(days=duration * 30.44) # More accurate average
    else: # days
        start_date_dt = datetime.now() - timedelta(days=duration)

    return duration, duration_type, start_date_dt

def main():
    global suffix, csv_file, GRAPH_ROOT

    VALID_LOOKBACK_PERIODS = {
        '1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60, '1h': 730,
    }

    country = input("Do you want to analyze stocks from 'US(us)' or 'India(india)'? ").strip().lower()
    if country not in ['us', 'india']: return

    suffix = "" if country == "us" else ".NS"
    csv_file = os.path.join("csv", "us.csv" if country == "us" else "india.csv")

    try:
        stock_limit = int(input("Enter number of stocks from top (by mCap): "))
    except ValueError:
        print("Invalid number. Exiting."); return

    analysis_duration, analysis_duration_type, analysis_start_date_dt = get_duration_details("\n--- Enter Analysis Duration (for % return calculation) ---")
    if not analysis_duration: return

    chart_duration, chart_duration_type, chart_start_date_dt = get_duration_details("\n--- Enter Chart Duration (for the plot visualization) ---")
    if not chart_duration: return

    chart_interval = input("Enter CHART interval (e.g. '1m','5m','1h','4h','1d'): ").strip()

    # --- MODIFIED: Validate the CHART request against yfinance limitations ---
    max_lookback_days = VALID_LOOKBACK_PERIODS.get(chart_interval)
    if max_lookback_days:
        earliest_allowed_date = datetime.now() - timedelta(days=max_lookback_days)
        if chart_start_date_dt < earliest_allowed_date:
            print("\n" + "="*80)
            print(f"⚠️  WARNING: INVALID DATE RANGE FOR '{chart_interval}' CHART INTERVAL")
            print(f"yfinance can only provide '{chart_interval}' data for the last {max_lookback_days} days.")
            print(f"Your requested chart duration is too long for this interval.")
            print("Please choose a shorter Chart duration, or a longer interval (like '1h' or '1d').")
            print("="*80)
            return

    symbols = read_csv_and_get_symbols(csv_file, stock_limit)
    if not symbols: print("No symbols found. Exiting."); return

    # --- NEW: Perform two separate, parallel data downloads ---
    # 1. Long-term DAILY data for analysis
    raw_analysis_data = fetch_all_stock_data(symbols, analysis_start_date_dt, '1d', description="long-term analysis data (daily)")
    if raw_analysis_data is None or raw_analysis_data.empty:
        print("Failed to download analysis data. Exiting."); return

    # 2. Short-term data for charting with user-specified interval
    raw_chart_data = fetch_all_stock_data(symbols, chart_start_date_dt, chart_interval, description=f"short-term chart data ({chart_interval})")
    if raw_chart_data is None or raw_chart_data.empty:
        print("Failed to download chart data. Exiting."); return

    # --- NEW: Process and combine data from the two sources ---
    results = []
    failed_symbols = set()
    print("\nProcessing and combining data for each symbol...")
    for sym in symbols:
        sym_with_suffix = f"{sym}{suffix}"

        # Get data for analysis and calculate return
        df_analysis = get_single_symbol_df_from_raw(raw_analysis_data, sym_with_suffix)
        if df_analysis is None:
            failed_symbols.add(sym)
            continue
        ret = calculate_return(df_analysis)
        if ret is None:
            failed_symbols.add(sym)
            continue

        # Get data for charting
        df_chart = get_single_symbol_df_from_raw(raw_chart_data, sym_with_suffix)
        if df_chart is None:
            failed_symbols.add(sym)
            continue

        results.append((sym, ret, df_chart))

    if not results:
        print("\nCould not process any data for the given symbols. Check symbol names and data availability.")
        return

    results.sort(key=lambda x: x[1], reverse=True)

    out_folder = os.path.join(GRAPH_ROOT, f"Analysis_{analysis_duration}{analysis_duration_type}_{chart_interval}")
    if os.path.exists(out_folder): shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    cpu = multiprocessing.cpu_count() or 4
    max_workers = min(32, cpu + 4)
    print(f"\nCreating {len(results)} charts in parallel using max_workers={max_workers}...")

    failed_charts = []
    created_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_sym = {
            ex.submit(save_candlestick_chart, sym, df, rank, ret, out_folder): (sym, rank)
            for rank, (sym, ret, df) in enumerate(results, start=1)
        }
        for fut in as_completed(future_to_sym):
            sym, rank = future_to_sym[fut]
            ok, info = fut.result()
            if not ok: failed_charts.append((sym, info))
            else: created_files.append(info)

    print("\nTop 20 results (based on analysis period):")
    for rank, (sym, ret, _) in enumerate(results[:20], start=1):
        print(f"{rank}. {sym} - {ret:.2f}%")

    if failed_symbols:
        print("\nFailed to get complete data for these symbols:")
        print(sorted(list(failed_symbols)))

    if failed_charts:
        print("\nFailed to create charts for these symbols:")
        for sym, err in failed_charts: print(f"{sym} -> {err}")

    print(f"\nSaved {len(created_files)} chart files to: {out_folder}")

    try:
        if os.name == 'nt': os.startfile(out_folder)
        elif os.name == 'posix': os.system(f'open "{out_folder}"' if 'darwin' in os.uname().sysname.lower() else f'xdg-open "{out_folder}"')
    except Exception as e:
        print("Could not open folder automatically:", e)

if __name__ == "__main__":
    main()