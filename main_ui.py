import os
import shutil
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import ttk, messagebox

# --- Matplotlib and mplfinance Setup ---
# IMPORTANT: set non-interactive backend before importing mplfinance / pyplot
import matplotlib
matplotlib.use("Agg")   # <-- prevents errors when plotting in threads
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import multiprocessing

# =============================================================================
# --- BACKEND LOGIC (Data Processing and Charting) ---
# These are the functions from your original script.
# =============================================================================

GRAPH_ROOT = "graph_stock"

# Create an mplfinance style
binance_dark = mpf.make_mpf_style(
    base_mpl_style="dark_background",
    marketcolors=mpf.make_marketcolors(
        up="#3dc985", down="#ef4f60", edge={"up": "#3dc985", "down": "#ef4f60"},
        wick={"up": "#3dc985", "down": "#ef4f60"}, volume={"up": "#247252", "down": "#82333f"},
    ), facecolor="#1b1f24", gridcolor="#2c2e31", gridstyle="--", y_on_right=True,
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

def read_csv_and_get_symbols(file_path, limit_from_top):
    df = pd.read_csv(file_path)
    if 'Symbol' not in df.columns: raise KeyError("CSV must have a 'Symbol' column.")
    syms = df['Symbol'].astype(str).str.strip().tolist()
    seen, cleaned = set(), []
    for s in syms:
        s_up = s.upper()
        if s_up not in seen:
            cleaned.append(s_up)
            seen.add(s_up)
    return cleaned[:limit_from_top]

def fetch_all_stock_data(symbols_with_suffix, start_date, interval, app_instance, description):
    app_instance.update_status(f"Fetching {description}...")
    raw = yf.download(symbols_with_suffix, start=start_date, interval=interval, group_by="ticker", threads=True, auto_adjust=True)
    return raw

def get_single_symbol_df_from_raw(raw_data, symbol_with_suffix):
    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            df = raw_data[symbol_with_suffix].copy()
        else:
            df = raw_data.copy()
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols): return None
        
        df = df[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
        df.index = pd.to_datetime(df.index)
        return df if not df.empty else None
    except (KeyError, AttributeError):
        return None

def calculate_return(df):
    if len(df) < 2: return 0.0
    start_price = float(df['Close'].iloc[0])
    end_price = float(df['Close'].iloc[-1])
    return 0.0 if start_price == 0 else ((end_price - start_price) / start_price) * 100

def save_candlestick_chart(symbol, df, rank, return_percent, out_folder):
    fname = os.path.join(out_folder, f"{rank:03d}_{symbol}.png")
    title = f"{symbol} - Return (from analysis period): {return_percent:.2f}%"
    mpf.plot(df, type='candle', style=binance_dark, title=title, ylabel='Price',
             savefig=dict(fname=fname, dpi=300, bbox_inches='tight'),
             figratio=(20, 9), figscale=0.8)
    return True, fname

# This is the main backend worker function, adapted from your `main`
def run_backend_processing(params, app_instance):
    try:
        # --- 1. Setup based on GUI parameters ---
        app_instance.update_status("Step 1/7: Initializing analysis...")
        country = params['country']
        suffix = "" if country == "us" else ".NS"
        csv_file = os.path.join("csv", f"{country}.csv")

        # Calculate start dates
        now = datetime.now()
        if params['analysis_unit'] == 'weeks':
            analysis_start_date = now - timedelta(weeks=params['analysis_duration'])
        elif params['analysis_unit'] == 'months':
            analysis_start_date = now - timedelta(days=params['analysis_duration'] * 30.44)
        else: # days
            analysis_start_date = now - timedelta(days=params['analysis_duration'])

        if params['chart_unit'] == 'weeks':
            chart_start_date = now - timedelta(weeks=params['chart_duration'])
        elif params['chart_unit'] == 'months':
            chart_start_date = now - timedelta(days=params['chart_duration'] * 30.44)
        else: # days
            chart_start_date = now - timedelta(days=params['chart_duration'])

        # --- 2. Read symbols ---
        app_instance.update_status("Step 2/7: Reading stock symbols...")
        symbols = read_csv_and_get_symbols(csv_file, params['stock_limit'])
        if not symbols: raise ValueError("No symbols found in CSV file.")
        symbols_with_suffix = [f"{s}{suffix}" for s in symbols]

        # --- 3. Fetch Data (Two separate downloads) ---
        raw_analysis_data = fetch_all_stock_data(symbols_with_suffix, analysis_start_date, '1d', app_instance, "long-term analysis data")
        if raw_analysis_data.empty: raise ValueError("Failed to download analysis data.")

        raw_chart_data = fetch_all_stock_data(symbols_with_suffix, chart_start_date, params['interval'], app_instance, "short-term chart data")
        if raw_chart_data.empty: raise ValueError("Failed to download chart data.")

        # --- 4. Process and combine data ---
        app_instance.update_status("Step 4/7: Processing and combining data...")
        results, failed_symbols = [], set()
        for sym, sym_suffix in zip(symbols, symbols_with_suffix):
            df_analysis = get_single_symbol_df_from_raw(raw_analysis_data, sym_suffix)
            if df_analysis is None:
                failed_symbols.add(sym)
                continue
            ret = calculate_return(df_analysis)

            df_chart = get_single_symbol_df_from_raw(raw_chart_data, sym_suffix)
            if df_chart is None:
                failed_symbols.add(sym)
                continue
            
            results.append((sym, ret, df_chart))
        
        if not results: raise ValueError("Could not process data for any symbol.")
        results.sort(key=lambda x: x[1], reverse=True)

        # --- 5. Prepare output folder ---
        app_instance.update_status("Step 5/7: Preparing output folder...")
        folder_name = f"Analysis_{params['analysis_duration']}{params['analysis_unit']}_{params['interval']}"
        out_folder = os.path.join(GRAPH_ROOT, folder_name)
        if os.path.exists(out_folder): shutil.rmtree(out_folder)
        os.makedirs(out_folder, exist_ok=True)
        app_instance.set_output_folder(out_folder)

        # --- 6. Create charts in parallel ---
        app_instance.update_status(f"Step 6/7: Creating {len(results)} charts in parallel...")
        cpu = multiprocessing.cpu_count() or 4
        with ThreadPoolExecutor(max_workers=min(32, cpu + 4)) as ex:
            list(ex.map(lambda p: save_candlestick_chart(*p), 
                [(sym, df, r, ret, out_folder) for r, (sym, ret, df) in enumerate(results, 1)]))

        # --- 7. Finalize ---
        final_message = f"Success! Saved {len(results)} charts."
        if failed_symbols:
            final_message += f" Failed on {len(failed_symbols)} symbols."
        app_instance.update_status(final_message, is_complete=True)

    except Exception as e:
        app_instance.update_status(f"Error: {e}", is_error=True)


# =============================================================================
# --- FRONTEND LOGIC (The GUI Application using Tkinter) ---
# =============================================================================

class StockAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Performance Analyzer")
        self.root.geometry("450x500")
        self.output_folder = None

        style = ttk.Style(self.root)
        style.theme_use('clam')

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill="both", expand=True)

        # --- Widgets for User Input ---
        self.create_widget(main_frame, "Select Country:", 0)
        self.country_var = tk.StringVar(value="us")
        ttk.Radiobutton(main_frame, text="US", variable=self.country_var, value="us").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Radiobutton(main_frame, text="India", variable=self.country_var, value="india").grid(row=0, column=2, sticky="w")
        
        self.stock_limit_entry = self.create_widget(main_frame, "Number of Stocks:", 1, ttk.Entry, width=10)
        self.stock_limit_entry.insert(0, "100")

        # Analysis Duration
        self.analysis_duration_entry = self.create_widget(main_frame, "Analysis Duration:", 2, ttk.Entry, width=10)
        self.analysis_duration_entry.insert(0, "12")
        self.analysis_unit_combo = ttk.Combobox(main_frame, values=["days", "weeks", "months"], width=10, state="readonly")
        self.analysis_unit_combo.grid(row=2, column=2, sticky="w")
        self.analysis_unit_combo.set("months")
        
        # Chart Duration
        self.chart_duration_entry = self.create_widget(main_frame, "Chart Duration:", 3, ttk.Entry, width=10)
        self.chart_duration_entry.insert(0, "7")
        self.chart_unit_combo = ttk.Combobox(main_frame, values=["days", "weeks", "months"], width=10, state="readonly")
        self.chart_unit_combo.grid(row=3, column=2, sticky="w")
        self.chart_unit_combo.set("days")
        
        # Interval
        intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo']
        self.interval_combo = self.create_widget(main_frame, "Chart Interval:", 4, ttk.Combobox, values=intervals, width=10, state="readonly")
        self.interval_combo.set("1d")
        
        # Start Button
        self.start_button = ttk.Button(main_frame, text="Start Analysis", command=self.start_analysis_thread)
        self.start_button.grid(row=5, column=0, columnspan=3, pady=25)

        # Status Area
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=6, column=0, columnspan=3, sticky="ew")
        self.status_label = ttk.Label(status_frame, text="Ready...", wraplength=400)
        self.status_label.pack(fill="x", expand=True)

        self.open_folder_button = ttk.Button(status_frame, text="Open Output Folder", state="disabled", command=self.open_output_folder)
        self.open_folder_button.pack(pady=10)

    def create_widget(self, parent, label_text, row, widget_class=None, **kwargs):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=5)
        if widget_class:
            widget = widget_class(parent, **kwargs)
            widget.grid(row=row, column=1, sticky="w", padx=5)
            return widget

    def update_status(self, message, is_complete=False, is_error=False):
        # Use root.after to safely update GUI from other threads
        def do_update():
            self.status_label.config(text=message)
            if is_complete or is_error:
                self.start_button.config(state="normal") # Re-enable button
                if is_complete and self.output_folder:
                    self.open_folder_button.config(state="normal")
        self.root.after(0, do_update)

    def set_output_folder(self, folder_path):
        self.output_folder = folder_path

    def open_output_folder(self):
        if self.output_folder and os.path.exists(self.output_folder):
            if os.name == 'nt': # Windows
                os.startfile(self.output_folder)
            elif os.name == 'posix': # MacOS/Linux
                if 'darwin' in os.uname().sysname.lower(): # MacOS
                    os.system(f'open "{self.output_folder}"')
                else: # Linux
                    os.system(f'xdg-open "{self.output_folder}"')
        else:
            messagebox.showerror("Error", "Output folder not found.")

    def start_analysis_thread(self):
        try:
            params = {
                "country": self.country_var.get(),
                "stock_limit": int(self.stock_limit_entry.get()),
                "analysis_duration": int(self.analysis_duration_entry.get()),
                "analysis_unit": self.analysis_unit_combo.get(),
                "chart_duration": int(self.chart_duration_entry.get()),
                "chart_unit": self.chart_unit_combo.get(),
                "interval": self.interval_combo.get()
            }
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for stock limit and durations.")
            return

        # Disable button to prevent multiple runs
        self.start_button.config(state="disabled")
        self.open_folder_button.config(state="disabled")
        self.output_folder = None
        
        # Run the backend processing in a separate thread
        analysis_thread = threading.Thread(
            target=run_backend_processing,
            args=(params, self)
        )
        analysis_thread.daemon = True # Allows main window to close even if thread is running
        analysis_thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalyzerApp(root)
    root.mainloop()