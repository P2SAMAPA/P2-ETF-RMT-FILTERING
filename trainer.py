import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from rmt_filter import rmt_filtered_centrality

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (RMT Filtering) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Store best centrality per ETF across windows
        best_per_etf = {}  # ticker -> (best_centrality, best_window)
        window_results = {}  # win -> result dict

        for win in config.WINDOWS:
            if len(returns) < win:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            result = rmt_filtered_centrality(returns, win)
            if result is None:
                continue
            window_results[win] = result
            centrality = result["centrality"]
            n_signal = result["n_signal"]
            print(f"  Window {win}d: {n_signal} signal eigenvalues")
            # Update best per ETF
            for ticker, cent in centrality.items():
                if ticker not in best_per_etf or cent > best_per_etf[ticker][0]:
                    best_per_etf[ticker] = (cent, win)

        if not best_per_etf:
            print("  No valid windows")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Sort ETFs by best centrality (descending)
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (cent, win) in sorted_etfs[:config.TOP_N]:
            top_etfs.append({
                "ticker": ticker,
                "centrality": float(cent),
                "best_window": win
            })
            full_scores[ticker] = {
                "best_centrality": float(cent),
                "best_window": win
            }
        print(f"  Top 3 ETFs by best centrality across windows: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": {str(win): {"n_signal": r["n_signal"]} for win, r in window_results.items()},
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/rmt_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Random Matrix Theory Filtering Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
