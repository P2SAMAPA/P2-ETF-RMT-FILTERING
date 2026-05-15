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
        if returns.empty or len(returns) < config.ROLLING_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        result = rmt_filtered_centrality(returns, config.ROLLING_WINDOW)
        if result is None:
            print("  RMT filtering failed")
            all_results[universe_name] = {"top_etfs": []}
            continue

        centrality = result["centrality"]
        n_signal = result["n_signal"]
        # Sort by centrality descending
        sorted_items = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, score in sorted_items:
            full_scores[ticker] = score
        for i, (ticker, score) in enumerate(sorted_items[:config.TOP_N]):
            top_etfs.append({"ticker": ticker, "centrality": score})
        print(f"  Top 3 ETFs by filtered eigenvector centrality: {[e['ticker'] for e in top_etfs]}")
        print(f"  Number of signal eigenvalues (T={config.ROLLING_WINDOW}): {n_signal}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "n_signal": n_signal,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/rmt_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Random Matrix Theory Filtering Engine complete ===")

if __name__ == "__main__":
    main()
