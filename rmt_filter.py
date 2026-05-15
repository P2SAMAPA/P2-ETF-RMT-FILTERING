import numpy as np
import pandas as pd

def marchenko_pastur_lambda_plus(n, T):
    """Upper bound of Marchenko‑Pastur distribution for random correlation matrix."""
    q = n / T
    lambda_plus = (1 + np.sqrt(q)) ** 2
    return lambda_plus

def rmt_filtered_centrality(returns_df, window):
    """
    For the last `window` days of returns:
    - Compute correlation matrix
    - Get eigenvalues and eigenvectors
    - Determine signal eigenvalues (those > lambda_plus)
    - Reconstruct filtered correlation matrix using only signal eigenvalues
    - Compute eigenvector centrality from the largest eigenvector of filtered matrix
    Returns:
        centrality: dict mapping ticker -> centrality score
        n_signal: number of signal eigenvalues
        filtered_corr: filtered correlation matrix (for diagnostics)
        raw_corr: raw correlation matrix
    """
    if len(returns_df) < window:
        return None
    data = returns_df.iloc[-window:].dropna(axis=1, how='any')
    if data.shape[1] < 2:
        return None
    n = data.shape[1]
    T = data.shape[0]
    corr = data.corr().values
    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = eigvals[::-1]          # descending
    eigvecs = eigvecs[:, ::-1]
    # Marchenko‑Pastur upper bound
    lambda_plus = marchenko_pastur_lambda_plus(n, T)
    # Identify signal eigenvalues (those > lambda_plus)
    signal_indices = np.where(eigvals > lambda_plus)[0]
    n_signal = len(signal_indices)
    if n_signal == 0:
        # fallback: take the largest eigenvalue as signal
        signal_indices = [0]
        n_signal = 1
    # Reconstruct filtered correlation matrix using only signal components
    # filtered_corr = sum_{i in signal} lambda_i * v_i v_i^T
    filtered_corr = np.zeros((n, n))
    for idx in signal_indices:
        ev = eigvals[idx]
        vec = eigvecs[:, idx]
        filtered_corr += ev * np.outer(vec, vec)
    # Set diagonal to 1
    np.fill_diagonal(filtered_corr, 1.0)
    # Ensure positive semidefinite (clip small negative)
    filtered_corr = np.maximum(filtered_corr, 0)
    # Eigenvector centrality from the largest eigenvector of filtered correlation
    # (equivalent to the first principal component of filtered matrix)
    f_eigvals, f_eigvecs = np.linalg.eigh(filtered_corr)
    f_eigvals = f_eigvals[::-1]
    f_eigvecs = f_eigvecs[:, ::-1]
    centralities = np.abs(f_eigvecs[:, 0])   # absolute value of first eigenvector
    # Normalise to sum to 1 (optional)
    centralities = centralities / np.sum(centralities)
    tickers = data.columns.tolist()
    centrality_dict = {ticker: float(centralities[i]) for i, ticker in enumerate(tickers)}
    # Also return raw eigenvector centrality from raw matrix? We'll keep filtered.
    return {
        "centrality": centrality_dict,
        "n_signal": int(n_signal),
        "filtered_corr": filtered_corr.tolist(),
        "raw_corr": corr.tolist(),
        "tickers": tickers
    }
