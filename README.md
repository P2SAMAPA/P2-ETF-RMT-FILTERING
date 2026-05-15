# P2-ETF-RMT-FILTERING

Welcome to the P2-ETF-RMT-FILTERING repository.
# Random Matrix Theory (RMT) Filtering Engine

Cleans correlation matrices of ETF returns by separating signal from noise using the Marchenko‑Pastur distribution. The filtered correlation matrix is reconstructed from eigenvalues that exceed the theoretical upper bound for random matrices. Eigenvector centrality of the filtered matrix ranks ETFs by importance in the true underlying market structure.

- **Rolling window:** 252 days
- **Output:** top 3 ETFs per universe by filtered eigenvector centrality
- **Dashboard:** shows top ETFs, full ranking table, and number of signal eigenvalues
- Runs daily on GitHub Actions

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
