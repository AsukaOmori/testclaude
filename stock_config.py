"""Configuration for stock-level regime-dependent lead-lag strategy."""

# US Leader Stocks (15 stocks, 5 sectors)
US_STOCKS = {
    "tech": ["NVDA", "AMD", "INTC", "AVGO"],
    "financials": ["JPM", "GS", "MS", "BAC"],
    "auto_ind": ["GM", "F", "CAT", "DE"],
    "energy": ["XOM", "CVX", "COP"],
    "consumer": ["AMZN", "WMT", "COST"],
}

# Japan Target Stocks (13 stocks, 4 sectors)
JP_STOCKS = {
    "tech": ["6857.T", "6920.T", "8035.T", "6758.T"],
    "financials": ["8306.T", "8316.T", "8411.T"],
    "auto_ind": ["7203.T", "7267.T", "6301.T"],
    "trading": ["8058.T", "8031.T", "8001.T"],
}

# Korea Target Stocks (10 stocks, 4 sectors)
KR_STOCKS = {
    "tech": ["005930.KS", "000660.KS", "035420.KS"],
    "auto_ind": ["005380.KS", "000270.KS"],
    "financials": ["055550.KS", "105560.KS"],
    "industrial": ["005490.KS", "051910.KS", "015760.KS"],
}

# Sector mapping for cross-correlation signal
# Maps target sectors to corresponding US sectors
SECTOR_MAPPING = {
    "tech": "tech",
    "financials": "financials",
    "auto_ind": "auto_ind",
    "trading": "auto_ind",      # JP trading houses → US industrials
    "industrial": "auto_ind",   # KR industrials → US industrials
}

# Flat ticker lists (derived)
US_TICKERS_FLAT = [t for tickers in US_STOCKS.values() for t in tickers]
JP_TICKERS_FLAT = [t for tickers in JP_STOCKS.values() for t in tickers]
KR_TICKERS_FLAT = [t for tickers in KR_STOCKS.values() for t in tickers]

N_US = len(US_TICKERS_FLAT)
N_JP = len(JP_TICKERS_FLAT)
N_KR = len(KR_TICKERS_FLAT)

# Parameters
WINDOW = 60            # Rolling window (L) in business days
K_COMPONENTS = 3       # Number of PCA components
LAMBDA_STOCK = 0.7     # Regularization strength (lower than ETF due to noise)
Q = 0.3                # Long-short quantile (tercile)
IC_WINDOW = 40         # Rolling IC window for regime detection
IC_THRESHOLD = 0.03    # IC threshold for active regime

# Date ranges
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
