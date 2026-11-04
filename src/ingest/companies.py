"""Default company universe for the corpus (US-listed 10-K filers).

Mix of banks, insurers and non-financial companies so that questions can be
about different sectors and so that cross-company comparisons are possible.
"""

DEFAULT_TICKERS = [
    # Banks
    "JPM",
    "BAC",
    "C",
    "WFC",
    "GS",
    "MS",
    # Insurers
    "MET",
    "AIG",
    "PRU",
    "TRV",
    # Technology
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "AAPL",
    # Other sectors
    "PFE",
    "JNJ",
    "WMT",
]
