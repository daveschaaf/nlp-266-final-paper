"""Leakage guardrails for using FinRank as a held-out benchmark."""

FINRANK_TICKERS = frozenset({
    "ABBV", "AMGN", "COP", "CTRA", "CVX", "DVN", "EOG", "EPD", "ET",
    "F", "GM", "JNJ", "KMI", "LLY", "LNG", "MPC", "MRK", "MRNA",
    "PFE", "TSLA", "WMB", "XOM",
})


def is_finrank_ticker(ticker: str) -> bool:
    return ticker.upper() in FINRANK_TICKERS


def assert_not_finrank_tickers(tickers: list[str] | tuple[str, ...]) -> None:
    blocked = sorted({ticker.upper() for ticker in tickers if is_finrank_ticker(ticker)})
    if blocked:
        raise ValueError(
            "FinRank benchmark issuers are excluded from extract ingestion and training: "
            + ", ".join(blocked)
        )


def fetch_finrank_ciks(connection) -> frozenset[str]:
    placeholders = ", ".join("?" for _ in FINRANK_TICKERS)
    rows = connection.execute(
        f"SELECT cik FROM company WHERE ticker IN ({placeholders})",
        tuple(FINRANK_TICKERS),
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)
