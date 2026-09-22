"""
Extract sector-grouped price data from marketDataUS.sqlite.

For each GICS sector present in the SP500 application, creates a CSV file
with adjClose prices for all assets in that sector, matching the same date
range and asset universe as marketData.csv.

Output: src/data/sectors/<SECTOR_NAME>.csv
        Columns: Date + one column per asset code (adjClose price)
        Rows: same trading days as marketData.csv
"""

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
SECTORS_DIR = DATA_DIR / "sectors"
SQLITE_PATH = DATA_DIR / "marketDataUS.sqlite"
MARKET_CSV = DATA_DIR / "marketData.csv"


def get_sector_asset_map(conn: sqlite3.Connection, asset_codes: set[str]) -> dict[str, list[str]]:
    """Return {sector: [asset_code, ...]} for assets present in marketData.csv."""
    q = """
        SELECT aa.asset, t.tag AS sector
        FROM ApplicationAsset aa
        JOIN TagApplicationAsset taa ON taa.asset = aa.asset AND taa.app = aa.app
        JOIN Tag t ON t.tag = taa.tag
        WHERE aa.app = 'SP500'
          AND t.category = 'SECTOR'
          AND t.domain = ''
    """
    df = pd.read_sql(q, conn)
    df = df[df["asset"].isin(asset_codes)]
    sectors: dict[str, list[str]] = {}
    for sector, group in df.groupby("sector"):
        sectors[sector] = sorted(group["asset"].tolist())
    return sectors


def fetch_sector_prices(
    conn: sqlite3.Connection,
    assets: list[str],
    dates: list[str],
) -> pd.DataFrame:
    """Fetch adjClose prices for given assets over the given date range."""
    placeholders = ",".join("?" * len(assets))
    q = f"""
        SELECT date, asset, adjClose
        FROM AssetPrice
        WHERE asset IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY date, asset
    """
    params = assets + [dates[0], dates[-1]]
    df = pd.read_sql(q, conn, params=params)
    pivot = df.pivot(index="date", columns="asset", values="adjClose")
    # Align to exact dates from marketData.csv (keeps same trading calendar)
    pivot = pivot.reindex(dates)
    pivot.index.name = "Date"
    return pivot


def sector_filename(sector: str) -> str:
    return sector.replace(" ", "_") + ".csv"


def main() -> None:
    market = pd.read_csv(MARKET_CSV, index_col="Date")
    asset_codes = {c for c in market.columns if c != "SP500"}
    dates = list(market.index)
    print(f"marketData.csv: {len(asset_codes)} assets, {len(dates)} dates ({dates[0]} -> {dates[-1]})")

    conn = sqlite3.connect(SQLITE_PATH)
    sectors = get_sector_asset_map(conn, asset_codes)

    untagged = asset_codes - {a for assets in sectors.values() for a in assets}
    if untagged:
        print(f"Warning: {len(untagged)} assets have no sector tag and will be skipped: {untagged}")

    SECTORS_DIR.mkdir(exist_ok=True)

    print(f"\nExtracting {len(sectors)} sectors -> {SECTORS_DIR}/")
    for sector, assets in sorted(sectors.items()):
        prices = fetch_sector_prices(conn, assets, dates)
        missing_dates = prices.index[prices.isna().all(axis=1)]
        if len(missing_dates) > 0:
            print(f"  [{sector}] {len(missing_dates)} dates with all-NaN (expected for holidays)")
        out_path = SECTORS_DIR / sector_filename(sector)
        prices.to_csv(out_path)
        print(f"  {sector}: {len(assets)} assets -> {out_path.name}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
