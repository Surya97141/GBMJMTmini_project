# NIFTY 50 + cross-market data acquisition and feature engineering for research-grade ARTEMIS
import argparse
import os

import numpy as np
import pandas as pd
import ta
import yfinance

SEED = 42
np.random.seed(SEED)

NORM_COLS = ["Close", "rsi", "macd_signal", "bb_position", "ema_distance", "volume_ratio"]


def raw_close(df: pd.DataFrame) -> pd.Series:
    # real (un-normalised) close price, whether or not df has been min-max scaled
    return df["Close_raw"] if "Close_raw" in df.columns else df["Close"]


class DataPipeline:
    TICKER = "^NSEI"
    CACHE_DIR = "data/cache"

    REGIME_DATES = {
        "bull": ("2020-01-01", "2022-01-01"),
        "bear": ("2022-01-01", "2023-01-01"),
        "sideways": ("2023-01-01", "2024-06-01"),
    }

    CROSS_MARKET_TICKERS = {
        "sensex": "^BSESN",
        "gold": "GC=F",
    }

    def __init__(self):
        self._min=None
        self._max = None

    def fetch_raw(self, start: str, end: str, ticker: str = None) -> pd.DataFrame:
        ticker = ticker or self.TICKER
        cache_path = f"{self.CACHE_DIR}/{ticker.replace('^', '').replace('=', '_')}_{start}_{end}.csv"
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = yfinance.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        df.to_csv(cache_path)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        volume = df["Volume"]
        df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd = ta.trend.MACD(close)
        df["macd_signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(close)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        df["bb_position"] = ((close - bb_lower) / (bb_upper - bb_lower + 1e-9)).clip(0, 1)
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        df["ema_distance"] = (close - ema20) / (ema20 + 1e-9)
        df["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)
        return df.dropna()

    def normalise(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        df["Close_raw"] = df["Close"]
        if fit or self._min is None:
            self._min = df[NORM_COLS].min()
            self._max = df[NORM_COLS].max()
        denom = (self._max - self._min).replace(0, 1e-9)
        df[NORM_COLS] = (df[NORM_COLS] - self._min) / denom
        df[NORM_COLS] = df[NORM_COLS].clip(0, 1)
        return df

    def get_regime(self, regime: str, normalise: bool = True, ticker: str = None) -> pd.DataFrame:
        start, end = self.REGIME_DATES[regime]
        df = self.fetch_raw(start, end, ticker=ticker)
        df = self.add_indicators(df)
        if normalise:
            df = self.normalise(df, fit=True)
        return df

    def split_all(self) -> dict:
        bull_df = self.get_regime("bull")

        bear_raw = self.get_regime("bear", normalise=False)
        bear_df = self.normalise(bear_raw, fit=False)

        sideways_raw = self.get_regime("sideways", normalise=False)
        sideways_df = self.normalise(sideways_raw, fit=False)

        return {"bull": bull_df, "bear": bear_df, "sideways": sideways_df}

    def get_cross_market(self, market_name: str) -> pd.DataFrame:
        ticker = self.CROSS_MARKET_TICKERS[market_name]
        start, end = self.REGIME_DATES["bull"][0], self.REGIME_DATES["sideways"][1]
        df = self.fetch_raw(start, end, ticker=ticker)
        df = self.add_indicators(df)
        df = self.normalise(df, fit=True)
        return df


if __name__ == "__main__":
    print("[ARTEMIS] Running data/fetcher.py...")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    pipeline = DataPipeline()
    for regime in ["bull", "bear", "sideways"]:
        df = pipeline.get_regime(regime)
        print(f"  {regime}: {len(df)} rows")
    for market in ["sensex", "gold"]:
        df = pipeline.get_cross_market(market)
        print(f"  {market}: {len(df)} rows")
