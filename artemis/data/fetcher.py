"""NIFTY 50 data acquisition and feature engineering for ARTEMIS."""
import argparse
import os

import numpy as np
import pandas as pd
import ta
import yfinance as yf

SEED = 42
np.random.seed(SEED)

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
NORM_COLS = ["Close", "rsi", "macd_signal", "bb_position", "ema_distance", "volume_ratio"]


class DataPipeline:
    TICKER = "^NSEI"

    REGIME_DATES = {
        "bull": ("2020-01-01", "2022-01-01"),
        "bear": ("2022-01-01", "2023-01-01"),
        "sideways": ("2023-01-01", "2024-06-01"),
    }

    def __init__(self):
        self._min = None
        self._max = None

    def fetch_raw(self, start: str, end: str) -> pd.DataFrame:
        cache_path = os.path.join("data", "cache", f"raw_{start}_{end}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(self.TICKER, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[OHLCV_COLS]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)
        df = df.dropna(how="any")
        return df[OHLCV_COLS]

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        macd = ta.trend.MACD(df["Close"])
        df["macd_signal"] = macd.macd_signal()

        bb = ta.volatility.BollingerBands(df["Close"])
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        df["bb_position"] = ((df["Close"] - bb_lower) / (bb_upper - bb_lower)).clip(0, 1)

        ema20 = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
        df["ema_distance"] = (df["Close"] - ema20) / ema20

        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        df = df.dropna(how="any")
        return df

    def normalise(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        if fit or self._min is None:
            self._min = df[NORM_COLS].min()
            self._max = df[NORM_COLS].max()
        denom = (self._max - self._min).replace(0, 1e-8)
        df[NORM_COLS] = (df[NORM_COLS] - self._min) / denom
        df[NORM_COLS] = df[NORM_COLS].clip(0, 1)
        return df

    def get_regime(self, regime: str, normalise: bool = True) -> pd.DataFrame:
        start, end = self.REGIME_DATES[regime]
        df = self.fetch_raw(start, end)
        df = self.add_indicators(df)
        if normalise:
            df = self.normalise(df, fit=True)
        return df

    def split_all(self) -> dict:
        bull_raw = self.add_indicators(self.fetch_raw(*self.REGIME_DATES["bull"]))
        bear_raw = self.add_indicators(self.fetch_raw(*self.REGIME_DATES["bear"]))
        sideways_raw = self.add_indicators(self.fetch_raw(*self.REGIME_DATES["sideways"]))

        bull = self.normalise(bull_raw, fit=True)
        bear = self.normalise(bear_raw, fit=False)
        sideways = self.normalise(sideways_raw, fit=False)

        return {"bull": bull, "bear": bear, "sideways": sideways}


def cache_data(output_dir: str = "data/cache") -> None:
    os.makedirs(output_dir, exist_ok=True)
    pipeline = DataPipeline()
    regimes = pipeline.split_all()
    for name, df in regimes.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path)
        print(f"[ARTEMIS] Cached {name} regime ({len(df)} rows) -> {path}")


if __name__ == "__main__":
    print("[ARTEMIS] Running data.fetcher...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/cache")
    args = parser.parse_args()
    cache_data(args.output_dir)
