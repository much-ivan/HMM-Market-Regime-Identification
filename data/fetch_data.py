import os
import time
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import requests

class BinanceDataFetcher:
    """Fetches cryptocurrency data from Binance API"""

    BASE_URL = "https://api.binance.com/api/v3/klines"
    DEFAULT_LIMIT = 1000
    REQUEST_DELAY = 0.5  # seconds
    
    KLINE_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    
    OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
    NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]
    
    def __init__(self, symbol: str = "BTCUSDT", interval: str = "4h"):
        self.symbol = symbol
        self.interval = interval
    
    def fetch_klines(
        self, 
        start_date: datetime, 
        end_date: datetime = None
    ) -> List[List[Any]]:
        """
        Fetch kline/candlestick data from Binance API
        
        Args:
            start_date: Starting date for data fetch
            end_date: Optional ending date (defaults to now)
            
        Returns:
            List of kline data arrays
        """
        all_data = []
        start_time = int(start_date.timestamp() * 1000)
        
        print(f"Fetching {self.symbol} {self.interval} data from Binance...")
        
        while True:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": start_time,
                "limit": self.DEFAULT_LIMIT,
            }
            
            if end_date:
                params["endTime"] = int(end_date.timestamp() * 1000)
            
            try:
                data = self._make_request(params)
            except Exception as e:
                print(f"Error fetching data: {e}")
                raise
            
            if not data:
                break
            
            all_data.extend(data)
            print(f"Fetched {len(all_data)} candles...", end="\r")
            
            # Update start time for next batch
            start_time = data[-1][0] + 1
            
            # Rate limiting
            time.sleep(self.REQUEST_DELAY)
            
            # Stop if we got less than requested (reached end)
            if len(data) < self.DEFAULT_LIMIT:
                break
        
        print(f"\nTotal candles fetched: {len(all_data)}")
        return all_data
    
    def _make_request(self, params: Dict[str, Any]) -> List[List[Any]]:
        """Make HTTP request to Binance API with error handling"""
        response = requests.get(
            self.BASE_URL, 
            params=params, 
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API error response
        if isinstance(data, dict):
            error_msg = data.get("msg", "Unknown error")
            raise RuntimeError(f"Binance API error: {error_msg}")
        
        return data
    
    def process_data(self, raw_data: List[List[Any]]) -> pd.DataFrame:
        """
        Convert raw kline data to cleaned DataFrame
        
        Args:
            raw_data: Raw kline data from API
            
        Returns:
            Cleaned pandas DataFrame
        """
        df = pd.DataFrame(raw_data, columns=self.KLINE_COLUMNS)
        
        # Convert timestamps
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # Convert numeric columns
        for col in self.NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Select and rename columns
        df_clean = df[["open_time"] + self.NUMERIC_COLUMNS].copy()
        df_clean.rename(columns={"open_time": "timestamp"}, inplace=True)
        
        return df_clean
    
    def save_to_parquet(
        self, 
        df: pd.DataFrame, 
        file_path: str
    ) -> None:
        """
        Save DataFrame to Parquet file
        
        Args:
            df: DataFrame to save
            file_path: Output file path
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_parquet(file_path, engine="pyarrow", index=False)
        print(f"\nSaved data to: {file_path}")
        print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total rows: {len(df)}")


def main():
    """Main execution function"""
    # Configuration
    SAVE_PATH = "data/raw/btcusdt_h4.parquet"
    START_DATE = datetime(2023, 1, 1)
    
    # Initialize fetcher
    fetcher = BinanceDataFetcher(symbol="BTCUSDT", interval="4h")
    
    # Fetch data
    raw_data = fetcher.fetch_klines(start_date=START_DATE)
    
    # Process data
    df_clean = fetcher.process_data(raw_data)
    
    # Save to file
    fetcher.save_to_parquet(df_clean, SAVE_PATH)


if __name__ == "__main__":
    main()