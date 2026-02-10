import os
import numpy as np
import pandas as pd
import talib
from hmmlearn.hmm import GaussianHMM

class FeatureConfig:
    """Configuration for feature engineering"""
    
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.raw_path = os.path.join(self.data_dir, "raw", "btcusdt_h4.parquet")
        self.feature_path = os.path.join(self.data_dir, "processed", "btcusdt_h4_features.parquet")
        
        # HMM parameters
        self.hmm_n_states = 3
        self.hmm_n_iter = 100
        self.hmm_random_state = 42
        
        # Indicator periods
        self.ema_periods = [9, 21, 50, 100, 200]
        self.rsi_period = 14
        self.atr_period = 14
        self.adx_period = 14
        self.bb_period = 20
        self.bb_std = 2.0


class TechnicalIndicators:
    """Calculate technical indicators using TA-Lib"""
    
    def calculate_all(self, df, config):
        """Calculate all technical indicators"""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        
        indicators = pd.DataFrame(index=df.index)
        
        # EMAs
        for period in config.ema_periods:
            indicators[f"ema_{period}"] = talib.EMA(close, timeperiod=period)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist
        
        # RSI
        indicators["rsi_14"] = talib.RSI(close, timeperiod=config.rsi_period)
        
        # ATR
        indicators["atr_14"] = talib.ATR(high, low, close, timeperiod=config.atr_period)
        
        # ADX
        indicators["adx_14"] = talib.ADX(high, low, close, timeperiod=config.adx_period)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=config.bb_period, 
                                            nbdevup=config.bb_std, nbdevdn=config.bb_std, matype=0)
        indicators["bb_upper"] = upper
        indicators["bb_middle"] = middle
        indicators["bb_lower"] = lower
        indicators["bb_width"] = (upper - lower) / middle * 100
        
        # OBV
        indicators["obv"] = talib.OBV(close, volume)
        
        return indicators.fillna(0)


class RegimeDetector:
    """Detect market regimes using various methods"""
    
    def detect_hmm_regime(self, df, config):
        """Detect market regime using Hidden Markov Model"""
        features = pd.DataFrame(index=df.index)
        
        try:
            # Prepare features
            returns = df['close'].pct_change().fillna(0)
            volatility = df['close'].rolling(14).std().fillna(0)
            X = np.column_stack([returns, volatility])
            
            # Fit HMM
            model = GaussianHMM(
                n_components=config.hmm_n_states,
                covariance_type="full",
                n_iter=config.hmm_n_iter,
                random_state=config.hmm_random_state
            )
            model.fit(X)
            
            # Predict states
            hidden_states = model.predict(X)
            
            # Map states by volatility
            mapped_states = self._map_states_by_volatility(hidden_states, volatility, config.hmm_n_states)
            
            # Features
            features['hmm_regime_high'] = (mapped_states == 2).astype(int)
            features['hmm_regime_duration'] = self._calculate_regime_duration(mapped_states, df.index)
            
        except Exception as e:
            print(f"HMM error: {e}")
            features['hmm_regime_high'] = 0
            features['hmm_regime_duration'] = 1
        
        return features
    
    def _map_states_by_volatility(self, states, volatility, n_states):
        """Map HMM states by volatility level"""
        state_volatility = []
        for state in range(n_states):
            mask = states == state
            avg_vol = volatility[mask].mean()
            state_volatility.append((state, avg_vol))
        
        state_volatility.sort(key=lambda x: x[1])
        state_mapping = {old: new for new, (old, _) in enumerate(state_volatility)}
        return np.array([state_mapping[s] for s in states])
    
    def _calculate_regime_duration(self, states, index):
        """Calculate duration of each regime"""
        regime_series = pd.Series(states, index=index)
        regime_changes = (regime_series != regime_series.shift(1)).cumsum()
        return regime_changes.groupby(regime_changes).cumcount() + 1
    
    def detect_trend_regime(self, df):
        """Detect trend-based regime"""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        try:
            # Get ADX
            if 'adx_14' in df.columns:
                adx = df['adx_14']
            else:
                adx = pd.Series(talib.ADX(df['high'].values, df['low'].values, 
                                         close.values, timeperiod=14), index=df.index)
            
            # Get EMAs
            if 'ema_21' in df.columns:
                ema_21, ema_50, ema_200 = df['ema_21'], df['ema_50'], df['ema_200']
            else:
                ema_21 = close.ewm(span=21).mean()
                ema_50 = close.ewm(span=50).mean()
                ema_200 = close.ewm(span=200).mean()
            
            # EMA alignment score
            above_ema21 = (close > ema_21).astype(int)
            above_ema50 = (close > ema_50).astype(int)
            above_ema200 = (close > ema_200).astype(int)
            alignment = above_ema21 + above_ema50 + above_ema200
            features['ema_alignment'] = alignment - 1.5
            
            # Strong bull trend
            ema_trend = ((ema_21 > ema_50).astype(int) * 2 - 1)
            trend_strong = (adx > 25).astype(int)
            features['trend_strong_bull'] = ((trend_strong == 1) & (ema_trend == 1)).astype(int)
            
        except Exception as e:
            print(f"Trend regime error: {e}")
            features['ema_alignment'] = 0
            features['trend_strong_bull'] = 0
        
        return features
    
    def detect_volume_regime(self, df):
        """Detect volume-based regime"""
        features = pd.DataFrame(index=df.index)
        volume = df['volume']
        
        try:
            # Volume percentile
            features['volume_percentile'] = volume.rolling(100, min_periods=1).apply(
                lambda x: (pd.Series(x).rank(pct=True).iloc[-1] * 100) if len(x) > 0 else 50
            )
            
            # Volume trend
            vol_ma_5 = volume.rolling(5, min_periods=1).mean()
            vol_ma_20 = volume.rolling(20, min_periods=1).mean()
            features['volume_trend'] = (vol_ma_5 / vol_ma_20) - 1
            
        except Exception as e:
            print(f"Volume regime error: {e}")
            features['volume_percentile'] = 50
            features['volume_trend'] = 0
        
        return features


class PriceFeatureExtractor:
    """Extract price-based features"""
    
    def extract_all(self, df):
        """Extract all price-based features"""
        features = pd.DataFrame(index=df.index)
        
        close = df["close"]
        high = df["high"].values
        low = df["low"].values
        close_arr = close.values
        volume = df["volume"]
        
        # Log returns
        features["log_return"] = np.log(close / close.shift(1))
        
        # Linear regression slope
        features["slope"] = talib.LINEARREG_SLOPE(close_arr, timeperiod=20)
        
        # Normalized ATR
        features["atr_norm"] = talib.ATR(high, low, close_arr, timeperiod=14) / close_arr
        
        # Volume Z-score
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        features["volume_z"] = (volume - vol_mean) / vol_std
        
        # Trend efficiency
        window = 20
        net_move = (close - close.shift(window)).abs()
        path_move = close.diff().abs().rolling(window).sum()
        features["efficiency"] = net_move / path_move
        
        return features


class FeatureEngineer:
    """Main feature engineering orchestrator"""
    
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.technical = TechnicalIndicators()
        self.regime = RegimeDetector()
        self.price = PriceFeatureExtractor()
    
    def build_features(self, df):
        """Build comprehensive feature set"""
        print("Building technical indicators...")
        technical = self.technical.calculate_all(df, self.config)
        
        print("Building market regime features...")
        hmm_regime = self.regime.detect_hmm_regime(df, self.config)
        trend_regime = self.regime.detect_trend_regime(df)
        volume_regime = self.regime.detect_volume_regime(df)
        
        print("Building price features...")
        price_features = self.price.extract_all(df)
        
        # Combine all features
        features = pd.concat([
            technical,
            hmm_regime,
            trend_regime,
            volume_regime,
            price_features
        ], axis=1)
        
        return features
    
    def load_data(self):
        """Load raw data from parquet file"""
        print(f"Loading data from: {self.config.raw_path}")
        df = pd.read_parquet(self.config.raw_path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    
    def save_features(self, df):
        """Save features to parquet file"""
        os.makedirs(os.path.dirname(self.config.feature_path), exist_ok=True)
        df.to_parquet(self.config.feature_path, engine="pyarrow", index=False)
        print(f"\nSaved to: {self.config.feature_path}")
    
    def print_summary(self, df_original, df_features):
        """Print feature engineering summary"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        
        print(f"\nOriginal data: {len(df_original):,} rows")
        print(f"Final data: {len(df_features):,} rows")
        print(f"Total features: {len(df_features.columns) - 1}")
        print(f"Date range: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
        
        self._print_feature_groups(df_features)
    
    def _print_feature_groups(self, df):
        """Print categorized feature list"""
        feature_groups = {
            "Technical Indicators": [
                'ema_9', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
                'macd', 'macd_signal', 'macd_hist', 'rsi_14',
                'atr_14', 'adx_14', 'bb_upper', 'bb_middle',
                'bb_lower', 'bb_width', 'obv'
            ],
            "Market Regimes": [
                'hmm_regime_high', 'hmm_regime_duration',
                'ema_alignment', 'trend_strong_bull',
                'volume_percentile', 'volume_trend'
            ],
            "Price Features": [
                'log_return', 'slope', 'atr_norm',
                'volume_z', 'efficiency'
            ]
        }
        
        print("\nFeature groups:")
        for group_name, feature_list in feature_groups.items():
            print(f"\n{group_name}:")
            existing_features = [f for f in feature_list if f in df.columns]
            for feature in existing_features:
                print(f"   - {feature}")
    
    def run(self):
        """Execute complete feature engineering pipeline"""
        print("=" * 60)
        print("FEATURE ENGINEERING - BTCUSDT H4")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        print(f"Loaded {len(df):,} rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Build features
        print("\nBuilding feature set...")
        features = self.build_features(df)
        
        # Combine with timestamp
        df_features = pd.concat([df[["timestamp"]], features], axis=1)
        
        # Clean up NaN values
        df_features = df_features.dropna().reset_index(drop=True)
        
        # Save results
        self.save_features(df_features)
        
        # Print summary
        self.print_summary(df, df_features)
        
        print("\n" + "=" * 60)
        
        return df_features


def main():
    """Main execution function"""
    engineer = FeatureEngineer()
    df_features = engineer.run()
    return df_features


if __name__ == "__main__":
    main()