from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """Create required directories"""
    directories = ['data/raw', 'data/processed', 'outputs/plots', 'img']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def fetch_data():
    """Fetch cryptocurrency data"""
    logger.info("Step 1: Fetching data")
    from data.fetch_data import BinanceDataFetcher
    
    fetcher = BinanceDataFetcher(symbol="BTCUSDT", interval="4h")
    start_date = datetime(2023, 1, 1)
    
    raw_data = fetcher.fetch_klines(start_date=start_date)
    df = fetcher.process_data(raw_data)
    fetcher.save_to_parquet(df, "data/raw/btcusdt_h4.parquet")
    
    logger.info(f"Data fetched: {len(df)} rows")
    return df


def engineer_features():
    """Create technical features"""
    logger.info("Step 2: Engineering features")
    from data.feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    df_features = engineer.run()
    
    logger.info(f"Features created: {len(df_features)} rows, {len(df_features.columns)} columns")
    return df_features


def preprocess():
    """Preprocess and scale data"""
    logger.info("Step 3: Preprocessing data")
    from data.preprocessing import FeaturePreprocessor
    
    preprocessor = FeaturePreprocessor()
    df_clean, X_scaled, scaler = preprocessor.run()
    
    logger.info(f"Preprocessing complete: {X_scaled.shape}")
    return df_clean, X_scaled, scaler


def train_model():
    """Train HMM model"""
    logger.info("Step 4: Training HMM model")
    from model.hmm_regime import HMMRegimeIdentifier
    
    identifier = HMMRegimeIdentifier()
    df_analysis, model, state_probs = identifier.run()
    
    logger.info(f"Model trained: {model.n_components} states, converged={model.monitor_.converged}")
    
    # Show regime distribution
    regime_counts = df_analysis['state_mapped'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        logger.info(f"Regime {regime}: {count} periods ({count/len(df_analysis)*100:.1f}%)")
    
    return df_analysis, model, state_probs


def main():
    """Execute pipeline"""
    try:
        start = datetime.now()
        logger.info("Starting HMM pipeline")
        
        ensure_directories()
        fetch_data()
        engineer_features()
        preprocess()
        df_analysis, model, state_probs = train_model()
        
        duration = datetime.now() - start
        logger.info(f"Pipeline complete in {duration}")
        logger.info(f"Current regime: {df_analysis['state_mapped'].iloc[-1]}")
        
        return df_analysis
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    print("Hello from hidden-markov-model!")
    main()