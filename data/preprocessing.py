import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

class PreprocessConfig:
    """Configuration for preprocessing"""
    
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.output_dir = os.path.join(self.base_dir, "outputs")
        
        # File paths
        self.feature_path = os.path.join(self.processed_dir, "btcusdt_h4_features.parquet")
        self.clean_path = os.path.join(self.processed_dir, "btcusdt_h4_features_clean.parquet")
        self.scaled_path = os.path.join(self.processed_dir, "btcusdt_h4_features_scaled.npy")
        self.scaler_path = os.path.join(self.processed_dir, "scaler.pkl")
        self.corr_plot_path = os.path.join(self.output_dir, "correlation_matrix.png")
        
        # Thresholds
        self.outlier_std_threshold = 3
        self.outlier_pct_threshold = 1.0
        self.corr_very_high = 0.90
        self.corr_high = 0.85
        self.corr_moderate = 0.70


class FeatureCleaner:
    """Clean features by removing invalid values"""
    
    def clean(self, df):
        """Remove inf, NaN values and maintain timestamp alignment"""
        print(f"\nCleaning features...")
        print(f"Original shape: {df.shape}")
        
        df = df.copy()
        
        # Separate timestamp
        timestamp = df["timestamp"]
        features = df.drop(columns=["timestamp"])
        
        # Replace infinite values with NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Count issues
        inf_count = np.isinf(df.drop(columns=["timestamp"]).values).sum()
        nan_count = features.isna().sum().sum()
        
        print(f"Found {inf_count} infinite values")
        print(f"Found {nan_count} NaN values")
        
        # Drop rows with NaN
        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        timestamp = timestamp.loc[valid_idx]
        
        # Reconstruct dataframe
        df_clean = pd.concat([timestamp, features], axis=1).reset_index(drop=True)
        
        print(f"Cleaned shape: {df_clean.shape}")
        print(f"Rows removed: {len(df) - len(df_clean)}")
        
        return df_clean


class FeatureAnalyzer:
    """Analyze feature statistics and correlations"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_statistics(self, df):
        """Analyze feature statistics before scaling"""
        print("\n" + "=" * 60)
        print("FEATURE STATISTICS (Before Scaling)")
        print("=" * 60)
        
        features = df.drop(columns=["timestamp"])
        stats = features.describe().T
        
        # Add skewness and kurtosis
        stats['skew'] = features.skew()
        stats['kurtosis'] = features.kurtosis()
        
        print(stats[['mean', 'std', 'min', 'max', 'skew', 'kurtosis']].round(4))
        
        # Check for outliers
        self._detect_outliers(features)
    
    def _detect_outliers(self, features):
        """Detect outliers using z-score method"""
        print("\n" + "=" * 60)
        print(f"OUTLIER DETECTION (|z-score| > {self.config.outlier_std_threshold})")
        print("=" * 60)
        
        for col in features.columns:
            z_scores = np.abs((features[col] - features[col].mean()) / features[col].std())
            outlier_count = (z_scores > self.config.outlier_std_threshold).sum()
            outlier_pct = (outlier_count / len(features)) * 100
            
            if outlier_pct > self.config.outlier_pct_threshold:
                print(f"{col:20s}: {outlier_count:5d} ({outlier_pct:5.2f}%)")
    
    def plot_correlation(self, df):
        """Generate and save correlation matrix heatmap"""
        print("\nGenerating correlation matrix...")
        
        features = df.drop(columns=["timestamp"])
        corr = features.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        
        plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save plot
        os.makedirs(self.config.output_dir, exist_ok=True)
        plt.savefig(self.config.corr_plot_path, dpi=150, bbox_inches='tight')
        print(f"Correlation plot saved to: {self.config.corr_plot_path}")
        
        plt.show()
        
        return corr
    
    def analyze_correlation(self, corr):
        """Analyze correlation and warn about multicollinearity"""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Find high correlation pairs
        high_corr = (
            corr.abs()
            .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        
        # Filter by thresholds
        very_high = high_corr[high_corr > self.config.corr_very_high]
        high = high_corr[(high_corr > self.config.corr_high) & (high_corr <= self.config.corr_very_high)]
        moderate = high_corr[(high_corr > self.config.corr_moderate) & (high_corr <= self.config.corr_high)]
        
        # Print results
        if len(very_high) > 0:
            print(f"\n⚠️  VERY HIGH correlation (>{self.config.corr_very_high}) - Consider removing one:")
            for (feat1, feat2), val in very_high.items():
                print(f"   {feat1:20s} ↔ {feat2:20s} : {val:.3f}")
        
        if len(high) > 0:
            print(f"\n⚠️  HIGH correlation ({self.config.corr_high}-{self.config.corr_very_high}):")
            for (feat1, feat2), val in high.items():
                print(f"   {feat1:20s} ↔ {feat2:20s} : {val:.3f}")
        
        if len(moderate) > 0:
            print(f"\nModerate correlation ({self.config.corr_moderate}-{self.config.corr_high}):")
            for (feat1, feat2), val in moderate.items():
                print(f"   {feat1:20s} ↔ {feat2:20s} : {val:.3f}")
        
        if len(very_high) == 0 and len(high) == 0:
            print(f"\n✅ No severe multicollinearity detected (all < {self.config.corr_high})")


class FeatureScaler:
    """Scale features using StandardScaler"""
    
    def scale(self, df):
        """Scale features and return scaled array with scaler object"""
        print("\nScaling features...")
        
        scaler = StandardScaler()
        features = df.drop(columns=["timestamp"])
        
        X_scaled = scaler.fit_transform(features.values)
        
        print(f"Scaled shape: {X_scaled.shape}")
        print(f"Mean (should be ~0): {X_scaled.mean(axis=0).mean():.6f}")
        print(f"Std (should be ~1): {X_scaled.std(axis=0).mean():.6f}")
        
        return X_scaled, scaler


class FeaturePreprocessor:
    """Main preprocessing orchestrator"""
    
    def __init__(self, config=None):
        self.config = config or PreprocessConfig()
        self.cleaner = FeatureCleaner()
        self.analyzer = FeatureAnalyzer(self.config)
        self.scaler = FeatureScaler()
    
    def load_data(self):
        """Load features from parquet file"""
        print(f"\nLoading features from: {self.config.feature_path}")
        df = pd.read_parquet(self.config.feature_path)
        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def save_outputs(self, df_clean, X_scaled, scaler):
        """Save all preprocessing outputs"""
        print("\n" + "=" * 60)
        print("SAVING OUTPUTS")
        print("=" * 60)
        
        os.makedirs(self.config.processed_dir, exist_ok=True)
        
        # Save clean dataframe
        df_clean.to_parquet(self.config.clean_path, engine="pyarrow", index=False)
        print(f"✓ Clean features: {self.config.clean_path}")
        
        # Save scaled array
        np.save(self.config.scaled_path, X_scaled)
        print(f"✓ Scaled features: {self.config.scaled_path}")
        
        # Save scaler
        joblib.dump(scaler, self.config.scaler_path)
        print(f"✓ Scaler object: {self.config.scaler_path}")
    
    def print_summary(self, df_clean, X_scaled):
        """Print preprocessing summary"""
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Final shape: {X_scaled.shape}")
        print(f"Features: {list(df_clean.columns[1:])}")
        print(f"Ready for HMM training!")
    
    def run(self):
        """Execute complete preprocessing pipeline"""
        print("=" * 60)
        print("FEATURE PREPROCESSING FOR HMM")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Clean features
        df_clean = self.cleaner.clean(df)
        
        # Analyze statistics
        self.analyzer.analyze_statistics(df_clean)
        
        # Check correlation
        corr = self.analyzer.plot_correlation(df_clean)
        self.analyzer.analyze_correlation(corr)
        
        # Scale features
        X_scaled, scaler = self.scaler.scale(df_clean)
        
        # Save outputs
        self.save_outputs(df_clean, X_scaled, scaler)
        
        # Print summary
        self.print_summary(df_clean, X_scaled)
        
        return df_clean, X_scaled, scaler


def main():
    """Main execution function"""
    preprocessor = FeaturePreprocessor()
    df_clean, X_scaled, scaler = preprocessor.run()
    return df_clean, X_scaled, scaler


if __name__ == "__main__":
    main()