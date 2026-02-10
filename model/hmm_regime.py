import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

class HMMConfig:
    """Configuration for HMM training and visualization"""
    
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.output_dir = os.path.join(self.base_dir, "outputs")
        self.plot_dir = os.path.join(self.output_dir, "plots")
        
        # File paths
        self.feature_path = os.path.join(self.processed_dir, "btcusdt_h4_features.parquet")
        self.output_path = os.path.join(self.processed_dir, "btcusdt_h4_regimes.parquet")
        
        # HMM parameters
        self.n_states = 3
        self.random_state = 42
        self.covariance_type = "full"
        self.n_iter = 500
        
        # Features for HMM
        self.feature_cols = [
            "log_return",
            "atr_norm",
            "adx_14",
            "slope",
            "volume_z",
            "efficiency",
        ]
        
        # Visualization
        self.regime_colors = {
            0: '#27ae60',  # Green - Low volatility
            1: '#f39c12',  # Orange - Medium volatility
            2: '#e74c3c',  # Red - High volatility
        }
        
        self.regime_labels = {
            0: "Low Volatility",
            1: "Medium Volatility",
            2: "High Volatility"
        }


class DataLoader:
    """Load and prepare data for HMM"""
    
    def __init__(self, config):
        self.config = config
    
    def load_and_prepare(self):
        """Load features and prepare data for HMM"""
        print("Loading feature data...")
        df = pd.read_parquet(self.config.feature_path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"Loaded {len(df):,} rows")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Verify all features exist
        missing = [col for col in self.config.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        print(f"\nUsing {len(self.config.feature_cols)} features for HMM:")
        for col in self.config.feature_cols:
            print(f"  - {col}")
        
        # Extract features
        X = df[self.config.feature_cols].values
        
        # Clean data
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        if not valid_mask.all():
            print(f"\n⚠️  Removing {(~valid_mask).sum()} rows with NaN/Inf")
            X = X[valid_mask]
            df = df[valid_mask].reset_index(drop=True)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"\nFeature matrix shape: {X_scaled.shape}")
        
        return df, X_scaled, scaler


class HMMTrainer:
    """Train Gaussian HMM model"""
    
    def __init__(self, config):
        self.config = config
    
    def train(self, X_scaled):
        """Train Gaussian HMM model"""
        print(f"\nTraining Gaussian HMM with {self.config.n_states} states...")
        
        model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            verbose=False,
        )
        
        model.fit(X_scaled)
        
        print(f"Model converged: {model.monitor_.converged}")
        print(f"Log likelihood: {model.score(X_scaled):.2f}")
        
        hidden_states = model.predict(X_scaled)
        state_probs = model.predict_proba(X_scaled)
        
        return model, hidden_states, state_probs


class StateAnalyzer:
    """Analyze HMM state characteristics"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_states(self, df, hidden_states):
        """Analyze characteristics of each state"""
        print("\n" + "=" * 60)
        print("STATE ANALYSIS")
        print("=" * 60)
        
        df_analysis = df.copy()
        df_analysis['state'] = hidden_states
        
        # Order states by volatility
        state_volatility = []
        for state in range(self.config.n_states):
            mask = df_analysis['state'] == state
            avg_vol = df_analysis.loc[mask, 'atr_norm'].mean()
            state_volatility.append((state, avg_vol))
        
        state_volatility.sort(key=lambda x: x[1])
        state_mapping = {old: new for new, (old, _) in enumerate(state_volatility)}
        
        df_analysis['state_mapped'] = df_analysis['state'].map(state_mapping)
        
        # Print analysis
        for new_state in sorted(state_mapping.values()):
            old_state = [k for k, v in state_mapping.items() if v == new_state][0]
            mask = df_analysis['state'] == old_state
            
            print(f"\n{self.config.regime_labels[new_state]}")
            print("-" * 40)
            print(f"Count: {mask.sum():,} ({mask.sum()/len(df_analysis)*100:.1f}%)")
            print(f"Avg Return: {df_analysis.loc[mask, 'log_return'].mean()*100:.4f}%")
            print(f"Avg Volatility: {df_analysis.loc[mask, 'atr_norm'].mean()*100:.4f}%")
            print(f"Avg ADX: {df_analysis.loc[mask, 'adx_14'].mean():.2f}")
            print(f"Avg Volume Z-score: {df_analysis.loc[mask, 'volume_z'].mean():.2f}")
        
        return df_analysis, state_mapping


class RegimeVisualizer:
    """Visualize HMM regime analysis with candlestick charts"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_comprehensive_analysis(self, df_analysis, state_probs, model):
        """Create simple and clear visualization"""
        print("\nGenerating visualizations...")
        
        os.makedirs(self.config.plot_dir, exist_ok=True)
        
        # Load OHLCV data if available
        df_analysis = self._load_ohlcv_data(df_analysis)
        
        # 1. Main Candlestick Chart with Regimes (focus visual)
        self._plot_candlestick_regimes(df_analysis)
        
        # 2. Simple Summary Dashboard
        self._plot_simple_dashboard(df_analysis, state_probs, model)
        
        plt.close('all')
    
    def _load_ohlcv_data(self, df_analysis):
        """Load OHLCV data from raw file"""
        raw_path = os.path.join(self.config.data_dir, "raw", "btcusdt_h4.parquet")
        if os.path.exists(raw_path):
            df_raw = pd.read_parquet(raw_path)
            df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
            
            # Merge OHLCV data
            ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in ohlcv_cols if col in df_raw.columns]
            
            df_analysis = df_analysis.merge(df_raw[available_cols], on='timestamp', how='left')
        
        return df_analysis
    
    def _plot_candlestick_regimes(self, df_analysis):
        """Plot candlestick chart with regime backgrounds"""
        print("  → Creating candlestick chart...")
        
        # Check if OHLCV data available
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df_analysis.columns for col in required_cols):
            print("    ⚠️  OHLCV data not available, using close price only")
            self._plot_line_chart_regimes(df_analysis)
            return
        
        # Create figure
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 0.5], hspace=0.15)
        
        # Main candlestick chart
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Limit to recent data for clarity (last 500 candles)
        df_plot = df_analysis.tail(500).copy().reset_index(drop=True)
        
        # Draw regime backgrounds
        self._draw_regime_backgrounds(ax1, df_plot)
        self._draw_regime_backgrounds(ax2, df_plot)
        
        # Draw candlesticks
        self._draw_candlesticks(ax1, df_plot)
        
        # Draw volume
        self._draw_volume(ax2, df_plot)
        
        # Draw regime timeline
        self._draw_regime_timeline(ax3, df_plot)
        
        # Formatting
        ax1.set_title('BTCUSDT H4 — Market Regime Classification (Recent 500 Candles)',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        ax3.set_ylabel('Regime', fontsize=10, fontweight='bold')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Low', 'Med', 'High'], fontsize=9)
        ax3.set_xlabel('Time (Index)', fontsize=12, fontweight='bold')
        
        # Hide x-axis labels for top plots
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.plot_dir, 'candlestick_regimes.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    ✓ Candlestick chart saved: {plot_path}")
        
        plt.close()
    
    def _draw_regime_backgrounds(self, ax, df_plot):
        """Draw colored backgrounds for different regimes"""
        current_state = None
        start_idx = 0
        
        for idx, row in df_plot.iterrows():
            state = row['state_mapped']
            
            if state != current_state:
                if current_state is not None:
                    # Draw previous regime
                    color = self.config.regime_colors[current_state]
                    ax.axvspan(start_idx, idx, alpha=0.15, color=color, zorder=0)
                
                current_state = state
                start_idx = idx
        
        # Draw last regime
        if current_state is not None:
            color = self.config.regime_colors[current_state]
            ax.axvspan(start_idx, len(df_plot), alpha=0.15, color=color, zorder=0)
    
    def _draw_candlesticks(self, ax, df_plot):
        """Draw candlestick chart"""
        width = 0.6
        width2 = 0.05
        
        # Separate bullish and bearish
        up = df_plot[df_plot.close >= df_plot.open]
        down = df_plot[df_plot.close < df_plot.open]
        
        # Plot up candles (green)
        ax.bar(up.index, up.close - up.open, width, bottom=up.open,
               color='#27ae60', edgecolor='#27ae60', linewidth=1, alpha=0.8, zorder=3)
        ax.bar(up.index, up.high - up.close, width2, bottom=up.close,
               color='#27ae60', linewidth=0, zorder=3)
        ax.bar(up.index, up.low - up.open, width2, bottom=up.open,
               color='#27ae60', linewidth=0, zorder=3)
        
        # Plot down candles (red)
        ax.bar(down.index, down.open - down.close, width, bottom=down.close,
               color='#e74c3c', edgecolor='#e74c3c', linewidth=1, alpha=0.8, zorder=3)
        ax.bar(down.index, down.high - down.open, width2, bottom=down.open,
               color='#e74c3c', linewidth=0, zorder=3)
        ax.bar(down.index, down.low - down.close, width2, bottom=down.close,
               color='#e74c3c', linewidth=0, zorder=3)
        
        # Add legend for regimes
        for state in sorted(df_plot['state_mapped'].unique()):
            ax.plot([], [], 's', markersize=10, alpha=0.3,
                   color=self.config.regime_colors[state],
                   label=self.config.regime_labels[state])
    
    def _draw_volume(self, ax, df_plot):
        """Draw volume bars"""
        # Color by price movement
        colors = ['#27ae60' if c >= o else '#e74c3c' 
                  for c, o in zip(df_plot['close'], df_plot['open'])]
        
        if 'volume' in df_plot.columns:
            ax.bar(df_plot.index, df_plot['volume'], color=colors, alpha=0.5, width=0.8)
        else:
            # Use dummy volume if not available
            ax.bar(df_plot.index, [1]*len(df_plot), color=colors, alpha=0.5, width=0.8)
            ax.set_ylabel('Volume (N/A)', fontsize=12, fontweight='bold')
    
    def _draw_regime_timeline(self, ax, df_plot):
        """Draw regime state timeline"""
        colors = [self.config.regime_colors[state] for state in df_plot['state_mapped']]
        ax.scatter(df_plot.index, df_plot['state_mapped'], 
                  c=colors, s=30, alpha=0.8, marker='s')
        ax.set_ylim(-0.5, 2.5)
    
    def _plot_line_chart_regimes(self, df_analysis):
        """Fallback: Plot line chart if OHLCV not available"""
        print("  → Creating line chart...")
        
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), 
                                gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1, ax2 = axes
        
        # Limit to recent data
        df_plot = df_analysis.tail(500).copy().reset_index(drop=True)
        
        # Draw regime backgrounds
        self._draw_regime_backgrounds(ax1, df_plot)
        
        # Price line
        if 'close' in df_plot.columns:
            ax1.plot(df_plot.index, df_plot['close'], 
                    color='black', linewidth=1.5, alpha=0.8, zorder=2)
        
        # Regime timeline
        self._draw_regime_timeline(ax2, df_plot)
        
        # Formatting
        ax1.set_title('BTCUSDT H4 — Market Regime Classification',
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        for state in sorted(df_plot['state_mapped'].unique()):
            ax1.plot([], [], 's', markersize=10, alpha=0.3,
                   color=self.config.regime_colors[state],
                   label=self.config.regime_labels[state])
        ax1.legend(loc='upper left', fontsize=10)
        
        ax2.set_ylabel('Regime', fontsize=10, fontweight='bold')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Low', 'Med', 'High'])
        ax2.set_xlabel('Time (Index)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.plot_dir, 'candlestick_regimes.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    ✓ Line chart saved: {plot_path}")
        
        plt.close()
    
    def _plot_simple_dashboard(self, df_analysis, state_probs, model):
        """Plot simple dashboard with key metrics"""
        print("  → Creating summary dashboard...")
        
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Regime Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_regime_distribution(ax1, df_analysis)
        
        # 2. Transition Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_transition_matrix(ax2, model)
        
        # 3. Volatility by Regime
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_volatility_comparison(ax3, df_analysis)
        
        # 4. Return Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_return_distribution(ax4, df_analysis)
        
        # 5. Regime Duration
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_regime_duration(ax5, df_analysis)
        
        # 6. Performance Metrics
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_performance_metrics(ax6, df_analysis)
        
        fig.suptitle('Market Regime Analysis — Summary Dashboard',
                     fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.plot_dir, 'regime_dashboard.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    ✓ Dashboard saved: {plot_path}")
        
        plt.close()
    
    def _plot_regime_distribution(self, ax, df_analysis):
        """Plot regime distribution"""
        state_counts = df_analysis['state_mapped'].value_counts().sort_index()
        colors = [self.config.regime_colors[i] for i in state_counts.index]
        
        bars = ax.bar(range(len(state_counts)), state_counts.values, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add percentage labels
        total = len(df_analysis)
        for i, (idx, count) in enumerate(state_counts.items()):
            pct = count / total * 100
            ax.text(i, count + total*0.02, f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(state_counts)))
        ax.set_xticklabels(['Low Vol', 'Med Vol', 'High Vol'], fontsize=10)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Regime Distribution', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def _plot_transition_matrix(self, ax, model):
        """Plot transition matrix"""
        trans_matrix = model.transmat_
        
        im = ax.imshow(trans_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(self.config.n_states):
            for j in range(self.config.n_states):
                text = ax.text(j, i, f'{trans_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black",
                             fontsize=12, fontweight='bold')
        
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Low', 'Med', 'High'], fontsize=10)
        ax.set_yticklabels(['Low', 'Med', 'High'], fontsize=10)
        ax.set_xlabel('To State', fontsize=11, fontweight='bold')
        ax.set_ylabel('From State', fontsize=11, fontweight='bold')
        ax.set_title('State Transition Probability', fontsize=13, fontweight='bold', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=10)
    
    def _plot_volatility_comparison(self, ax, df_analysis):
        """Plot volatility comparison"""
        volatilities = []
        labels = []
        colors_list = []
        
        for state in sorted(df_analysis['state_mapped'].unique()):
            mask = df_analysis['state_mapped'] == state
            vol = df_analysis.loc[mask, 'atr_norm'].mean() * 100
            volatilities.append(vol)
            labels.append(self.config.regime_labels[state].split()[0])
            colors_list.append(self.config.regime_colors[state])
        
        bars = ax.bar(range(len(volatilities)), volatilities, 
                      color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, v in enumerate(volatilities):
            ax.text(i, v + max(volatilities)*0.02, f'{v:.3f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('ATR Normalized (%)', fontsize=11, fontweight='bold')
        ax.set_title('Average Volatility by Regime', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def _plot_return_distribution(self, ax, df_analysis):
        """Plot return distribution by regime"""
        for state in sorted(df_analysis['state_mapped'].unique()):
            mask = df_analysis['state_mapped'] == state
            returns = df_analysis.loc[mask, 'log_return'] * 100
            
            ax.hist(returns, bins=40, alpha=0.6,
                   label=self.config.regime_labels[state],
                   color=self.config.regime_colors[state],
                   edgecolor='black', linewidth=0.5)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Log Return (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Return Distribution by Regime', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_regime_duration(self, ax, df_analysis):
        """Plot average regime duration"""
        # Calculate regime changes
        df_temp = df_analysis.copy()
        df_temp['regime_change'] = (df_temp['state_mapped'] != 
                                     df_temp['state_mapped'].shift(1)).astype(int)
        df_temp['regime_id'] = df_temp['regime_change'].cumsum()
        
        regime_durations = df_temp.groupby('regime_id').agg({
            'state_mapped': 'first',
            'timestamp': 'count'
        }).rename(columns={'timestamp': 'duration'})
        
        avg_durations = []
        labels = []
        colors_list = []
        
        for state in sorted(regime_durations['state_mapped'].unique()):
            mask = regime_durations['state_mapped'] == state
            avg_dur = regime_durations.loc[mask, 'duration'].mean()
            avg_durations.append(avg_dur)
            labels.append(self.config.regime_labels[state].split()[0])
            colors_list.append(self.config.regime_colors[state])
        
        bars = ax.bar(range(len(avg_durations)), avg_durations,
                      color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, v in enumerate(avg_durations):
            ax.text(i, v + max(avg_durations)*0.02, f'{v:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Duration (4h candles)', fontsize=11, fontweight='bold')
        ax.set_title('Average Regime Duration', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    def _plot_performance_metrics(self, ax, df_analysis):
        """Plot performance metrics table"""
        # Calculate metrics per regime
        metrics = []
        
        for state in sorted(df_analysis['state_mapped'].unique()):
            mask = df_analysis['state_mapped'] == state
            
            avg_return = df_analysis.loc[mask, 'log_return'].mean() * 100
            vol = df_analysis.loc[mask, 'atr_norm'].mean() * 100
            sharpe = avg_return / vol if vol > 0 else 0
            count = mask.sum()
            pct = count / len(df_analysis) * 100
            
            metrics.append([
                self.config.regime_labels[state],
                f'{avg_return:.4f}%',
                f'{vol:.4f}%',
                f'{sharpe:.2f}',
                f'{pct:.1f}%'
            ])
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metrics,
                        colLabels=['Regime', 'Avg Return', 'Volatility', 'Sharpe', 'Frequency'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by regime
        for i, state in enumerate(sorted(df_analysis['state_mapped'].unique())):
            for j in range(5):
                table[(i+1, j)].set_facecolor(self.config.regime_colors[state])
                table[(i+1, j)].set_alpha(0.3)
        
        ax.set_title('Performance Metrics by Regime', 
                    fontsize=13, fontweight='bold', pad=20)
    
    def plot_feature_comparison(self, df_analysis):
        """Optional: Plot feature distributions - removed for simplicity"""
        pass
    
    def plot_duration_analysis(self, df_analysis):
        """Optional: Plot duration analysis - removed for simplicity"""
        pass


class ResultSaver:
    """Save HMM results"""
    
    def __init__(self, config):
        self.config = config
    
    def save_results(self, df_analysis, state_probs):
        """Save regime predictions to parquet"""
        print("\nSaving results...")
        
        # Add probability columns
        for i in range(self.config.n_states):
            df_analysis[f'prob_state_{i}'] = state_probs[:, i]
        
        # Select columns to save
        cols_to_save = ['timestamp', 'state', 'state_mapped'] + \
                       [f'prob_state_{i}' for i in range(self.config.n_states)]
        
        # Add close price if available
        if 'close' in df_analysis.columns:
            cols_to_save.insert(1, 'close')
        
        df_output = df_analysis[cols_to_save].copy()
        
        # Save
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        df_output.to_parquet(self.config.output_path, engine='pyarrow', index=False)
        
        print(f"✓ Results saved: {self.config.output_path}")


class HMMRegimeIdentifier:
    """Main HMM regime identification orchestrator"""
    
    def __init__(self, config=None):
        self.config = config or HMMConfig()
        self.loader = DataLoader(self.config)
        self.trainer = HMMTrainer(self.config)
        self.analyzer = StateAnalyzer(self.config)
        self.visualizer = RegimeVisualizer(self.config)
        self.saver = ResultSaver(self.config)
    
    def run(self):
        """Execute complete HMM regime identification pipeline"""
        print("=" * 60)
        print("HMM MARKET REGIME IDENTIFICATION")
        print("=" * 60)
        
        # Load and prepare data
        df, X_scaled, scaler = self.loader.load_and_prepare()
        
        # Train HMM
        model, hidden_states, state_probs = self.trainer.train(X_scaled)
        
        # Analyze states
        df_analysis, state_mapping = self.analyzer.analyze_states(df, hidden_states)
        
        # Visualizations (simplified)
        self.visualizer.plot_comprehensive_analysis(df_analysis, state_probs, model)
        
        # Save results
        self.saver.save_results(df_analysis, state_probs)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nGenerated visualizations:")
        print("  1. candlestick_regimes.png - Main chart with regime overlay")
        print("  2. regime_dashboard.png - Summary metrics and statistics")
        
        return df_analysis, model, state_probs


def main():
    """Main execution function"""
    identifier = HMMRegimeIdentifier()
    df_analysis, model, state_probs = identifier.run()
    return df_analysis, model, state_probs


if __name__ == "__main__":
    main()