# HMM Market Regime Identification

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/Package_Manager-UV-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

Hidden Markov Model untuk identifikasi regime pasar cryptocurrency menggunakan data BTCUSDT timeframe 4 jam.

---

## Tujuan Program

Program ini dirancang untuk:

1. Mengidentifikasi regime pasar (Low / Medium / High Volatility)
2. Menganalisis probabilitas transisi antar regime
3. Mengukur karakteristik statistik setiap regime
4. Menyediakan visualisasi berbasis data untuk analisis trading

---

## Metodologi

### Hidden Markov Model (HMM)

Model yang digunakan adalah **Gaussian Hidden Markov Model** dengan konfigurasi:

- Jumlah hidden state: 3
- Covariance type: `full`
- Iterasi maksimum: 500
- Input features: 6 indikator teknikal

### Features

| Feature      | Deskripsi                        |
| ------------ | -------------------------------- |
| `log_return` | Log return harga                 |
| `atr_norm`   | ATR ternormalisasi               |
| `adx_14`     | Kekuatan tren                    |
| `slope`      | Kemiringan harga                 |
| `volume_z`   | Z-score volume                   |
| `efficiency` | Rasio efisiensi pergerakan harga |

---

## Ringkasan Insight

### Regime Volatilitas

#### Low Volatility

- Frekuensi: 37.2%
- ATR rata-rata: 1.095%
- Return rata-rata: -0.0125%
- Sharpe Ratio: -0.01
- Durasi rata-rata: 26.9 candle

Karakteristik utama:

- Pasar cenderung sideways
- Aktivitas trading tidak optimal
- Regime sangat stabil

Strategi:

- Range trading terbatas
- Menunggu breakout
- Menghindari trend following

---

#### Medium Volatility

- Frekuensi: 32.0%
- ATR rata-rata: 1.320%
- Return rata-rata: 0.0101%
- Sharpe Ratio: -0.1
- Durasi rata-rata: 13.6 candle

Karakteristik utama:

- Risk-adjusted return terbaik
- Volatilitas cukup untuk trend

Strategi:

- Trend following
- Breakout trading
- Swing trading

---

#### High Volatility

- Frekuensi: 30.9%
- ATR rata-rata: 1.774%
- Return rata-rata: 0.0932%
- Sharpe Ratio: 0.05
- Durasi rata-rata: 13.6 candle

Karakteristik utama:

- Fluktuasi harga ekstrem
- Risiko tinggi dengan Sharpe negatif

Strategi:

- Mengurangi ukuran posisi
- Stop loss lebih lebar
- Fokus pada manajemen risiko

---

## Transition Matrix

Setiap regime bersifat sangat persisten.

```
Low    -> Low    : 95%
Low    -> Medium : 3%
Low    -> High   : 2%

Medium -> Medium: 94%
Medium -> Low   : 4%
Medium -> High  : 2%

High   -> High  : 95%
High   -> Low   : 3%
High   -> Medium: 2%
```

Implikasi:

- Perubahan regime jarang terjadi
- Transisi biasanya bertahap
- High volatility bersifat sementara

---

## Regime Duration

| Regime | Durasi Rata-rata | Hari |
| ------ | ---------------- | ---- |
| Low    | 22.7 candle      | 3.8  |
| Medium | 22.5 candle      | 3.8  |
| High   | 16.6 candle      | 2.8  |

---

## Aturan Trading Berbasis Regime

### Position Sizing

```python
if regime == "Low":
    position_size = base_size * 0.3
elif regime == "Medium":
    position_size = base_size * 1.5
elif regime == "High":
    position_size = base_size * 0.5
```

### Transition-Based Decision

```python
if prev == "Low" and curr == "Medium":
    signal = "BUY"
elif prev == "Medium" and curr == "High":
    signal = "REDUCE"
elif prev == "High" and curr == "Medium":
    signal = "RE-ENTER"
```

---

## Quick Start

### Instalasi UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

### Instalasi Dependensi

```bash
uv pip install -r requirements.txt
```

### Menjalankan Program

```bash
uv run main.py
```

---

## Struktur Proyek

```
.
├── data
│   ├── feature_engineering.py
│   ├── fetch_data.py
│   ├── preprocessing.py
│   ├── processed
│   │   ├── btcusdt_h4_features_clean.parquet
│   │   ├── btcusdt_h4_features.parquet
│   │   ├── btcusdt_h4_features_scaled.npy
│   │   ├── btcusdt_h4_regimes.parquet
│   │   └── scaler.pkl
│   └── raw
│       └── btcusdt_h4.parquet
├── main.py
├── model
│   └── hmm_regime.py
├── outputs
│   ├── correlation_matrix.png
│   └── plots
│       ├── candlestick_regimes.png
│       └── regime_dashboard.png
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

---

## Output

- `data/processed/btcusdt_h4_regimes.parquet`
- `outputs/plots/candlestick_regimes.png`
- `outputs/plots/regime_dashboard.png`

---

## Konfigurasi Model

```python
self.n_states = 3
self.covariance_type = "full"
self.n_iter = 500
self.random_state = 42
```

---

## Keterbatasan

- Model mendeteksi regime saat ini, bukan memprediksi masa depan
- Terdapat lag saat regime berubah
- Sensitif terhadap pemilihan feature

---

## Disclaimer

Proyek ini ditujukan untuk riset dan edukasi. Bukan saran investasi.
