"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MARKET REGIME DETECTION SYSTEM — Full Implementation               ║
║          NSE/BSE Indian Stock Market | Resume-Grade ML Project              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Techniques Used:
  - Hidden Markov Models (HMM) via hmmlearn
  - KMeans Clustering (unsupervised)
  - Technical Indicators: ATR, Bollinger Bands, RSI, MACD
  - Regime: Bull 📈 | Bear 📉 | High Volatility ⚡ | Sideways ➖

Author  : You
Dataset : NSE/BSE via yfinance (NIFTY 50 Index)
"""

# ─────────────────────────────────────────────
# 0. DEPENDENCIES
# ─────────────────────────────────────────────
# pip install yfinance hmmlearn scikit-learn pandas numpy matplotlib seaborn

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta

# Data
import yfinance as yf

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM

# ─────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────

def fetch_nse_data(ticker="^NSEI", start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    
    print(f"[DATA] Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    # ✅ FIX: Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    print(f"[DATA] Downloaded {len(df)} rows.\n")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_features(df):
    """
    Compute a rich feature set for regime detection:
      - Returns (1d, 5d, 20d)
      - Volatility (rolling std)
      - ATR (Average True Range)
      - Bollinger Band Width
      - RSI (Relative Strength Index)
      - MACD & Signal Line
      - Trend Strength (ADX-like ratio)
    """
    data = df.copy()
    close = data["Close"].squeeze()
    high  = data["High"].squeeze()
    low   = data["Low"].squeeze()

    # ── Returns ──────────────────────────────
    data["return_1d"]  = close.pct_change(1)
    data["return_5d"]  = close.pct_change(5)
    data["return_20d"] = close.pct_change(20)

    # ── Volatility ───────────────────────────
    data["vol_10d"]  = data["return_1d"].rolling(10).std()
    data["vol_20d"]  = data["return_1d"].rolling(20).std()
    data["vol_60d"]  = data["return_1d"].rolling(60).std()
    data["vol_ratio"] = data["vol_10d"] / data["vol_60d"]  # short vs long vol

    # ── ATR (Average True Range) ─────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    data["ATR_14"] = tr.rolling(14).mean()
    data["ATR_pct"] = data["ATR_14"] / close  # normalised ATR

    # ── Bollinger Bands ──────────────────────
    bb_window = 20
    bb_mean = close.rolling(bb_window).mean()
    bb_std  = close.rolling(bb_window).std()
    data["BB_upper"] = bb_mean + 2 * bb_std
    data["BB_lower"] = bb_mean - 2 * bb_std
    data["BB_width"] = (data["BB_upper"] - data["BB_lower"]) / bb_mean
    data["BB_pct"]   = (close - data["BB_lower"]) / (data["BB_upper"] - data["BB_lower"])

    # ── RSI (14-period) ──────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    data["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"]        = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"]   = data["MACD"] - data["MACD_signal"]

    # ── Trend (SMA ratios) ───────────────────
    data["SMA_50"]  = close.rolling(50).mean()
    data["SMA_200"] = close.rolling(200).mean()
    data["trend_strength"] = (data["SMA_50"] - data["SMA_200"]) / data["SMA_200"]

    data.dropna(inplace=True)
    print(f"[FEATURES] Computed {len(data.columns)} columns. Rows after dropna: {len(data)}")
    return data


# ─────────────────────────────────────────────
# 3. FEATURE SELECTION FOR MODELS
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "vol_10d", "vol_20d", "vol_ratio",
    "ATR_pct",
    "BB_width", "BB_pct",
    "RSI",
    "MACD_hist",
    "trend_strength"
]

def prepare_features(data, feature_cols=FEATURE_COLS):
    X = data[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ─────────────────────────────────────────────
# 4. MODEL A — KMEANS CLUSTERING
# ─────────────────────────────────────────────

def find_optimal_k(X_scaled, k_range=range(2, 8)):
    """Elbow + Silhouette to pick best K."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    
    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"[KMEANS] Best K by silhouette = {best_k}  (score={max(sil_scores):.3f})")
    return best_k, inertias, sil_scores, list(k_range)


def fit_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)
    print(f"[KMEANS] Fitted {n_clusters} clusters. Inertia = {km.inertia_:.2f}")
    return km, labels


def interpret_kmeans_regimes(data, labels, feature_cols=FEATURE_COLS):
    """
    Auto-label clusters as regimes based on cluster centroids:
    """
    data = data.copy()
    data["cluster"] = labels

    # ✅ FIX: keep inside function
    feature_cols = [col for col in feature_cols if col in data.columns]

    summary = data.groupby("cluster")[feature_cols].mean(numeric_only=True)

    regime_map = {}

    for cluster_id, row in summary.iterrows():
        ret   = row["return_20d"]
        vol   = row["vol_20d"]
        trend = row["trend_strength"]

        if vol > summary["vol_20d"].quantile(0.75):
            regime_map[cluster_id] = "High Volatility ⚡"
        elif ret > summary["return_20d"].quantile(0.6) and trend > 0:
            regime_map[cluster_id] = "Bull Market 📈"
        elif ret < summary["return_20d"].quantile(0.4) and trend < 0:
            regime_map[cluster_id] = "Bear Market 📉"
        else:
            regime_map[cluster_id] = "Sideways Market ➖"

    data["regime_kmeans"] = data["cluster"].map(regime_map)

    return data, regime_map


# ─────────────────────────────────────────────
# 5. MODEL B — HIDDEN MARKOV MODEL (HMM)
# ─────────────────────────────────────────────

def fit_hmm(X_scaled, n_states=4, n_iter=200):
    """
    Fit a Gaussian HMM. States correspond to market regimes.
    Uses GaussianHMM from hmmlearn.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42
    )
    model.fit(X_scaled)
    hidden_states = model.predict(X_scaled)
    log_likelihood = model.score(X_scaled)

    print(f"\n[HMM] Fitted {n_states}-state HMM.")
    print(f"[HMM] Log-Likelihood = {log_likelihood:.2f}")
    print(f"[HMM] Transition Matrix:\n{np.round(model.transmat_, 3)}")
    return model, hidden_states


def interpret_hmm_regimes(data, hidden_states, feature_cols=FEATURE_COLS):
    """Auto-label HMM states using same logic as KMeans interpretation."""

    data = data.copy()
    data["hmm_state"] = hidden_states

    # ✅ keep inside function
    feature_cols = [col for col in feature_cols if col in data.columns]

    summary = data.groupby("hmm_state")[feature_cols].mean(numeric_only=True)

    regime_map = {}

    for state_id, row in summary.iterrows():
        ret   = row["return_20d"]
        vol   = row["vol_20d"]
        trend = row["trend_strength"]

        if vol > summary["vol_20d"].quantile(0.75):
            regime_map[state_id] = "High Volatility ⚡"
        elif ret > summary["return_20d"].quantile(0.6) and trend > 0:
            regime_map[state_id] = "Bull Market 📈"
        elif ret < summary["return_20d"].quantile(0.4) and trend < 0:
            regime_map[state_id] = "Bear Market 📉"
        else:
            regime_map[state_id] = "Sideways Market ➖"

    data["regime_hmm"] = data["hmm_state"].map(regime_map)

    print("\n[HMM] State → Regime mapping:")
    for k, v in regime_map.items():
        print(f"  State {k} → {v}")

    return data, regime_map


# ─────────────────────────────────────────────
# 6. ENSEMBLE: COMBINE HMM + KMEANS
# ─────────────────────────────────────────────

def ensemble_regime(data):
    """
    Simple majority-vote ensemble between HMM and KMeans.
    If they agree → high confidence. Else → flag as uncertain.
    """
    def vote(row):
        if row["regime_hmm"] == row["regime_kmeans"]:
            return row["regime_hmm"]
        # Fallback: prefer HMM (probabilistic, more theoretically sound)
        return row["regime_hmm"] + " *"

    data["regime_final"] = data.apply(vote, axis=1)
    agreement_rate = (data["regime_hmm"] == data["regime_kmeans"]).mean()
    print(f"\n[ENSEMBLE] HMM ↔ KMeans agreement rate: {agreement_rate:.1%}")
    return data


# ─────────────────────────────────────────────
# 7. BACKTESTING: REGIME-AWARE STRATEGY
# ─────────────────────────────────────────────

def backtest_regime_strategy(data):
    """
    Simple regime-aware strategy:
      - Bull Market    → Long (invested)
      - Bear Market    → Short or Cash
      - High Volatility→ Reduce position (50%)
      - Sideways       → Cash / flat
    """
    regime_to_position = {
        "Bull Market 📈":        1.0,
        "Bear Market 📉":       -0.5,
        "High Volatility ⚡":    0.5,
        "Sideways Market ➖":    0.0,
    }

    data = data.copy()
    # Clean regime labels (remove " *" from uncertain)
    data["regime_clean"] = data["regime_final"].str.replace(r" \*", "", regex=True)
    data["position"] = data["regime_clean"].map(regime_to_position).fillna(0)

    data["strategy_return"] = data["position"].shift(1) * data["return_1d"]
    data["buy_hold_return"]  = data["return_1d"]

    data["cum_strategy"]  = (1 + data["strategy_return"]).cumprod()
    data["cum_buy_hold"]  = (1 + data["buy_hold_return"]).cumprod()

    # ── Performance Metrics ──────────────────
    n_years = len(data) / 252

    strat_cagr = (data["cum_strategy"].iloc[-1] ** (1/n_years)) - 1
    bh_cagr    = (data["cum_buy_hold"].iloc[-1] ** (1/n_years)) - 1

    strat_sharpe = (data["strategy_return"].mean() / data["strategy_return"].std()) * np.sqrt(252)
    bh_sharpe    = (data["buy_hold_return"].mean() / data["buy_hold_return"].std()) * np.sqrt(252)

    strat_dd = (data["cum_strategy"] / data["cum_strategy"].cummax() - 1).min()
    bh_dd    = (data["cum_buy_hold"] / data["cum_buy_hold"].cummax() - 1).min()

    metrics = {
        "Strategy": {
            "CAGR":        f"{strat_cagr:.1%}",
            "Sharpe":      f"{strat_sharpe:.2f}",
            "Max Drawdown":f"{strat_dd:.1%}",
            "Final Return":f"{(data['cum_strategy'].iloc[-1]-1):.1%}"
        },
        "Buy & Hold": {
            "CAGR":        f"{bh_cagr:.1%}",
            "Sharpe":      f"{bh_sharpe:.2f}",
            "Max Drawdown":f"{bh_dd:.1%}",
            "Final Return":f"{(data['cum_buy_hold'].iloc[-1]-1):.1%}"
        }
    }

    print("\n" + "═"*50)
    print("  BACKTEST RESULTS")
    print("═"*50)
    for model_name, m in metrics.items():
        print(f"\n  {model_name}:")
        for k, v in m.items():
            print(f"    {k:<15} {v}")
    print("═"*50)
    return data, metrics


# ─────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────

REGIME_COLORS = {
    "Bull Market 📈":       "#00C896",
    "Bear Market 📉":       "#FF4D6D",
    "High Volatility ⚡":   "#FFB830",
    "Sideways Market ➖":   "#7B8CDE",
}

def regime_to_color(regime_str):
    for key, color in REGIME_COLORS.items():
        if key.split()[0] in regime_str:
            return color
    return "#888888"


def plot_full_dashboard(data, metrics, ticker="NIFTY 50"):
    """Generate a comprehensive 6-panel dashboard."""
    fig = plt.figure(figsize=(22, 20), facecolor="#0D0D1A")
    fig.suptitle(
        f"Market Regime Detection System — {ticker}",
        fontsize=22, fontweight="bold", color="white", y=0.98
    )

    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35)

    dark_bg   = "#0D0D1A"
    panel_bg  = "#13132B"
    text_col  = "#E8E8FF"
    grid_col  = "#2A2A4A"

    def style_ax(ax, title):
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color=text_col, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=text_col, labelsize=8)
        ax.spines[:].set_color(grid_col)
        ax.xaxis.label.set_color(text_col)
        ax.yaxis.label.set_color(text_col)
        ax.grid(axis="y", color=grid_col, linewidth=0.5, alpha=0.6)

    close = data["Close"].squeeze()

    # ── Panel 1: Price + Regime Background ───
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "📊 NIFTY 50 Price with Detected Regimes")

    ax1.plot(data.index, close, color="#A0A8FF", linewidth=1.2, zorder=3)

    current_regime = None
    start_idx = data.index[0]
    for i, (idx, row) in enumerate(data.iterrows()):
        regime = row["regime_final"].replace(" *", "")
        if regime != current_regime:
            if current_regime is not None:
                ax1.axvspan(start_idx, idx,
                            alpha=0.22,
                            color=regime_to_color(current_regime),
                            zorder=2)
            current_regime = regime
            start_idx = idx
    ax1.axvspan(start_idx, data.index[-1],
                alpha=0.22,
                color=regime_to_color(current_regime), zorder=2)

    legend_patches = [
        mpatches.Patch(color=c, label=r, alpha=0.7)
        for r, c in REGIME_COLORS.items()
    ]
    ax1.legend(handles=legend_patches, loc="upper left",
               framealpha=0.3, facecolor=panel_bg, labelcolor=text_col, fontsize=8)
    ax1.set_ylabel("Price (₹)", color=text_col)

    # ── Panel 2: HMM Regime Probability ──────
    ax2 = fig.add_subplot(gs[1, :2])
    style_ax(ax2, "🔮 HMM Hidden State over Time")
    regime_numeric = data["hmm_state"].astype(float)
    ax2.fill_between(data.index, regime_numeric, alpha=0.6,
                     color="#7B8CDE", linewidth=0)
    ax2.plot(data.index, regime_numeric, color="#A0A8FF", linewidth=0.8)
    ax2.set_ylabel("HMM State", color=text_col)

    # ── Panel 3: Regime Distribution Pie ─────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(panel_bg)
    ax3.set_title("📈 Regime Distribution", color=text_col, fontsize=11, fontweight="bold", pad=8)
    regime_counts = data["regime_final"].str.replace(" \*", "", regex=True).value_counts()
    colors_pie = [regime_to_color(r) for r in regime_counts.index]
    wedges, texts, autotexts = ax3.pie(
        regime_counts.values,
        labels=[r.split("(")[0].strip() for r in regime_counts.index],
        colors=colors_pie,
        autopct="%1.1f%%",
        textprops={"color": text_col, "fontsize": 8},
        wedgeprops={"linewidth": 1.5, "edgecolor": dark_bg}
    )
    for at in autotexts:
        at.set_color(dark_bg)
        at.set_fontweight("bold")

    # ── Panel 4: Volatility (ATR) ─────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, "⚡ ATR % (Normalised Volatility)")
    ax4.fill_between(data.index, data["ATR_pct"]*100, alpha=0.7,
                     color="#FFB830", linewidth=0)
    ax4.set_ylabel("ATR %", color=text_col)

    # ── Panel 5: Bollinger Band Width ─────────
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5, "〰️ Bollinger Band Width")
    ax5.fill_between(data.index, data["BB_width"]*100, alpha=0.7,
                     color="#FF6B9D", linewidth=0)
    ax5.set_ylabel("BB Width %", color=text_col)

    # ── Panel 6: RSI ──────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    style_ax(ax6, "📉 RSI (14-Period)")
    ax6.plot(data.index, data["RSI"], color="#00C896", linewidth=0.8)
    ax6.axhline(70, color="#FF4D6D", linewidth=0.8, linestyle="--", alpha=0.7)
    ax6.axhline(30, color="#00C896", linewidth=0.8, linestyle="--", alpha=0.7)
    ax6.fill_between(data.index, data["RSI"], 50, alpha=0.15,
                     where=data["RSI"] >= 50, color="#00C896")
    ax6.fill_between(data.index, data["RSI"], 50, alpha=0.15,
                     where=data["RSI"] < 50, color="#FF4D6D")
    ax6.set_ylim(0, 100)
    ax6.set_ylabel("RSI", color=text_col)

    # ── Panel 7: Strategy vs Buy & Hold ──────
    ax7 = fig.add_subplot(gs[3, :2])
    style_ax(ax7, "💰 Regime Strategy vs Buy & Hold")
    ax7.plot(data.index, data["cum_strategy"],
             color="#00C896", linewidth=1.5, label="Regime Strategy")
    ax7.plot(data.index, data["cum_buy_hold"],
             color="#7B8CDE", linewidth=1.5, label="Buy & Hold", linestyle="--")
    ax7.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=9,
               framealpha=0.4)
    ax7.set_ylabel("Cumulative Return", color=text_col)
    ax7.set_xlabel("Date", color=text_col)

    # ── Panel 8: Metrics Table ────────────────
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.set_facecolor(panel_bg)
    ax8.set_title("📋 Performance Metrics", color=text_col, fontsize=11,
                  fontweight="bold", pad=8)
    ax8.axis("off")

    rows = []
    for model_name, m in metrics.items():
        for metric, val in m.items():
            rows.append([model_name, metric, val])

    table_data = [["Model", "Metric", "Value"]] + rows
    table = ax8.table(
        cellText=rows,
        colLabels=["Model", "Metric", "Value"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1A1A3A" if r % 2 == 0 else panel_bg)
        cell.set_text_props(color=text_col)
        cell.set_edgecolor(grid_col)
        if r == 0:
            cell.set_facecolor("#2A2A5A")
            cell.set_text_props(fontweight="bold", color=text_col)

    plt.savefig("regime_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=dark_bg)
    print("\n[PLOT] Dashboard saved → regime_dashboard.png")
    plt.show()


# ─────────────────────────────────────────────
# 9. CURRENT REGIME REPORT
# ─────────────────────────────────────────────

def print_current_regime(data):
    latest = data.iloc[-1]
    print("\n" + "═"*50)
    print("  📊 CURRENT MARKET REGIME REPORT")
    print("═"*50)
    print(f"  Date          : {data.index[-1].date()}")
    print(f"  KMeans Regime : {latest['regime_kmeans']}")
    print(f"  HMM    Regime : {latest['regime_hmm']}")
    print(f"  FINAL  Regime : {latest['regime_final']}")
    print(f"  RSI           : {latest['RSI']:.1f}")
    print(f"  BB Width      : {latest['BB_width']*100:.2f}%")
    print(f"  ATR %         : {latest['ATR_pct']*100:.2f}%")
    print(f"  20d Return    : {latest['return_20d']*100:.2f}%")
    print(f"  Trend Strength: {latest['trend_strength']*100:.2f}%")
    print("═"*50)


# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(ticker="^NSEI", start="2015-01-01"):
    print("╔" + "═"*60 + "╗")
    print("║  MARKET REGIME DETECTION SYSTEM — NIFTY 50 (NSE)          ║")
    print("╚" + "═"*60 + "╝\n")

    # Step 1: Data
    raw_df = fetch_nse_data(ticker=ticker, start=start)

    # Step 2: Features
    data = compute_features(raw_df)

    # Step 3: Prepare scaled features
    X_scaled, scaler = prepare_features(data)

    # Step 4: Optimal K
    best_k, inertias, sil_scores, k_range = find_optimal_k(X_scaled)

    # Step 5: KMeans
    km_model, km_labels = fit_kmeans(X_scaled, n_clusters=best_k)
    data, km_regime_map = interpret_kmeans_regimes(data, km_labels)

    # Step 6: HMM
    hmm_model, hmm_states = fit_hmm(X_scaled, n_states=best_k)
    data, hmm_regime_map = interpret_hmm_regimes(data, hmm_states)

    # Step 7: Ensemble
    data = ensemble_regime(data)

    # Step 8: Backtest
    data, metrics = backtest_regime_strategy(data)

    # Step 9: Current regime
    print_current_regime(data)

    # Step 10: Visualise
    plot_full_dashboard(data, metrics, ticker=ticker)

    return data, km_model, hmm_model, scaler


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data, km_model, hmm_model, scaler = run_pipeline(
        ticker="^NSEI",     # Change to ^BSESN for SENSEX, or RELIANCE.NS etc.
        start="2015-01-01"
    )

    # ── Save data ──────────────────────────────
    output_cols = [
        "Close", "return_1d", "vol_20d", "ATR_pct",
        "RSI", "BB_width", "MACD_hist", "trend_strength",
        "regime_kmeans", "regime_hmm", "regime_final"
    ]
    data[output_cols].to_csv("regime_output.csv")
    print("\n[SAVED] regime_output.csv")