
# 📘 Project Overview: Temporal Feature Fusion for Financial Time-Series

---

## 📌 Project Objective

This project aims to:
1. **Replicate a high-impact journal paper (CNNpred)**
2. **Modify its architecture** to fix fundamental mathematical flaws.
3. **Introduce a meaningful novelty** (Regime-Aware Labeling).
4. **Implement and evaluate the improved model rigorously** against strict quantitative baselines.

This follows strict academic requirements:
* Must use a **journal article (not conference)**.
* Must ensure **high impact factor**.
* Must involve **CNN-based architecture**.
* Must include **architecture modification + novelty**.

---

## 📄 Base Paper Details

* **Title:** CNNpred: CNN-based stock market prediction using several data sources
* **Authors:** Hoseinzade & Haratizadeh
* **Year:** 2019
* **Journal:** Expert Systems with Applications
* **Impact Factor:** ~8.5

---

## 🎯 Original Problem & Limitations (The Research Gap)

The base paper attempts to predict next-day stock market direction (Up/Down) using technical indicators. However, it suffers from three critical flaws:

1.  **Spatial Incoherence (Incorrect Input Representation):** Uses a 2D matrix (days × features) and applies 2D-CNNs like image processing. This falsely assumes spatial relationships between unrelated financial features (e.g., RSI and Volume).
2.  **Naive Labeling:** Uses a binary label (Up if Close(t+1) > Close(t)). This lacks a threshold, making the model highly sensitive to market noise, and lacks a "Neutral" class.
3.  **Weak Validation:** Relies on a static 60/20/20 split, failing to test the model's robustness across changing market regimes (e.g., bull markets vs. crashes).

---

## 🚀 Proposed Improvements

1.  **Temporal Reformulation:** Replace the 2D-CNN with a **1D-CNN**, treating the financial data strictly as a time-series signal to extract local temporal shocks.
2.  **Attention Mechanism:** Add a **Transformer Encoder (Self-Attention)** block after the CNN to capture long-range historical dependencies and global market context.
3.  **Improved Labeling:** Introduce a **Dynamic Volatility Threshold** with a 3-class system (Down / Neutral / Up) to filter out market noise.
4.  **Robust Evaluation:** Implement **Anchored Walk-Forward Validation** to evaluate the model across expanding historical market regimes.

---

## 🧠 Data Representation

### Input Shape
X ∈ ℝ(C, T)
* **C** = Number of features (e.g., 7)
* **T** = Time window / Lookback period (e.g., 20 days)

### Features (Strictly Stationary)
* Log returns of OHLC (NOT raw prices)
* Rolling volatility (20-day standard deviation)
* Normalized technical indicators (RSI, MACD)

---

## 🏷️ Labeling Strategy (Regime-Aware)

### Future Return Target
R_t = log(P_{t+k} / P_t)

### Dynamic Threshold
ε_t = λ × σ(t-30, t)
*(Where λ is found via grid-search on training data only to balance the neutral class).*

### Classes
* **Down (0):** R_t < -ε_t
* **Neutral (1):** -ε_t ≤ R_t ≤ ε_t
* **Up (2):** R_t > ε_t

---

## ⚠️ Data Leakage Prevention (Strict Guidelines)

### Alignment Rule
* **Features:** Data from [t−L → t]
* **Target:** Derived from [t → t+k]
* **Safeguard:** Features and labels are explicitly index-matched. The last `k` samples of the dataset are dropped to prevent target leakage.

### Normalization Rule
* Fit the Z-score scaler **ONLY** on the training fold.
* Apply the fitted scaler to the test fold.
* NEVER fit or `fit_transform` on validation/test data.

---

## 🧩 Model Architecture

1.  **1D-CNN Block:** Temporal convolutions to capture short-term patterns, momentum bursts, and local volatility shocks.
2.  **Transformer Block:** Applies Multi-Head Self-Attention (with positional encoding) to the CNN feature maps to weigh the importance of distant historical events.
3.  **Classification Head:** Global Average Pooling followed by Fully Connected layers outputting probabilities for {Down, Neutral, Up}.

---

## 🔁 Validation Strategy

* **Method:** Anchored Walk-Forward Validation.
* **Logic:** The training set is expanding. It starts at index 0 and grows with each fold, allowing the model to retain long-term historical memory of market regimes.
* **Constraints:** No shuffling. No standard K-Fold cross-validation.

---

## 📊 Baselines & Evaluation

### Mandatory Baselines
1.  **Dummy Classifier** (Majority-class predictor — Lower Bound)
2.  **XGBoost** (Classical ML baseline)
3.  **LSTM** (Standard Deep Learning baseline)
4.  **1D-CNN** (Ablation study — proposed model without attention)

### Evaluation Metrics
* **Primary:** Macro F1-score, MCC (Matthews Correlation Coefficient).
* **Secondary:** Confusion matrix, Minority class (Down) recall.
* *Note: Accuracy is reported as a reference but is not the primary measure of success.*

### Regime-Aware Analysis
* Test fold performance is evaluated based on the median volatility of the test period.
* Metrics are reported separately for **Low Volatility** and **High Volatility** regimes.

---

## 🔍 Interpretability
* Extract Attention Weights from the Transformer Encoder.
* Generate heatmaps overlaying attention focus against the lookback window.
* Determine if the model relies on recent momentum (Days 15-20) or long-term history (Days 1-10) to make decisions.

---

## ⚙️ Implementation Details (Tensor Tracking)

| Stage | Tensor Shape | Description |
| :--- | :--- | :--- |
| **Input** | `(N, C, T)` | Batch, Channels (Features), Time |
| **CNN Output** | `(N, C', T)` | Batch, Extracted Channels, Time |
| **Transformer Input** | `(T, N, C')` | Requires `.permute(2, 0, 1)` |
| **Transformer Output**| `(T, N, C')` | Context-aware sequence |
| **Classifier Input** | `(N, C')` | Requires `.mean(dim=0)` pooling |

---

## 🚫 Scope & Boundaries

**This project is:**
✔ A methodological improvement on data representation.
✔ An empirical study on Deep Learning robustness in finance.
✔ A rigorously evaluated, leak-free pipeline.

**This project is NOT:**
❌ A new architecture invention from scratch.
❌ A live trading system or automated bot.
❌ Making claims about financial profitability or Sharpe Ratios.

---
**Status:** Architecture Designed. Data Pipeline Built. Ready for Final Hyperparameter Tuning and Execution.
