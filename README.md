# Hawkes–Merton Contagion Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

A comprehensive hybrid credit risk model that combines **structural Merton framework** with **Hawkes jump‑diffusion** and **contagion** to quantify systemic risk in portfolios of up to 50 firms. Designed for quantitative analysts, risk managers, and researchers who need a realistic, fast, and extensible tool for credit portfolio stress‑testing and CDO/CLO valuation.

**Key features:**
- **KMV calibration** – infer asset value and volatility from market equity data.
- **Stochastic asset dynamics** – GBM or Heston (optional) with **jump‑diffusion** (Zhou, 2001).
- **Dynamic default barrier** – mean‑reverting leverage (Collin‑Dufresne & Goldstein, 2001).
- **Hawkes contagion** – self‑exciting default intensity with mutual excitation.
- **Incomplete information** – noisy barrier observation (Duffie & Lando, 2001).
- **Regime switching** – Markov‑switching between normal and stress states (optional).
- **Stochastic recovery** – recovery rate negatively correlated with default intensity.
- **Parallel Monte Carlo** – sequential or vectorised simulation (fast).
- **Interactive visualisation** – Plotly charts for asset paths, intensities, and default heatmaps.
- **Full analytics** – VaR, CVaR, EL/UL, scenario analysis, sector aggregation, tornado sensitivity, and portfolio optimisation.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/hawkes-merton-contagion.git
cd hawkes-merton-contagion
pip install -r requirements.txt
```

Requirements
* Python ≥ 3.8
* NumPy, SciPy, Pandas
* Matplotlib, Plotly
* tqdm, yfinance, pandas-datareader
* (optional) Numba for speed

## Outputs & Interpretation

 a. VaR (99%) – typical $1.0–$2.5M for a $50M portfolio of IG names.
 b. CVaR (99%) – $1.2–$3.0M, depending on contagion/jump parameters.
 c. Default probability – ~40% that at least one firm defaults within 1 year.
 d. Distance‑to‑Default – top risky names: ORCL, TSLA, AVGO, etc.
 e. Implicit credit spread – ~5000 bps, which is structurally conservative; use as relative metric.

📚 Theoretical Background
  1. Merton (1974) – structural model of default.
  2. KMV – iterative calibration of asset value and volatility.
  3. Zhou (2001) – jump‑diffusion for short‑term spreads.
  4. Collin‑Dufresne & Goldstein (2001) – dynamic leverage barrier.
  5. Duffie & Lando (2001) – incomplete information.
  6. Hawkes (1971) – self‑exciting point processes for contagion.

.
├── hawkes_merton_model.py      # Main model class and utilities
├── data_loader.py             # MarketDataExtractor50
├── requirements.txt
├── README.md
├── LICENSE
└── notebooks/
    └── demo.ipynb             # Interactive walkthrough

```bash
python hawkes_merton_model.py
```
🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📄 License
Distributed under the MIT License. See LICENSE for more information.

📧 Contact
Ognjen Raketic – ognjen.raketic@gmail.com

⭐ Acknowledgements
FRED (Federal Reserve Economic Data) for treasury and corporate bond yields.
Yahoo Finance for equity data.
Open‑source contributors of NumPy, SciPy, Pandas, Plotly, and tqdm.
