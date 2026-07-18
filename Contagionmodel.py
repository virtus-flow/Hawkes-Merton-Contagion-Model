# ================================================================
# HAWKES-MERTON CONTAGION - KOMPLETAN ISPRAVLJEN KOD (SVE ANALIZE)
# ================================================================

!pip install -q yfinance plotly tqdm requests pandas-datareader

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
import os
import yfinance as yf
import requests
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---------- 1. Pomoćne funkcije ----------
def nearest_positive_definite(A, epsilon=1e-8):
    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < epsilon] = epsilon
    A_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(A_corr))
    A_corr = A_corr / np.outer(d, d)
    return A_corr

def fetch_fred_series(series_id, start_date='2020-01-01', end_date=None):
    try:
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        series = pdr.DataReader(series_id, 'fred', start_date, end_date)
        return series[series_id]
    except:
        return pd.Series()

# ---------- 2. Ekstraktor podataka za 50 firmi ----------
class MarketDataExtractor50:
    @staticmethod
    def get_sp500_tickers(n=50):
        sp500_top = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM',
            'V', 'TSLA', 'XOM', 'UNH', 'PG', 'MA', 'JNJ', 'HD', 'COST', 'MRK',
            'ABBV', 'WMT', 'BAC', 'CRM', 'CVX', 'NFLX', 'ADBE', 'KO', 'PEP', 'TMO',
            'LIN', 'DIS', 'ORCL', 'CSCO', 'MCD', 'ACN', 'IBM', 'ABT', 'CAT', 'GE',
            'DHR', 'VZ', 'NOW', 'GS', 'PM', 'SPGI', 'QCOM', 'RTX', 'TXN', 'NEE'
        ]
        return sp500_top[:n]
    
    @staticmethod
    def fetch_equity_data(tickers, start_date, end_date):
        data = {}
        stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = stock_data
                else:
                    df = stock_data[ticker]
                if df.empty:
                    continue
                close = df['Close']
                info = yf.Ticker(ticker).info
                market_cap = info.get('marketCap')
                if market_cap is None:
                    shares = info.get('sharesOutstanding')
                    if shares is not None:
                        market_cap = close.iloc[-1] * shares
                    else:
                        market_cap = None
                returns = np.log(close / close.shift(1)).dropna()
                volatility = returns.std() * np.sqrt(252)
                data[ticker] = {
                    'market_cap': market_cap,
                    'volatility': volatility,
                    'prices': close,
                    'returns': returns
                }
            except:
                continue
        return data
    
    @staticmethod
    def extract_all(tickers, start_date, end_date, debt_ratio=0.4):
        equity_data = MarketDataExtractor50.fetch_equity_data(tickers, start_date, end_date)
        valid_tickers = [t for t in tickers if t in equity_data]
        if not valid_tickers:
            raise ValueError("Nijedan ticker nije validan.")
        returns_df = pd.DataFrame()
        for t in valid_tickers:
            returns_df[t] = equity_data[t]['returns']
        returns_df = returns_df.dropna(axis=0, how='any')
        corr_matrix = returns_df.corr().values
        equity_values = []
        equity_vols = []
        debt_values = []
        prices = {}
        for ticker in valid_tickers:
            eq = equity_data[ticker]
            market_cap = eq['market_cap']
            vol = eq['volatility']
            prices[ticker] = eq['prices']
            if market_cap is None:
                print(f"⚠️ Market cap za {ticker} nije dostupan, koristim procenu.")
                market_cap = eq['prices'].iloc[-1] * 1_000_000_000
            debt = debt_ratio * market_cap
            equity_values.append(market_cap)
            equity_vols.append(vol)
            debt_values.append(debt)
        return {
            'tickers': valid_tickers,
            'equity_values': np.array(equity_values),
            'equity_vols': np.array(equity_vols),
            'debt_values': np.array(debt_values),
            'correlation_matrix': corr_matrix,
            'returns': returns_df,
            'prices': pd.DataFrame(prices)
        }

# ---------- 3. GLAVNA KLASA MODELA ----------
class HawkesMertonContagion:
    def __init__(self, n_companies, T=1.0, dt=1/252,
                 use_heston=False, use_stochastic_rate=False,
                 barrier_growth_rate=0.02, barrier_target=None,
                 barrier_mean_reversion=0.5,
                 jump_intensity=0.5, jump_mean=-0.15, jump_std=0.10,
                 recovery_base=0.4, recovery_sensitivity=-0.5,
                 regime_switching=False):
        self.N = n_companies
        self.T = T
        self.dt = dt
        self.steps = int(self.T / self.dt) + 1
        self.use_heston = use_heston
        self.use_stochastic_rate = use_stochastic_rate
        self.V0 = np.ones(self.N) * 100.0
        self.drift = np.ones(self.N) * 0.05
        self.vol = np.ones(self.N) * 0.20
        self.kappa = np.ones(self.N) * 2.0
        self.theta = np.ones(self.N) * 0.04
        self.xi = np.ones(self.N) * 0.3
        self.v0 = np.ones(self.N) * 0.04
        self.rho_asset_vol = np.ones(self.N) * -0.7
        self.D0 = np.ones(self.N) * 60.0
        self.barrier_growth_rate = barrier_growth_rate
        self.barrier_target = barrier_target
        self.barrier_mean_reversion = barrier_mean_reversion
        self.r0 = 0.02
        self.r_mean = 0.03
        self.r_speed = 0.5
        self.r_vol = 0.01
        self.rho_asset_rate = -0.2
        self.base_intensity = np.ones(self.N) * 0.01
        self.beta = 2.0
        self.gamma = np.zeros((self.N, self.N))
        self.corr_assets = np.eye(self.N)
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.recovery_base = recovery_base
        self.recovery_sensitivity = recovery_sensitivity
        self._L_assets = None
        self._L_gbm_rate = None
        self._L_heston = None
        self.exposures = None
        
        # Regime-Switching
        self.regime_switching = regime_switching
        if regime_switching:
            self.regime_transition = np.array([[0.95, 0.05], [0.10, 0.90]])
            self.regime_jump_intensity = [0.1, 0.8]
            self.regime_recovery_base = [0.55, 0.35]

    def set_contagion_network(self, gamma_matrix):
        if gamma_matrix.shape != (self.N, self.N):
            raise ValueError("Gamma matrica mora biti NxN.")
        np.fill_diagonal(gamma_matrix, 0.0)
        self.gamma = gamma_matrix

    def set_correlation_assets(self, corr_matrix):
        if corr_matrix.shape != (self.N, self.N):
            raise ValueError("Korelaciona matrica mora biti NxN.")
        self.corr_assets = corr_matrix

    def calibrate_kmv(self, equity_values, equity_vols, debt_values,
                      risk_free_rate, time_horizon=1.0, max_iter=100, tol=1e-6):
        N = self.N
        V0_cal = np.zeros(N)
        vol_cal = np.zeros(N)
        for i in range(N):
            E = equity_values[i]
            sigma_E = equity_vols[i]
            D = debt_values[i]
            r = risk_free_rate
            T = time_horizon
            V = E + D
            sigma_V = sigma_E * E / V
            sigma_V = min(sigma_V, 0.8)
            for _ in range(max_iter):
                d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
                d2 = d1 - sigma_V * np.sqrt(T)
                E_model = V * stats.norm.cdf(d1) - D * np.exp(-r * T) * stats.norm.cdf(d2)
                sigma_E_model = (V / E_model) * stats.norm.cdf(d1) * sigma_V
                err_E = E_model - E
                err_sigma = sigma_E_model - sigma_E
                if abs(err_E) < tol and abs(err_sigma) < tol:
                    break
                if abs(err_E) > 1e-4:
                    V = V - 0.5 * err_E
                if abs(err_sigma) > 1e-6:
                    sigma_V = sigma_V - 0.3 * err_sigma
                V = max(V, D + 1e-6)
                sigma_V = max(sigma_V, 0.01)
                sigma_V = min(sigma_V, 0.8)
            V0_cal[i] = V
            vol_cal[i] = sigma_V
        self.V0 = V0_cal
        self.vol = vol_cal
        self.D0 = debt_values
        return V0_cal, vol_cal

    def _prepare_cholesky(self):
        if self.use_heston:
            dim = 2 * self.N + 1
            corr_full = np.eye(dim)
            corr_full[:self.N, :self.N] = self.corr_assets
            for i in range(self.N):
                corr_full[i, self.N + i] = self.rho_asset_vol[i]
                corr_full[self.N + i, i] = self.rho_asset_vol[i]
            corr_full[:self.N, -1] = self.rho_asset_rate
            corr_full[-1, :self.N] = self.rho_asset_rate
            corr_full = nearest_positive_definite(corr_full)
            self._L_heston = cholesky(corr_full, lower=True)
        else:
            if self.use_stochastic_rate:
                dim = self.N + 1
                corr_full = np.eye(dim)
                corr_full[:self.N, :self.N] = self.corr_assets
                corr_full[:self.N, -1] = self.rho_asset_rate
                corr_full[-1, :self.N] = self.rho_asset_rate
                corr_full = nearest_positive_definite(corr_full)
                self._L_gbm_rate = cholesky(corr_full, lower=True)
            else:
                corr_assets = nearest_positive_definite(self.corr_assets)
                self._L_assets = cholesky(corr_assets, lower=True)

    def simulate_single_path(self, return_paths=False):
        """
        Sekvencijalna simulacija jedne putanje (pouzdana, koristi se za sve MC)
        """
        if (self._L_assets is None and self._L_gbm_rate is None and self._L_heston is None):
            self._prepare_cholesky()
        N = self.N
        steps = self.steps
        dt = self.dt
        T = self.T
        use_heston = self.use_heston
        use_rate = self.use_stochastic_rate
        V = np.zeros((steps, N))
        lambda_t = np.zeros((steps, N))
        default_state = np.zeros((steps, N), dtype=bool)
        r_path = np.zeros(steps) if use_rate else None
        v_path = np.zeros((steps, N)) if use_heston else None
        V[0, :] = self.V0
        lambda_t[0, :] = self.base_intensity
        if use_rate:
            r_path[0] = self.r0
        if use_heston:
            v_path[0, :] = self.v0

        if self.regime_switching:
            regime = np.zeros(N, dtype=int)

        if use_heston:
            Z = np.random.standard_normal((steps - 1, self._L_heston.shape[0]))
            W = Z @ self._L_heston.T
            dW_asset = W[:, :N] * np.sqrt(dt)
            dW_vol = W[:, N:2*N] * np.sqrt(dt)
            dW_rate = W[:, -1:] * np.sqrt(dt)
        else:
            if use_rate:
                Z = np.random.standard_normal((steps - 1, self._L_gbm_rate.shape[0]))
                W = Z @ self._L_gbm_rate.T
                dW_asset = W[:, :N] * np.sqrt(dt)
                dW_rate = W[:, -1:] * np.sqrt(dt)
                dW_vol = None
            else:
                Z = np.random.standard_normal((steps - 1, N))
                W = Z @ self._L_assets.T
                dW_asset = W * np.sqrt(dt)
                dW_rate = None
                dW_vol = None
        has_defaulted = np.zeros(N, dtype=bool)
        if use_heston:
            v_curr = self.v0.copy()
        else:
            v_curr = self.vol**2

        for t in range(1, steps):
            if self.regime_switching:
                for i in range(N):
                    if regime[i] == 0:
                        if np.random.rand() < self.regime_transition[0, 1]:
                            regime[i] = 1
                    else:
                        if np.random.rand() < self.regime_transition[1, 0]:
                            regime[i] = 0
                current_jump_intensity = np.array([self.regime_jump_intensity[r] for r in regime])
                current_recovery_base = np.array([self.regime_recovery_base[r] for r in regime])
            else:
                current_jump_intensity = self.jump_intensity
                current_recovery_base = self.recovery_base

            if use_rate:
                r_prev = r_path[t-1]
                dr = self.r_speed * (self.r_mean - r_prev) * dt + self.r_vol * dW_rate[t-1, 0]
                r_curr = r_prev + dr
                r_path[t] = r_curr
            else:
                r_curr = 0.0
            if use_heston:
                v_prev = v_curr
                dv = self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev) * dW_vol[t-1, :]
                v_curr = v_prev + dv
                v_curr = np.maximum(v_curr, 0.0)
                v_path[t, :] = v_curr
            if use_heston:
                drift_asset = (self.drift - 0.5 * v_curr) * dt
                vol_asset = np.sqrt(v_curr) * dW_asset[t-1, :]
            else:
                drift_asset = (self.drift - 0.5 * self.vol**2) * dt
                vol_asset = self.vol * dW_asset[t-1, :]
            V[t, :] = V[t-1, :] * np.exp(drift_asset + vol_asset)

            if self.regime_switching:
                n_jumps = np.random.poisson(current_jump_intensity * dt, N)
            else:
                if self.jump_intensity > 0:
                    n_jumps = np.random.poisson(self.jump_intensity * dt, N)
                else:
                    n_jumps = 0
            if np.any(n_jumps > 0):
                jump_sizes = np.random.normal(self.jump_mean, self.jump_std, N)
                V[t, :] *= np.exp(jump_sizes * n_jumps)
                V[t, :] = np.maximum(V[t, :], 0.0)

            V[t, has_defaulted] = 0.0
            if self.barrier_target is not None:
                D_curr = self.D0 * np.exp(-self.barrier_mean_reversion * t * dt) + \
                         self.barrier_target * (1 - np.exp(-self.barrier_mean_reversion * t * dt))
            else:
                D_curr = self.D0 * np.exp(self.barrier_growth_rate * t * dt)
            if use_rate:
                D_curr = D_curr * np.exp(-r_curr * (T - t * dt))

            if t > 1:
                prev_new = default_state[t-1, :] & (~default_state[t-2, :])
            else:
                prev_new = np.zeros(N, dtype=bool)
            decay = self.beta * (self.base_intensity - lambda_t[t-1, :]) * dt
            jump_contagion = prev_new.astype(float) @ self.gamma
            lambda_curr = lambda_t[t-1, :] + decay + jump_contagion
            lambda_curr = np.maximum(lambda_curr, 0.0)
            lambda_t[t, :] = lambda_curr

            merton_default = (V[t, :] < D_curr) & (~has_defaulted)
            hazard = 1.0 - np.exp(-lambda_curr * dt)
            poisson_trigger = np.random.uniform(0, 1, N) < hazard
            poisson_default = poisson_trigger & (~has_defaulted) & (~merton_default)
            new_defaults = merton_default | poisson_default
            has_defaulted = has_defaulted | new_defaults
            default_state[t, :] = has_defaulted
            lambda_t[t, has_defaulted] = np.nan

        if return_paths:
            self.V = V
            self.lambda_t = lambda_t
            self.default_state = default_state
            self.r_path = r_path
            self.v_path = v_path
            return V, lambda_t, default_state, r_path, v_path
        else:
            if self.exposures is not None:
                current_lambda = lambda_t[-1, :]
                current_lambda = np.nan_to_num(current_lambda, nan=0.0)
                if self.regime_switching:
                    recovery_rates = current_recovery_base + self.recovery_sensitivity * current_lambda
                else:
                    recovery_rates = self.recovery_base + self.recovery_sensitivity * current_lambda
                recovery_rates = np.clip(recovery_rates, 0.0, 1.0)
                loss = np.sum(default_state[-1, :] * self.exposures * (1 - recovery_rates))
                return loss
            else:
                return 0.0

# ---------- 4. MONTE KARLO SEKVENCIJALNO ----------
def run_monte_carlo_sequential(model, n_simulations, exposures, alpha=0.01, show_progress=True):
    model.exposures = exposures
    model._prepare_cholesky()
    losses = np.zeros(n_simulations)
    for i in tqdm(range(n_simulations), desc="Monte Carlo", disable=not show_progress):
        losses[i] = model.simulate_single_path(return_paths=False)
    sorted_losses = np.sort(losses)
    var_idx = int(np.ceil((1 - alpha) * n_simulations)) - 1
    VaR = sorted_losses[var_idx]
    CVaR = np.mean(sorted_losses[var_idx:])
    return losses, VaR, CVaR

# ---------- 5. AUTOMATSKA KALIBRACIJA SA CRN ----------
def compute_implicit_spread(params, model_base, exposures, n_sims=2000, alpha=0.01):
    np.random.seed(42)
    jump_intensity, gamma_multiplier, recovery_base, recovery_sensitivity = params
    N = model_base.N
    corr = model_base.corr_assets
    model_new = HawkesMertonContagion(
        n_companies=N,
        T=1.0,
        dt=1/252,
        use_heston=False,
        use_stochastic_rate=False,
        barrier_growth_rate=0.02,
        barrier_target=None,
        jump_intensity=jump_intensity,
        jump_mean=-0.15,
        jump_std=0.10,
        recovery_base=recovery_base,
        recovery_sensitivity=recovery_sensitivity
    )
    model_new.V0 = model_base.V0.copy()
    model_new.vol = model_base.vol.copy()
    model_new.D0 = model_base.D0.copy()
    model_new.corr_assets = corr.copy()
    model_new._prepare_cholesky()
    gamma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                gamma[i, j] = 0.05 * gamma_multiplier * max(0, corr[i, j])
    model_new.set_contagion_network(gamma)
    losses, VaR, CVaR = run_monte_carlo_sequential(
        model_new, n_sims, exposures, alpha=alpha, show_progress=False
    )
    default_prob = np.mean(losses > 0) * 100
    spread_bps = -np.log(1 - default_prob/100) * 10000
    return spread_bps

def objective(params, model_base, exposures, target_spread_bps=153, n_sims=2000):
    spread = compute_implicit_spread(params, model_base, exposures, n_sims)
    error = (spread - target_spread_bps) ** 2
    print(f"   Parametri: {[round(p,4) for p in params]}, Spread: {spread:.0f} bps, Greška: {error:.0f}")
    return error

def calibrate_model(model_base, exposures, target_spread_bps=153, n_sims=2000, maxiter=20):
    initial = [model_base.jump_intensity, 1.0, model_base.recovery_base, model_base.recovery_sensitivity]
    bounds = [(0.01, 0.5), (0.1, 3.0), (0.3, 0.8), (-0.8, -0.1)]
    print("Pokrećem optimizaciju (može potrajati ~10-15 minuta)...")
    result = minimize(
        objective,
        initial,
        args=(model_base, exposures, target_spread_bps, n_sims),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-4}
    )
    if result.success:
        opt = result.x
        print("\n✅ Optimizacija uspješna!")
        print(f"   Optimalni parametri:")
        print(f"      jump_intensity       = {opt[0]:.4f}")
        print(f"      gamma_multiplier     = {opt[1]:.4f}")
        print(f"      recovery_base        = {opt[2]:.4f}")
        print(f"      recovery_sensitivity = {opt[3]:.4f}")
        print(f"   Minimalna greška: {result.fun:.0f}")
        return opt
    else:
        print("❌ Optimizacija nije uspjela.")
        print(result.message)
        return None

# ---------- 6. Glavni izvršni dio ----------
print("=== 1. PREUZIMANJE KREDITNOG SPREADA SA FRED ===")
try:
    treasury = fetch_fred_series('GS10', '2020-01-01')
    baa = fetch_fred_series('BAA', '2020-01-01')
    if not treasury.empty and not baa.empty:
        df_spread = pd.DataFrame({'Treasury': treasury, 'BAA': baa}).dropna()
        df_spread['Spread'] = df_spread['BAA'] - df_spread['Treasury']
        risk_free_rate = df_spread['Treasury'].iloc[-1] / 100
        market_spread = df_spread['Spread'].iloc[-1]
        print(f"✅ Trenutna 10Y Treasury: {risk_free_rate:.2%}")
        print(f"📈 Trenutni BAA-Treasury spread: {market_spread:.2f}%")
        print(f"📈 Prosečan spread: {df_spread['Spread'].mean():.2f}%")
    else:
        risk_free_rate = 0.045
        market_spread = 1.96
        print(f"⚠️ FRED nije dostupan, koristim r={risk_free_rate:.2%}, spread={market_spread:.2f}%")
except:
    risk_free_rate = 0.045
    market_spread = 1.96
    print(f"⚠️ FRED nije dostupan, koristim r={risk_free_rate:.2%}, spread={market_spread:.2f}%")

print("\n=== 2. PREUZIMANJE 50 S&P 500 FIRMI ===")
tickers = MarketDataExtractor50.get_sp500_tickers(50)
print(f"✅ Preuzeto {len(tickers)} tickera: {tickers[:5]}...")

end_date = datetime.today()
start_date = end_date - timedelta(days=2*365)
data = MarketDataExtractor50.extract_all(tickers, start_date, end_date)
valid_tickers = data['tickers']
print(f"✅ Validnih tickera: {len(valid_tickers)}")

equity_values = data['equity_values']
equity_vols = data['equity_vols']
debt_values = data['debt_values']
corr_matrix = data['correlation_matrix']

print(f"📊 Market cap range: ${equity_values.min():.2e} - ${equity_values.max():.2e}")
print(f"📊 Volatility range: {equity_vols.min():.2%} - {equity_vols.max():.2%}")

print("\n=== 3. INICIJALIZACIJA MODELA ===")
N = len(valid_tickers)
model = HawkesMertonContagion(
    n_companies=N,
    T=1.0,
    dt=1/252,
    use_heston=False,
    use_stochastic_rate=False,
    barrier_growth_rate=0.02,
    barrier_target=None,
    jump_intensity=0.05,
    jump_mean=-0.15,
    jump_std=0.10,
    recovery_base=0.65,
    recovery_sensitivity=-0.3,
    regime_switching=False
)
model.set_correlation_assets(corr_matrix)
print("✅ Model inicijalizovan")

print("\n=== 4. KMV KALIBRACIJA ===")
V0_cal, vol_cal = model.calibrate_kmv(
    equity_values=equity_values,
    equity_vols=equity_vols,
    debt_values=debt_values,
    risk_free_rate=risk_free_rate,
    time_horizon=1.0
)
print(f"✅ KMV kalibracija završena")
print(f"📊 V0 range: {V0_cal.min():.2e} - {V0_cal.max():.2e}")
print(f"📊 Asset vol range: {vol_cal.min():.2%} - {vol_cal.max():.2%}")

print("\n=== 5. KREIRANJE MATRICE ZARAZE (baseline) ===")
gamma_base = np.zeros((N, N))
gamma_multiplier = 0.3
for i in range(N):
    for j in range(N):
        if i != j:
            gamma_base[i, j] = 0.05 * gamma_multiplier * max(0, corr_matrix[i, j])
model.set_contagion_network(gamma_base)
print(f"✅ Gamma matrica kreirana (prosek: {gamma_base.mean():.4f}, max: {gamma_base.max():.4f})")

exposures = np.ones(N) * 1_000_000

# ---------- 6. BASELINE SIMULACIJA ----------
print("\n=== 6. BASELINE MONTE KARLO (10k) ===")
n_sims = 10_000
losses_base, VaR_base, CVaR_base = run_monte_carlo_sequential(
    model, n_sims, exposures, alpha=0.01, show_progress=True
)

print(f"\nBaseline rezultati (10k):")
print(f"  VaR (99%):  ${VaR_base:,.0f}")
print(f"  CVaR (99%): ${CVaR_base:,.0f}")
print(f"  Prosečan gubitak: ${losses_base.mean():,.0f}")
print(f"  Max gubitak: ${losses_base.max():,.0f}")
print(f"  Default prob: {(losses_base > 0).mean()*100:.2f}%")

# ---------- 7. SCENARIO ANALIZA (ISPRAVLJENA) ----------
print("\n=== 7. SCENARIO ANALYSIS (6 scenarija, 10k svaki) ===")

# Definicija scenarija – sada samo specificiramo šta se mijenja u odnosu na baseline
scenario_modifiers = {
    'Baseline': {
        'gamma_multiplier': 1.0,
        'jump_intensity': 0.3,
        'recovery_sensitivity': -0.5,
        'description': 'Osnovni model (veća zaraza i skokovi)'
    },
    'High Contagion': {
        'gamma_multiplier': 2.5,
        'jump_intensity': 0.3,
        'recovery_sensitivity': -0.5,
        'description': 'Pojačana zaraza'
    },
    'Severe Jumps': {
        'gamma_multiplier': 1.0,
        'jump_intensity': 0.8,
        'jump_mean': -0.25,
        'jump_std': 0.15,
        'recovery_sensitivity': -0.5,
        'description': 'Snažni skokovi'
    },
    'Low Recovery': {
        'gamma_multiplier': 1.0,
        'jump_intensity': 0.3,
        'recovery_sensitivity': -0.8,
        'description': 'Nizak oporavak'
    },
    'Compound Crisis': {
        'gamma_multiplier': 2.5,
        'jump_intensity': 0.8,
        'jump_mean': -0.25,
        'jump_std': 0.15,
        'recovery_sensitivity': -0.8,
        'description': 'Kombinovana kriza'
    },
    'Mild Stress': {
        'gamma_multiplier': 0.5,
        'jump_intensity': 0.1,
        'jump_mean': -0.05,
        'jump_std': 0.05,
        'recovery_sensitivity': -0.3,
        'description': 'Blagi stres'
    }
}

def run_scenario(model_base, mods, n_sims, exposures):
    """
    Kreira novi model od model_base, primjenjuje modifikacije iz mods.
    """
    N = model_base.N
    corr = model_base.corr_assets
    # Počni od parametara model_base
    params = {
        'n_companies': N,
        'T': model_base.T,
        'dt': model_base.dt,
        'use_heston': model_base.use_heston,
        'use_stochastic_rate': model_base.use_stochastic_rate,
        'barrier_growth_rate': model_base.barrier_growth_rate,
        'barrier_target': model_base.barrier_target,
        'barrier_mean_reversion': model_base.barrier_mean_reversion,
        'jump_intensity': model_base.jump_intensity,
        'jump_mean': model_base.jump_mean,
        'jump_std': model_base.jump_std,
        'recovery_base': model_base.recovery_base,
        'recovery_sensitivity': model_base.recovery_sensitivity,
        'regime_switching': model_base.regime_switching
    }
    # Primijeni modifikacije (preklapanje)
    for key, val in mods.items():
        if key in params:
            params[key] = val
    model_new = HawkesMertonContagion(**params)
    model_new.V0 = model_base.V0.copy()
    model_new.vol = model_base.vol.copy()
    model_new.D0 = model_base.D0.copy()
    model_new.corr_assets = corr.copy()
    model_new._prepare_cholesky()
    gamma = np.zeros((N, N))
    gamma_mult = mods.get('gamma_multiplier', 1.0)
    for i in range(N):
        for j in range(N):
            if i != j:
                gamma[i, j] = 0.05 * gamma_mult * max(0, corr[i, j])
    model_new.set_contagion_network(gamma)
    losses, VaR, CVaR = run_monte_carlo_sequential(
        model_new, n_sims, exposures, alpha=0.01, show_progress=False
    )
    default_prob = np.mean(losses > 0) * 100
    avg_loss = losses.mean()
    std_loss = losses.std()
    max_loss = losses.max()
    dd = (model_new.V0 - model_new.D0) / (model_new.V0 * model_new.vol)
    avg_dd = np.mean(dd)
    min_dd = np.min(dd)
    max_dd = np.max(dd)
    return {
        'VaR (99%)': VaR,
        'CVaR (99%)': CVaR,
        'Avg Loss': avg_loss,
        'Std Loss': std_loss,
        'Max Loss': max_loss,
        'Default Prob (%)': default_prob,
        'Avg DD': avg_dd,
        'Min DD': min_dd,
        'Max DD': max_dd,
        'Gamma Mean': gamma.mean(),
        'Gamma Max': gamma.max()
    }

results = {}
for name, mods in tqdm(scenario_modifiers.items(), desc="Scenariji"):
    print(f"\n▶️ {name}: {mods['description']}")
    results[name] = run_scenario(model, mods, n_sims, exposures)
    print(f"   VaR: ${results[name]['VaR (99%)']:,.0f}")
    print(f"   CVaR: ${results[name]['CVaR (99%)']:,.0f}")
    print(f"   Default Prob: {results[name]['Default Prob (%)']:.2f}%")

df_results = pd.DataFrame(results).T
df_results.index.name = 'Scenario'

# ---------- 8. GRAFIKONI ----------
print("\n=== 8. GRAFIKONI I POREĐENJE SA TRŽIŠTEM ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
x = np.arange(len(df_results))
width = 0.35
ax.bar(x - width/2, df_results['VaR (99%)'], width, label='VaR 99%', color='skyblue')
ax.bar(x + width/2, df_results['CVaR (99%)'], width, label='CVaR 99%', color='darkred')
ax.set_xticks(x)
ax.set_xticklabels(df_results.index, rotation=45, ha='right')
ax.set_ylabel('Gubitak ($)')
ax.set_title('VaR i CVaR po scenarijima')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
colors = ['green' if v < 20 else 'orange' if v < 40 else 'red' for v in df_results['Default Prob (%)']]
ax.bar(df_results.index, df_results['Default Prob (%)'], color=colors, alpha=0.7)
ax.axhline(y=df_results['Default Prob (%)'].mean(), color='blue', linestyle='--', 
           label=f'Prosek: {df_results["Default Prob (%)"].mean():.1f}%')
ax.set_ylabel('Verovatnoća defaulta (%)')
ax.set_title('Verovatnoća gubitka > 0')
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax = axes[1, 0]
ax.plot(df_results.index, df_results['Avg DD'], 'o-', label='Prosečan DD', color='blue', linewidth=2)
ax.plot(df_results.index, df_results['Min DD'], 's--', label='Min DD', color='red', linewidth=2)
ax.plot(df_results.index, df_results['Max DD'], '^--', label='Max DD', color='green', linewidth=2)
ax.axhline(y=2.5, color='orange', linestyle=':', label='Granica srednjeg rizika (DD=2.5)')
ax.axhline(y=1.5, color='red', linestyle=':', label='Granica visokog rizika (DD=1.5)')
ax.set_ylabel('Distance-to-Default')
ax.set_title('DD metrika po scenarijima')
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax = axes[1, 1]
df_results['Implicit Spread (bps)'] = -np.log(1 - df_results['Default Prob (%)']/100) * 10000
ax.bar(df_results.index, df_results['Implicit Spread (bps)'], alpha=0.7, color='purple')
ax.axhline(y=market_spread * 100, color='red', linestyle='--', 
           label=f'Tržišni BAA-Treasury spread: {market_spread:.2f}%')
ax.set_ylabel('Spread (bps)')
ax.set_title('Implicitni kreditni spread vs Tržišni spread')
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# ---------- 9. DETALJNA TABELA ----------
print("\n=== DETALJNI REZULTATI ===")
df_display = df_results[['VaR (99%)', 'CVaR (99%)', 'Default Prob (%)', 'Avg DD', 'Min DD', 'Max DD']].round(2)
print(df_display)

print("\n=== POREĐENJE SA TRŽIŠNIM SPREAD-OM ===")
df_compare = df_results[['Default Prob (%)', 'Implicit Spread (bps)', 'Avg DD', 'Min DD']].copy()
df_compare['Market Spread (bps)'] = market_spread * 100
df_compare['Spread Difference (bps)'] = df_compare['Implicit Spread (bps)'] - df_compare['Market Spread (bps)']
print(df_compare.round(2))

best_match = df_compare.iloc[(df_compare['Spread Difference (bps)'].abs()).argmin()]
print(f"\n🏆 Scenario koji najbolje replicira tržišni spread ({market_spread:.2f}%):")
print(f"   {best_match.name} sa spread-om od {best_match['Implicit Spread (bps)']:.0f} bps")
print(f"   Razlika: {best_match['Spread Difference (bps)']:.0f} bps")

print("\n=== SCENARIO RANG LISTA PO RIZIKU ===")
ranked = df_results.sort_values('CVaR (99%)', ascending=False)
print(ranked[['CVaR (99%)', 'Default Prob (%)', 'Avg DD']].round(2))

print("\n✅ SVE GOTOVO! Model kalibriran, baseline i 6 scenarija analizirani.")

# ================================================================
# ============== DODATNA ANALIZA: TOP FIRME I PUTANJE ============
# ================================================================

print("\n" + "="*80)
print("=== 9. DODATNA ANALIZA: TOP FIRME, DEFAULTI PO SCENARIJU I PUTANJE ===")
print("="*80)

# 9a. Top 10 najrizičnijih firmi (Distance-to-Default) sa šumom
noise_std = 0.2
D_obs = model.D0 * np.exp(np.random.normal(-0.1, noise_std, model.N))
dd = (model.V0 - D_obs) / (model.V0 * model.vol)

dd_sorted_idx = np.argsort(dd)
print("\n📊 TOP 10 NAJRIZIČNIJIH FIRMI (najmanji DD - bazni model):")
print("   Rang  Ticker     DD      V0 (B$)   Dug (B$)   Vol (%)")
print("   --------------------------------------------------------")
for i in range(10):
    idx = dd_sorted_idx[i]
    print(f"   {i+1:2d}.   {valid_tickers[idx]:6s}   {dd[idx]:.3f}   {model.V0[idx]/1e9:8.1f}   {model.D0[idx]/1e9:8.1f}   {model.vol[idx]*100:5.1f}%")

# 9b. Funkcija za prebrojavanje defaulta po firmi – ispravljena (koristi kalibrirani model)
def count_defaults_per_firm(model_base, mods, n_paths=1000, exposures=None):
    # Kreiraj model od model_base sa modifikacijama (isto kao u run_scenario)
    N = model_base.N
    corr = model_base.corr_assets
    params = {
        'n_companies': N,
        'T': model_base.T,
        'dt': model_base.dt,
        'use_heston': model_base.use_heston,
        'use_stochastic_rate': model_base.use_stochastic_rate,
        'barrier_growth_rate': model_base.barrier_growth_rate,
        'barrier_target': model_base.barrier_target,
        'barrier_mean_reversion': model_base.barrier_mean_reversion,
        'jump_intensity': model_base.jump_intensity,
        'jump_mean': model_base.jump_mean,
        'jump_std': model_base.jump_std,
        'recovery_base': model_base.recovery_base,
        'recovery_sensitivity': model_base.recovery_sensitivity,
        'regime_switching': model_base.regime_switching
    }
    for key, val in mods.items():
        if key in params:
            params[key] = val
    model_new = HawkesMertonContagion(**params)
    model_new.V0 = model_base.V0.copy()
    model_new.vol = model_base.vol.copy()
    model_new.D0 = model_base.D0.copy()
    model_new.corr_assets = corr.copy()
    model_new._prepare_cholesky()
    gamma_mult = mods.get('gamma_multiplier', 1.0)
    gamma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                gamma[i, j] = 0.05 * gamma_mult * max(0, corr[i, j])
    model_new.set_contagion_network(gamma)
    if exposures is None:
        exposures = np.ones(N) * 1.0
    model_new.exposures = exposures

    default_counts = np.zeros(N, dtype=int)
    for _ in tqdm(range(n_paths), desc=f"Brojim default-e", leave=False):
        V, lam, default_state, r, v = model_new.simulate_single_path(return_paths=True)
        default_counts += default_state[-1, :].astype(int)
    return default_counts, model_base.V0  # tickers su van funkcije

# 9c. Pokreni za 4 ključna scenarija
scenarios_to_analyze = ['Baseline', 'High Contagion', 'Severe Jumps', 'Compound Crisis']
n_paths_per_scenario = 1000

default_dict = {}
for name in scenarios_to_analyze:
    mods = scenario_modifiers[name]
    print(f"\n▶️ Brojim default-e za scenario: {name} ({mods['description']})")
    counts, _ = count_defaults_per_firm(model, mods, n_paths=n_paths_per_scenario)
    default_dict[name] = counts

# 9d. Prikaz top 5 firmi po defaultu za svaki scenario
print("\n=== TOP 5 FIRMI PO BROJU DEFAULTA (od 1000 simulacija) ===")
for name, counts in default_dict.items():
    sorted_idx = np.argsort(counts)[::-1]
    top5 = [(valid_tickers[i], counts[i]) for i in sorted_idx[:5]]
    print(f"\n🔥 Scenario: {name}")
    print("   Rang  Ticker   Defaulti (od 1000)")
    for r, (ticker, cnt) in enumerate(top5, 1):
        print(f"   {r:2d}.   {ticker:6s}   {cnt:3d}")

# 9e. Interaktivne putanje za 5 najrizičnijih firmi (prema DD) za svaki scenario
print("\n=== INTERAKTIVNE PUTANJE ZA 5 NAJRIZIČNIJIH FIRMI (po DD) ===")

top5_dd_idx = dd_sorted_idx[:5]
top5_dd_tickers = [valid_tickers[i] for i in top5_dd_idx]

for name in scenarios_to_analyze:
    mods = scenario_modifiers[name]
    print(f"\n▶️ Generišem putanje za scenario: {name}")
    model_temp = HawkesMertonContagion(
        n_companies=N,
        T=1.0,
        dt=1/252,
        use_heston=False,
        use_stochastic_rate=False,
        barrier_growth_rate=model.barrier_growth_rate,
        barrier_target=model.barrier_target,
        barrier_mean_reversion=model.barrier_mean_reversion,
        jump_intensity=mods.get('jump_intensity', model.jump_intensity),
        jump_mean=mods.get('jump_mean', model.jump_mean),
        jump_std=mods.get('jump_std', model.jump_std),
        recovery_base=model.recovery_base,
        recovery_sensitivity=mods.get('recovery_sensitivity', model.recovery_sensitivity)
    )
    model_temp.V0 = model.V0.copy()
    model_temp.vol = model.vol.copy()
    model_temp.D0 = model.D0.copy()
    model_temp.corr_assets = corr_matrix.copy()
    gamma_mult = mods.get('gamma_multiplier', 1.0)
    gamma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                gamma[i, j] = 0.05 * gamma_mult * max(0, corr_matrix[i, j])
    model_temp.set_contagion_network(gamma)
    model_temp._prepare_cholesky()

    V, lam, default, r, v = model_temp.simulate_single_path(return_paths=True)
    t_axis = np.linspace(0, model.T, model.steps)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f'{name} - Putanje imovine (5 najrizičnijih)',
                                       f'{name} - Hawkes intenzitet (λ)'),
                        vertical_spacing=0.15)

    for idx in top5_dd_idx:
        ticker = valid_tickers[idx]
        fig.add_trace(go.Scatter(x=t_axis, y=V[:, idx], mode='lines',
                                 name=f'{ticker} (V0=${model.V0[idx]/1e9:.1f}B)',
                                 line=dict(width=2)),
                      row=1, col=1)

    for idx in top5_dd_idx:
        ticker = valid_tickers[idx]
        y = lam[:, idx]
        x = t_axis[~np.isnan(y)]
        y = y[~np.isnan(y)]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 name=f'{ticker} (λ)',
                                 line=dict(dash='dot')),
                      row=2, col=1)

    D0_0 = model.D0[top5_dd_idx[0]]
    barrier_curve = D0_0 * np.exp(model.barrier_growth_rate * t_axis)
    fig.add_trace(go.Scatter(x=t_axis, y=barrier_curve, mode='lines',
                             name='Barrier (TSLA)',
                             line=dict(color='black', dash='dash')),
                  row=1, col=1)

    max_lambda = np.nanmax(lam)
    if max_lambda > 0:
        fig.update_yaxes(range=[0, max_lambda * 1.2], row=2, col=1)
    else:
        fig.update_yaxes(range=[0, 0.05], row=2, col=1)

    fig.update_layout(height=800, width=1000,
                      title_text=f'Scenarij: {name} - 5 najrizičnijih firmi (DD)',
                      showlegend=True)
    fig.show()

# 9f. Uporedni bar chart verovatnoće defaulta sa oznakama
print("\n📊 Verovatnoća defaulta po scenariju (sa oznakama):")
fig, ax = plt.subplots(figsize=(10,6))
colors = ['green' if v < 30 else 'orange' if v < 50 else 'red' for v in df_results['Default Prob (%)']]
bars = ax.bar(df_results.index, df_results['Default Prob (%)'], color=colors, alpha=0.7)
ax.axhline(y=df_results['Default Prob (%)'].mean(), color='blue', linestyle='--',
           label=f'Prosek: {df_results["Default Prob (%)"].mean():.1f}%')
ax.set_ylabel('Verovatnoća defaulta (%)')
ax.set_title('Verovatnoća gubitka > 0 po scenarijima')
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, val in zip(bars, df_results['Default Prob (%)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

print("\n✅ DODATNA ANALIZA ZAVRŠENA!")
print("   - Top 10 DD firmi")
print("   - Top 5 firmi po defaultima za 4 ključna scenarija")
print("   - Interaktivne putanje za 5 najrizičnijih firmi za svaki scenario")

# ================================================================
# DODATAK: NASUMIČNE FIRME + DUBOKA ANALIZA (BENCHMARK)
# ================================================================

print("\n" + "="*80)
print("=== DODATNA ANALIZA: 50 NASUMIČNIH FIRMI IZ S&P 500 (BEZ TOP 50) ===")
print("="*80)

def clean_equity_data(equity_data):
    cleaned = {}
    for ticker, eq in equity_data.items():
        if eq['volatility'] is None or np.isnan(eq['volatility']) or eq['volatility'] == 0:
            print(f"   ⚠️ Uklanjam {ticker} (volatilnost: {eq['volatility']})")
            continue
        if eq['market_cap'] is None or np.isnan(eq['market_cap']) or eq['market_cap'] <= 0:
            print(f"   ⚠️ Uklanjam {ticker} (market cap: {eq['market_cap']})")
            continue
        cleaned[ticker] = eq
    return cleaned

sp500_extended = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM',
    'V', 'TSLA', 'XOM', 'UNH', 'PG', 'MA', 'JNJ', 'HD', 'COST', 'MRK',
    'ABBV', 'WMT', 'BAC', 'CRM', 'CVX', 'NFLX', 'ADBE', 'KO', 'PEP', 'TMO',
    'LIN', 'DIS', 'ORCL', 'CSCO', 'MCD', 'ACN', 'IBM', 'ABT', 'CAT', 'GE',
    'DHR', 'VZ', 'NOW', 'GS', 'PM', 'SPGI', 'QCOM', 'RTX', 'TXN', 'NEE',
    'T', 'LOW', 'SYK', 'BLK', 'MS', 'AXP', 'UNP', 'HON', 'COP', 'BMY',
    'CVS', 'ISRG', 'PLTR', 'PFE', 'SCHW', 'DE', 'BA', 'SBUX', 'AMGN', 'GILD',
    'NKE', 'MDT', 'FIS', 'REGN', 'ADI', 'MRNA', 'PANW', 'ADSK', 'MELI', 'LRCX',
    'CMCSA', 'INTU', 'AMAT', 'VRTX', 'SNPS', 'CDNS', 'KLAC', 'MU', 'WDAY', 'TEAM',
    'SQ', 'ROKU', 'ZM', 'DOCU', 'CRWD', 'OKTA', 'ZS', 'NET', 'DDOG', 'MDB',
    'UBER', 'SNOW', 'DASH', 'COIN', 'HOOD', 'RBLX', 'DKNG', 'BABA', 'JD', 'PDD'
]

top_50_set = set(MarketDataExtractor50.get_sp500_tickers(50))
available_tickers = [t for t in sp500_extended if t not in top_50_set]
if len(available_tickers) < 50:
    available_tickers = sp500_extended

np.random.seed(123)
random_tickers = np.random.choice(available_tickers, 50, replace=False).tolist()
print(f"✅ Izabrano 50 nasumičnih tickera (bez top 50): {random_tickers[:5]}...")

end_date = datetime.today()
start_date = end_date - timedelta(days=2*365)
equity_data_raw = MarketDataExtractor50.fetch_equity_data(random_tickers, start_date, end_date)
equity_data_clean = clean_equity_data(equity_data_raw)

if len(equity_data_clean) < 50:
    print(f"⚠️ Samo {len(equity_data_clean)} tickera, dodajem još...")
    current_tickers = set(equity_data_clean.keys())
    additional = [t for t in sp500_extended if t not in current_tickers and t not in random_tickers]
    np.random.seed(456)
    needed = 50 - len(equity_data_clean)
    if len(additional) >= needed:
        extra = np.random.choice(additional, needed, replace=False).tolist()
    else:
        extra = additional
    random_tickers_final = list(current_tickers) + extra
    equity_data_raw2 = MarketDataExtractor50.fetch_equity_data(random_tickers_final, start_date, end_date)
    equity_data_clean = clean_equity_data(equity_data_raw2)

valid_rand = list(equity_data_clean.keys())
print(f"✅ Nakon čišćenja: {len(valid_rand)} validnih tickera")

if len(valid_rand) < 10:
    raise ValueError("Premalo validnih tickera za analizu.")

returns_df_rand = pd.DataFrame()
for t in valid_rand:
    returns_df_rand[t] = equity_data_clean[t]['returns']
returns_df_rand = returns_df_rand.dropna(axis=0, how='any')
corr_matrix_rand = returns_df_rand.corr().values

equity_values_rand = []
equity_vols_rand = []
debt_values_rand = []
for ticker in valid_rand:
    eq = equity_data_clean[ticker]
    equity_values_rand.append(eq['market_cap'])
    equity_vols_rand.append(eq['volatility'])
    debt_values_rand.append(0.4 * eq['market_cap'])

equity_values_rand = np.array(equity_values_rand)
equity_vols_rand = np.array(equity_vols_rand)
debt_values_rand = np.array(debt_values_rand)

print(f"📊 Market cap range: ${equity_values_rand.min():.2e} - ${equity_values_rand.max():.2e}")
print(f"📊 Volatility range: {equity_vols_rand.min():.2%} - {equity_vols_rand.max():.2%}")

N_rand = len(valid_rand)
model_rand = HawkesMertonContagion(
    n_companies=N_rand,
    T=1.0,
    dt=1/252,
    use_heston=False,
    use_stochastic_rate=False,
    barrier_growth_rate=0.02,
    barrier_target=None,
    jump_intensity=0.3,
    jump_mean=-0.15,
    jump_std=0.10,
    recovery_base=0.4,
    recovery_sensitivity=-0.5
)
model_rand.set_correlation_assets(corr_matrix_rand)

V0_rand, vol_rand = model_rand.calibrate_kmv(
    equity_values=equity_values_rand,
    equity_vols=equity_vols_rand,
    debt_values=debt_values_rand,
    risk_free_rate=risk_free_rate,
    time_horizon=1.0
)

gamma_rand = np.zeros((N_rand, N_rand))
for i in range(N_rand):
    for j in range(N_rand):
        if i != j:
            gamma_rand[i, j] = 0.05 * max(0, corr_matrix_rand[i, j])
model_rand.set_contagion_network(gamma_rand)

exposures_rand = np.ones(N_rand) * 1_000_000

print("\n=== BASELINE MC za nasumične firme (5k) ===")
losses_rand, VaR_rand, CVaR_rand = run_monte_carlo_sequential(
    model_rand, 5_000, exposures_rand, alpha=0.01, show_progress=True
)

scenarios_rand = {k: scenario_modifiers[k] for k in ['Baseline', 'High Contagion', 'Severe Jumps', 'Compound Crisis']}
results_rand = {}
for name, mods in tqdm(scenarios_rand.items(), desc="Scenariji (random)"):
    print(f"\n▶️ {name}: {mods['description']}")
    res = run_scenario(model_rand, mods, 5_000, exposures_rand)
    results_rand[name] = res
    print(f"   VaR: ${res['VaR (99%)']:,.0f}, CVaR: ${res['CVaR (99%)']:,.0f}")

df_rand = pd.DataFrame(results_rand).T
df_rand.index.name = 'Scenario'

dd_rand = (model_rand.V0 - model_rand.D0) / (model_rand.V0 * model_rand.vol)
dd_sorted_idx_rand = np.argsort(dd_rand)
print("\n📊 TOP 10 NAJRIZIČNIJIH (nasumične firme):")
print("   Rang  Ticker     DD      V0 (B$)   Dug (B$)   Vol (%)")
print("   --------------------------------------------------------")
for i in range(10):
    idx = dd_sorted_idx_rand[i]
    print(f"   {i+1:2d}.   {valid_rand[idx]:6s}   {dd_rand[idx]:.3f}   {model_rand.V0[idx]/1e9:8.1f}   {model_rand.D0[idx]/1e9:8.1f}   {model_rand.vol[idx]*100:5.1f}%")

print("\n▶️ Defaulti po firmi za nasumične (Baseline, 500 putanja)")
counts_rand, _ = count_defaults_per_firm(
    model_rand, scenario_modifiers['Baseline'], n_paths=500, exposures=exposures_rand
)
sorted_idx_rand = np.argsort(counts_rand)[::-1]
top5_rand = [(valid_rand[i], counts_rand[i]) for i in sorted_idx_rand[:5]]
print("\nTop 5 firmi po defaultima (nasumične):")
for r, (ticker, cnt) in enumerate(top5_rand, 1):
    print(f"   {r}. {ticker}: {cnt} defaulta od 500")

# ================================================================
# DUBOKA ANALIZA: TOP 50 vs NASUMIČNE FIRME (BENCHMARK)
# ================================================================
print("\n" + "="*80)
print("=== DUBOKA ANALIZA: TOP 50 vs NASUMIČNE FIRME ===")
print("="*80)

print("\n📊 Poređenje baseline metrika:")
print(f"   Top 50:        VaR=${VaR_base:,.0f}, CVaR=${CVaR_base:,.0f}, Default={ (losses_base > 0).mean()*100:.2f}%")
print(f"   Nasumične:     VaR=${VaR_rand:,.0f}, CVaR=${CVaR_rand:,.0f}, Default={ (losses_rand > 0).mean()*100:.2f}%")

print(f"\n📊 Prosečan DD: Top 50 = {np.mean(dd):.3f}, Nasumične = {np.mean(dd_rand):.3f}")
print(f"   Min DD: Top 50 = {np.min(dd):.3f}, Nasumične = {np.min(dd_rand):.3f}")
print(f"   Max DD: Top 50 = {np.max(dd):.3f}, Nasumične = {np.max(dd_rand):.3f}")

from scipy.stats import mannwhitneyu, kstest
stat, p = mannwhitneyu(dd, dd_rand)
print(f"\n📊 Mann-Whitney U test za DD: stat={stat:.3f}, p={p:.4f}")
if p < 0.05:
    print("   → Postoji statistički značajna razlika u DD distribuciji.")
else:
    print("   → Nema statistički značajne razlike u DD distribuciji.")

compare_cvar = pd.DataFrame({
    'Top 50': df_results['CVaR (99%)'],
    'Nasumične': df_rand['CVaR (99%)']
})
print("\n📋 CVaR po scenarijima (uporedno):")
print(compare_cvar.round(0))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [VaR_base, VaR_rand], width, label='VaR 99%', color='skyblue')
ax.bar(x + width/2, [CVaR_base, CVaR_rand], width, label='CVaR 99%', color='darkred')
ax.set_xticks(x)
ax.set_xticklabels(['Top 50', 'Nasumične'])
ax.set_ylabel('Gubitak ($)')
ax.set_title('VaR i CVaR (baseline)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.bar(['Top 50', 'Nasumične'], [(losses_base > 0).mean()*100, (losses_rand > 0).mean()*100], color=['blue', 'orange'])
ax.set_ylabel('Verovatnoća defaulta (%)')
ax.set_title('Verovatnoća gubitka > 0 (baseline)')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.boxplot([dd, dd_rand], labels=['Top 50', 'Nasumične'])
ax.set_ylabel('Distance-to-Default')
ax.set_title('Distribucija DD')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(df_results.index, df_results['CVaR (99%)'], 'o-', label='Top 50', markersize=8)
ax.plot(df_rand.index, df_rand['CVaR (99%)'], 's-', label='Nasumične', markersize=8)
ax.set_xticklabels(df_results.index, rotation=45)
ax.set_ylabel('CVaR (99%)')
ax.set_title('CVaR po scenarijima')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("=== ZAKLJUČCI I PREPORUKE ===")
print("="*80)

avg_dd_ratio = np.mean(dd) / np.mean(dd_rand)
if avg_dd_ratio > 1.1:
    print("🔴 Top 50 firme imaju VIŠI prosječni DD (niži rizik) od nasumičnih.")
elif avg_dd_ratio < 0.9:
    print("🟡 Top 50 firme imaju NIŽI prosječni DD (viši rizik) od nasumičnih.")
else:
    print("🟢 Top 50 i nasumične firme imaju sličan prosječni rizik.")

best_top = df_compare.iloc[(df_compare['Spread Difference (bps)'].abs()).argmin()]
df_compare_rand = df_rand.copy()
df_compare_rand['Implicit Spread (bps)'] = -np.log(1 - df_compare_rand['Default Prob (%)']/100) * 10000
df_compare_rand['Market Spread (bps)'] = market_spread * 100
df_compare_rand['Spread Difference (bps)'] = df_compare_rand['Implicit Spread (bps)'] - df_compare_rand['Market Spread (bps)']
best_rand = df_compare_rand.iloc[(df_compare_rand['Spread Difference (bps)'].abs()).argmin()]

print(f"\n🏆 Najbolji scenario za Top 50: {best_top.name} (razlika {best_top['Spread Difference (bps)']:.0f} bps)")
print(f"🏆 Najbolji scenario za nasumične: {best_rand.name} (razlika {best_rand['Spread Difference (bps)']:.0f} bps)")

print("\n💡 PREPORUKE ZA POBOLJŠANJE OTPORNOSTI:")
if best_top.name == 'Compound Crisis' or best_top.name == 'Severe Jumps':
    print("   - Top 50 model je najotporniji u scenarijima sa skokovima – predlažemo da se fokusirate na hedžiranje skokova.")
else:
    print("   - Top 50 model je najosjetljiviji na zarazu – preporučuje se diverzifikacija u manje korelirane sektore.")

if best_rand.name == 'Mild Stress':
    print("   - Nasumične firme su najbolje kalibrirane u blagom stresu, što sugerira da su manje volatilne – preporučuje se uključivanje takvih firmi za stabilnost.")

print("\n📈 Stabilnost portfolija (CVaR range):")
cvar_range_top = df_results['CVaR (99%)'].max() - df_results['CVaR (99%)'].min()
cvar_range_rand = df_rand['CVaR (99%)'].max() - df_rand['CVaR (99%)'].min()
print(f"   Top 50 CVaR raspon: ${cvar_range_top:,.0f}")
print(f"   Nasumične CVaR raspon: ${cvar_range_rand:,.0f}")
if cvar_range_top < cvar_range_rand:
    print("   ✅ Top 50 portfolio je stabilniji (manji raspon CVaR).")
else:
    print("   ✅ Nasumične firme su stabilnije (manji raspon CVaR).")

print("\n✅ Duboka analiza završena.")

# ================================================================
# 12. NAPREDNE ANALIZE: SEKTORI, OSJETLJIVOST, EL/UL, OPTIMIZACIJA
# ================================================================

print("\n" + "="*80)
print("=== 12. NAPREDNE ANALIZE ===")
print("="*80)

# ---------- 12a. Sektorska analiza ----------
print("\n📊 SEKTORSKA ANALIZA (top 50)")

sector_map = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'Tech',
    'META': 'Tech', 'BRK-B': 'Financials', 'LLY': 'Healthcare', 'AVGO': 'Tech', 'JPM': 'Financials',
    'V': 'Financials', 'TSLA': 'Consumer', 'XOM': 'Energy', 'UNH': 'Healthcare', 'PG': 'Consumer',
    'MA': 'Financials', 'JNJ': 'Healthcare', 'HD': 'Consumer', 'COST': 'Consumer', 'MRK': 'Healthcare',
    'ABBV': 'Healthcare', 'WMT': 'Consumer', 'BAC': 'Financials', 'CRM': 'Tech', 'CVX': 'Energy',
    'NFLX': 'Tech', 'ADBE': 'Tech', 'KO': 'Consumer', 'PEP': 'Consumer', 'TMO': 'Healthcare',
    'LIN': 'Materials', 'DIS': 'Consumer', 'ORCL': 'Tech', 'CSCO': 'Tech', 'MCD': 'Consumer',
    'ACN': 'Tech', 'IBM': 'Tech', 'ABT': 'Healthcare', 'CAT': 'Industrials', 'GE': 'Industrials',
    'DHR': 'Healthcare', 'VZ': 'Telecom', 'NOW': 'Tech', 'GS': 'Financials', 'PM': 'Consumer',
    'SPGI': 'Financials', 'QCOM': 'Tech', 'RTX': 'Industrials', 'TXN': 'Tech', 'NEE': 'Utilities'
}

sectors = [sector_map.get(t, 'Other') for t in valid_tickers]
sector_df = pd.DataFrame({'Ticker': valid_tickers, 'Sector': sectors, 'DD': dd, 'Vol': model.vol, 'V0': model.V0})

sector_agg = sector_df.groupby('Sector').agg({
    'DD': ['mean', 'min', 'max'],
    'Vol': 'mean',
    'V0': 'sum'
}).round(3)
sector_agg.columns = ['DD_avg', 'DD_min', 'DD_max', 'Vol_avg', 'V0_sum']
sector_agg['V0_sum_B'] = sector_agg['V0_sum'] / 1e9
print("\n📋 Sektorska agregacija:")
print(sector_agg.to_string())

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sector_agg['DD_avg'].sort_values().plot(kind='barh', ax=ax[0], color='skyblue', edgecolor='black')
ax[0].set_xlabel('Prosečan DD')
ax[0].set_title('Prosečan Distance-to-Default po sektoru')
ax[0].grid(True, alpha=0.3)
sector_agg['V0_sum_B'].sort_values().plot(kind='barh', ax=ax[1], color='coral', edgecolor='black')
ax[1].set_xlabel('Ukupna V0 (milijarde $)')
ax[1].set_title('Veličina imovine po sektoru')
ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- 12b. Analiza osetljivosti (Tornado plot) ----------
print("\n📊 ANALIZA OSJETLJIVOSTI (Tornado plot)")

param_ranges = {
    'jump_intensity': [0.1, 0.3, 0.5, 0.8, 1.0],
    'gamma_multiplier': [0.5, 1.0, 1.5, 2.0, 2.5],
    'recovery_sensitivity': [-0.2, -0.5, -0.8, -1.0],
    'jump_mean': [-0.05, -0.10, -0.15, -0.20, -0.25]
}

def test_sensitivity(model_base, param_name, param_values, exposures, n_sims=2000):
    results = []
    for val in param_values:
        mods = {'gamma_multiplier': 1.0, 'jump_intensity': 0.3, 'recovery_sensitivity': -0.5}  # početni
        mods[param_name] = val
        res = run_scenario(model_base, mods, n_sims, exposures)
        results.append({'param_value': val, 'CVaR': res['CVaR (99%)']})
    return pd.DataFrame(results)

sensitivity_results = {}
for param in ['jump_intensity', 'gamma_multiplier', 'recovery_sensitivity', 'jump_mean']:
    print(f"   Testiram {param}...")
    df_sens = test_sensitivity(model, param, param_ranges[param], exposures, n_sims=2000)
    sensitivity_results[param] = df_sens

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (param, df) in zip(axes.flatten(), sensitivity_results.items()):
    ax.plot(df['param_value'], df['CVaR'], 'o-', color='blue', linewidth=2)
    ax.set_xlabel(param)
    ax.set_ylabel('CVaR (99%)')
    ax.set_title(f'Osjetljivost na {param}')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- 12c. EL i UL ----------
print("\n📊 OČEKIVANI I NEOČEKIVANI GUBITAK (EL / UL)")

EL = losses_base.mean()
UL = losses_base.std()
print(f"Očekivani gubitak (EL): ${EL:,.0f}")
print(f"Neočekivani gubitak (UL): ${UL:,.0f}")
print(f"UL/EL odnos: {UL/EL:.2f}")

# ---------- 12d. Optimizacija portfolija (minimizacija CVaR) ----------
print("\n📊 OPTIMIZACIJA PORTFOLIJA (min CVaR)")

from scipy.optimize import minimize

def portfolio_cvar(weights, model_base, exposures_base, alpha=0.01):
    scaled_exposures = weights * exposures_base.sum()
    losses, VaR, CVaR = run_monte_carlo_sequential(model_base, 500, scaled_exposures, alpha, show_progress=False)
    return CVaR

initial_weights = np.ones(N) / N
bounds = [(0, 1) for _ in range(N)]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

print("   Pokrećem optimizaciju (500 simulacija po iteraciji, max 3 iteracije)...")
result = minimize(
    portfolio_cvar,
    initial_weights,
    args=(model, exposures),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 3, 'ftol': 1e-6, 'disp': True}
)

if result.success:
    optimal_weights = result.x
    print(f"✅ Optimizacija uspješna. Minimalni CVaR: ${result.fun:,.0f}")
    top_weights = pd.DataFrame({'Ticker': valid_tickers, 'Weight': optimal_weights}).sort_values('Weight', ascending=False)
    print("\n📋 Top 5 firmi u optimalnom portfoliju:")
    print(top_weights.head(5).to_string())
    print("\n📋 Bottom 5 firmi u optimalnom portfoliju (najmanje težine):")
    print(top_weights.tail(5).to_string())
else:
    print("❌ Optimizacija nije uspjela.")
    print(result.message)

print("\n✅ SVE NAPREDNE ANALIZE ZAVRŠENE!")

# ================================================================
# 13. RASPODELA OČEKIVANOG I NEOČEKIVANOG GUBITKA PO FIRMAMA
# ================================================================

print("\n" + "="*80)
print("=== 13. RASPODELA GUBITKA PO FIRMAMA (na osnovu DD) ===")
print("="*80)

pd_firm = stats.norm.cdf(-dd)
pd_firm = np.clip(pd_firm, 1e-6, 0.5)
lgd = 1 - model.recovery_base
ead_firm = exposures
el_firm = pd_firm * lgd * ead_firm
ul_firm = np.sqrt(pd_firm * (1 - pd_firm)) * lgd * ead_firm

el_df = pd.DataFrame({
    'Ticker': valid_tickers,
    'DD': dd,
    'PD (%)': pd_firm * 100,
    'EL ($)': el_firm,
    'UL ($)': ul_firm
}).sort_values('EL ($)', ascending=False)

sum_el = el_firm.sum()
sum_ul = np.sqrt((ul_firm**2).sum())
mc_el = losses_base.mean()
mc_ul = losses_base.std()

print("\n📊 POREĐENJE AGREGATNIH GUBITAKA:")
print(f"   Metoda           |  EL ($)     |  UL ($)")
print(f"   -------------------------------------------------")
print(f"   Suma po firmama  | ${sum_el:,.0f} | ${sum_ul:,.0f}")
print(f"   Monte Karlo      | ${mc_el:,.0f} | ${mc_ul:,.0f}")
print(f"   Razlika (EL)     | ${sum_el - mc_el:,.0f} ({((sum_el/mc_el)-1)*100:.1f}%)")

print("\n🔥 TOP 10 FIRMI PO OČEKIVANOM GUBITKU (EL):")
print(el_df.head(10).to_string(index=False, float_format=lambda x: f'{x:,.0f}'))

print("\n🟢 BOTTOM 10 FIRMI PO OČEKIVANOM GUBITKU (najmanji rizik):")
print(el_df.tail(10).to_string(index=False, float_format=lambda x: f'{x:,.0f}'))

fig, ax = plt.subplots(figsize=(14, 8))
top20 = el_df.head(20)
colors = ['red' if el > 500000 else 'orange' if el > 200000 else 'green' for el in top20['EL ($)']]
ax.barh(top20['Ticker'], top20['EL ($)'], color=colors, alpha=0.7)
ax.axvline(x=top20['EL ($)'].mean(), color='blue', linestyle='--', label=f'Prosjek: ${top20["EL ($)"].mean():,.0f}')
ax.set_xlabel('Očekivani gubitak (EL) po firmi ($)')
ax.set_title('Top 20 firmi po očekivanom gubitku (EL)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

try:
    sectors = [sector_map.get(t, 'Other') for t in valid_tickers]
    el_df['Sector'] = sectors
    sector_el = el_df.groupby('Sector')['EL ($)'].sum().sort_values(ascending=False)
    print("\n📊 OČEKIVANI GUBITAK PO SEKTORIMA:")
    print(sector_el.to_string(float_format=lambda x: f'${x:,.0f}'))
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_el.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_ylabel('Ukupan EL ($)')
    ax.set_title('Očekivani gubitak po sektorima')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠️ Sektorska analiza nije moguća: {e}")

print("\n📊 STATISTIKA RASPODJELE EL:")
print(f"   Ukupan EL (suma po firmama): ${sum_el:,.0f}")
print(f"   Prosečan EL po firmi:       ${el_firm.mean():,.0f}")
print(f"   Medijan EL po firmi:        ${np.median(el_firm):,.0f}")
print(f"   Std dev EL po firmi:        ${el_firm.std():,.0f}")
print(f"   Min EL:                     ${el_firm.min():,.0f} ({valid_tickers[np.argmin(el_firm)]})")
print(f"   Max EL:                     ${el_firm.max():,.0f} ({valid_tickers[np.argmax(el_firm)]})")
print(f"   Broj firmi sa EL > $500k:   {np.sum(el_firm > 500_000)}")
print(f"   Broj firmi sa EL > $100k:   {np.sum(el_firm > 100_000)}")

print("\n✅ RASPODELA GUBITKA PO FIRMAMA ZAVRŠENA!")

# ================================================================
# POKRETAČ AUTOMATSKE KALIBRACIJE
# ================================================================
optimal_params = calibrate_model(model, exposures, target_spread_bps=153, n_sims=2000, maxiter=20)
 if optimal_params is not None:
     print("\n📌 Preporučeni parametri za kalibraciju na tržišni spread (153 bps):")
     print(f"   jump_intensity = {optimal_params[0]:.4f}")
     print(f"   gamma_multiplier = {optimal_params[1]:.4f}")
     print(f"   recovery_base = {optimal_params[2]:.4f}")
     print(f"   recovery_sensitivity = {optimal_params[3]:.4f}")
 else:
     print("Kalibracija nije uspjela.")

