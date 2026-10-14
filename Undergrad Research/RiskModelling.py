# ==============================================================================
# RISK MODELLING PIPELINE - OBJECT-ORIENTED VERSION
# Indices: IBOVESPA, S&P 500, STOXX 600, Hang Seng
# ==============================================================================

import os
import time
import math
import warnings
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import xlogy

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. CENTRALIZED CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    """All project parameters in a single place."""
    start_date: str = "2005-01-01"
    end_date: str = "2026-05-31"
    tickers: dict = field(default_factory=lambda: {
        "^BVSP": "IBOVESPA",
        "^GSPC": "S&P 500",
        "^STOXX": "STOXX 600",
        "^HSI": "Hang Seng",
    })
    colors: dict = field(default_factory=lambda: {
        "IBOVESPA": "green", "S&P 500": "blue",
        "STOXX 600": "orange", "Hang Seng": "red",
    })
    alpha: float = 0.01                 # 99% confidence level
    garch_dist: str = "studentst"
    corr_window: int = 60
    gpd_threshold: float = 0.90
    var_methods: tuple = ("Normal", "Cornish-Fisher", "t-Student", "Johnson SU", "Historical")
    n_simulations: int = 100_000
    seed: int = 42
    max_retries: int = 3
    cache_file: str = "prices_cache.csv"
    output_dir: str = "outputs"
    show_plots: bool = True             # False => only saves the figures (useful on a server)
    print_model_summary: bool = True


# ==============================================================================
# 2. MARKET DATA
# ==============================================================================
class MarketData:
    """Downloads (with retry and CSV cache), cleans and converts prices into returns."""

    def __init__(self, config: Config):
        self.cfg = config
        self.prices = None
        self.log_returns = None

    def _download(self):
        """download from Yahoo Finance up to `max_retries` times (while loop)."""
        import yfinance as yf  

        attempt = 0
        while attempt < self.cfg.max_retries:
            attempt += 1
            try:
                raw = yf.download(list(self.cfg.tickers), start=self.cfg.start_date,
                                  end=self.cfg.end_date, auto_adjust=True, progress=False)
                close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
                if close is not None and not close.empty:
                    return close
                print(f"  Attempt {attempt}: empty download.")
            except Exception as err:
                print(f"  Attempt {attempt} failed: {err}")
            time.sleep(2 * attempt)
        return None

    def fetch(self):
        print(f"Downloading data from {self.cfg.start_date} to {self.cfg.end_date}...")
        close = None
        try:
            close = self._download()
        except ImportError:
            print("  yfinance is not installed (pip install yfinance).")

        if close is not None:
            close.to_csv(self.cfg.cache_file)          # saves cache for offline use
        elif os.path.exists(self.cfg.cache_file):
            print(f"  Using local cache: {self.cfg.cache_file}")
            close = pd.read_csv(self.cfg.cache_file, index_col=0, parse_dates=True)
        else:
            raise RuntimeError("Could not obtain the data (no internet and no cache).")

        close = close.rename(columns=self.cfg.tickers)
        ordered = [n for n in self.cfg.tickers.values() if n in close.columns]
        self.prices = close[ordered].dropna()
        return self.prices

    def compute_log_returns(self):
        if self.prices is None:
            self.fetch()
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        return self.log_returns


# ==============================================================================
# 3. STATISTICS AND TESTS
# ==============================================================================
class SeriesDiagnostics:
    """Descriptive statistics and tests (normality, stationarity, ARCH)."""

    @staticmethod
    def descriptive(returns: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Mean": returns.mean(), "Variance": returns.var(), "Std Dev": returns.std(),
            "Skewness": returns.skew(), "Kurtosis": returns.kurt(),
        })

    @staticmethod
    def raw_return_tests(returns: pd.DataFrame) -> pd.DataFrame:
        """KS (normality), ADF (stationarity), ARCH-LM (heteroscedasticity)."""
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.stats.diagnostic import het_arch

        rows = []
        for col in returns.columns:
            series = returns[col].dropna()
            standardized = (series - series.mean()) / series.std()
            try:
                ks_p = stats.kstest(standardized, "norm")[1]
                adf_p = adfuller(series)[1]
                arch_p = het_arch(series)[1]
            except Exception as err:
                print(f"  Test failed for {col}: {err}")
                ks_p = adf_p = arch_p = np.nan
            rows.append({"Index": col, "KS p-value": ks_p,
                         "ADF p-value": adf_p, "ARCH p-value": arch_p})
        return pd.DataFrame(rows).set_index("Index")

    @staticmethod
    def residual_tests(std_resid: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
        """Ljung-Box, ARCH-LM and KS on the EGARCH standardized residuals."""
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

        rows = {}
        for col in std_resid.columns:
            series = std_resid[col].to_numpy(dtype=float)
            try:
                lb_p = float(acorr_ljungbox(series, lags=[lags], return_df=True).iloc[0, 1])
                arch_p = het_arch(series, nlags=lags)[1]
                ks_p = stats.kstest(series, "norm", args=(series.mean(), series.std()))[1]
            except Exception as err:
                print(f"  Diagnostic failed for {col}: {err}")
                lb_p = arch_p = ks_p = np.nan
            rows[col] = (lb_p, arch_p, ks_p)
        return pd.DataFrame(rows, index=["Ljung-Box p", "ARCH-LM p", "KS p"]).T


# ==============================================================================
# 4. AR(1) FILTER AND EGARCH MODEL
# ==============================================================================
class ARFilter:
    """Removes linear autocorrelation from returns with an AR(1)."""

    def __init__(self, order=(1, 0, 0)):
        self.order = order

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        from statsmodels.tsa.arima.model import ARIMA

        out = pd.DataFrame(index=returns.index)
        for col in returns.columns:
            try:
                out[col] = ARIMA(returns[col].dropna(), order=self.order).fit().resid
            except Exception as err:
                print(f"  AR(1) failed for {col} ({err}); using unfiltered returns.")
                out[col] = returns[col]
        return out

class VolatilityModeler:
    """Fits an EGARCH(1,1,1) to each index and extracts standardized residuals."""

    def __init__(self, config: Config):
        self.cfg = config
        self.results = {}       # name -> arch result
        self.cond_vol = {}      # name -> conditional volatility (%)
        self.std_resid = {}     # name -> standardized residuals
        self.pct_returns = {}   # name -> series in % used in the fit

    def fit(self, name: str, series: pd.Series):
        from arch import arch_model

        pct = series.dropna() * 100            # % scale helps the optimizer
        result = None
        for dist in (self.cfg.garch_dist, "normal"):   # fallback in case the Student-t fails
            try:
                model = arch_model(pct, vol="EGARCH", p=1, o=1, q=1, dist=dist, rescale=False)
                result = model.fit(disp="off")
                if dist != self.cfg.garch_dist:
                    print(f"  {name}: using '{dist}' distribution (fallback).")
                break
            except Exception as err:
                print(f"  {name}: EGARCH with '{dist}' failed: {err}")
        if result is None:
            raise RuntimeError(f"EGARCH did not converge for {name}.")

        self.results[name] = result
        self.pct_returns[name] = pct
        self.cond_vol[name] = result.conditional_volatility
        self.std_resid[name] = result.resid / result.conditional_volatility

        if self.cfg.print_model_summary:
            print(result.summary())
        return result

    def fit_all(self, ar_residuals: pd.DataFrame):
        for name in ar_residuals.columns:
            print(f"\nFitting EGARCH: {name}")
            self.fit(name, ar_residuals[name])
        return self.standardized_dataframe()

    def standardized_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.std_resid).astype(float).dropna()

    def news_impact_curve(self, name: str, z=None):
        """News impact curve (leverage effect) from alpha and gamma."""
        params = self.results[name].params
        z = np.linspace(-5, 5, 100) if z is None else z
        log_var = params["alpha[1]"] * (np.abs(z) - np.sqrt(2 / np.pi)) + params["gamma[1]"] * z
        return z, np.exp(log_var)


# ==============================================================================
# 5. EXTREME VALUE THEORY (EVT)
# ==============================================================================
class EVTAnalyzer:
    """Mean Excess, Hill, PIT e ajuste da GPD (Peaks-Over-Threshold)."""

    def __init__(self, config: Config):
        self.cfg = config

    @staticmethod
    def left_tail_losses(residuals) -> np.ndarray:
        """Positive losses = -negative residuals (left tail)."""
        data = np.asarray(residuals, dtype=float)
        data = data[~np.isnan(data)]
        return -data[data < 0]

    @staticmethod
    def mean_excess(losses: np.ndarray):
        ordered = np.sort(losses)
        cut = max(1, int(len(ordered) * 0.05))      # drops the top 5% (too few observations)
        thresholds = ordered[:-cut]
        means = [np.mean(ordered[ordered > u] - u) for u in thresholds]
        return thresholds, np.array(means)

    @staticmethod
    def hill(losses: np.ndarray):
        y = np.sort(losses)[::-1]
        n = len(y)
        k_values, xis = [], []
        k = 10
        while k < int(n * 0.20):                    # k between 10 and 20% of the sample
            xis.append(np.mean(np.log(y[:k]) - np.log(y[k])))
            k_values.append(k)
            k += 1
        return np.array(k_values), np.array(xis)

    @staticmethod
    def pseudo_observations(std_resid: pd.DataFrame) -> pd.DataFrame:
        """Empirical probability integral transform (PIT) via ranks."""
        return std_resid.rank(method="average") / (len(std_resid) + 1)

    def fit_gpd(self, std_resid: pd.DataFrame):
        """Fits the GPD to excesses above threshold u (90th percentile of losses)."""
        params, excess_data = {}, {}
        for col in std_resid.columns:
            losses = -std_resid[col].to_numpy(dtype=float)
            u = np.percentile(losses, self.cfg.gpd_threshold * 100)
            excesses = losses[losses > u] - u        # excesses = loss - threshold
            if len(excesses) < 20:
                print(f"  {col}: too few excesses ({len(excesses)}), GPD skipped.")
                continue
            try:
                xi, _, sigma = stats.genpareto.fit(excesses, floc=0)
            except Exception as err:
                print(f"  {col}: GPD fit failed: {err}")
                continue
            params[col] = {"Threshold (u)": u, "Shape (xi)": xi, "Scale (sigma)": sigma,
                           "N total": len(losses), "N excesses": len(excesses)}
            excess_data[col] = excesses
        return pd.DataFrame(params).T, excess_data


# ==============================================================================
# 6. VaR / CVaR
# ==============================================================================
class RiskMetrics:
    """VaR and CVaR using several methods (Normal, Historical, t, Johnson SU, Cornish-Fisher)."""

    def __init__(self, config: Config):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def _simulated_cvar(self, sample: np.ndarray, var: float) -> float:
        tail = sample[sample <= var]
        return float(tail.mean()) if tail.size > 0 else np.nan

    def calculate(self, series: pd.Series) -> dict:
        """Returns {method: (VaR, CVaR)} in decimal."""
        a = self.cfg.alpha
        n_sim = self.cfg.n_simulations
        s = series.dropna()
        r = s.to_numpy(dtype=float)
        mu, sigma, skew, kurt = s.mean(), s.std(ddof=1), s.skew(), s.kurtosis()

        out = {}
        for method in self.cfg.var_methods:
            var = cvar = np.nan
            try:
                if method == "Normal":
                    z = stats.norm.ppf(a)
                    var = mu + sigma * z
                    cvar = mu - sigma * stats.norm.pdf(z) / a
                elif method == "Historical":
                    var = np.percentile(r, a * 100)
                    cvar = r[r <= var].mean()
                elif method == "Johnson SU":
                    p = stats.johnsonsu.fit(r)
                    var = stats.johnsonsu.ppf(a, *p[:2], loc=p[2], scale=p[3])
                    sim = stats.johnsonsu.rvs(*p[:2], loc=p[2], scale=p[3],
                                              size=n_sim, random_state=self.rng)
                    cvar = self._simulated_cvar(sim, var)
                elif method == "t-Student":
                    df_t, loc_t, scale_t = stats.t.fit(r)
                    var = stats.t.ppf(a, df_t, loc=loc_t, scale=scale_t)
                    sim = stats.t.rvs(df_t, loc=loc_t, scale=scale_t,
                                      size=n_sim, random_state=self.rng)
                    cvar = self._simulated_cvar(sim, var)
                elif method == "Cornish-Fisher":
                    z = stats.norm.ppf(a)
                    z_cf = (z + (1/6) * (z**2 - 1) * skew
                            + (1/24) * (z**3 - 3*z) * kurt
                            - (1/36) * (2*z**3 - 5*z) * skew**2)
                    var = mu + sigma * z_cf          # CVaR has no simple closed form -> NaN
                else:
                    print(f"  Unknown method skipped: {method}")
                    continue
            except Exception as err:
                print(f"  {method} failed: {err}")
            out[method] = (var, cvar)
        return out

    def calculate_all(self, returns: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in returns.columns:
            for method, (var, cvar) in self.calculate(returns[col]).items():
                rows.append({"Index": col, "Method": method,
                             "VaR": var, "CVaR": cvar})
        return pd.DataFrame(rows)


# ==============================================================================
# 7. BACKTESTING
# ==============================================================================
class BacktestEngine:
    """Kupiec (unconditional coverage), Christoffersen (independence) and joint test."""

    @staticmethod
    def kupiec(returns: np.ndarray, var: float, alpha: float):
        n = len(returns)
        v = int(np.sum(returns < var))
        p_hat = v / n if n > 0 else 0.0
        # xlogy(0, 0) = 0, so v == 0 or v == n need no special handling
        lr = -2 * (xlogy(n - v, 1 - alpha) + xlogy(v, alpha)) \
             + 2 * (xlogy(n - v, 1 - p_hat) + xlogy(v, p_hat))
        p_value = 1 - stats.chi2.cdf(lr, 1)
        return v, n * alpha, lr, p_value

    @staticmethod
    def christoffersen(returns: np.ndarray, var: float):
        hits = (returns < var).astype(int)
        prev, curr = hits[:-1], hits[1:]
        t00 = int(np.sum((prev == 0) & (curr == 0)))
        t01 = int(np.sum((prev == 0) & (curr == 1)))
        t10 = int(np.sum((prev == 1) & (curr == 0)))
        t11 = int(np.sum((prev == 1) & (curr == 1)))

        pi01 = t01 / (t00 + t01) if (t00 + t01) > 0 else 0.0
        pi11 = t11 / (t10 + t11) if (t10 + t11) > 0 else 0.0
        total = t00 + t01 + t10 + t11
        pi_hat = (t01 + t11) / total if total > 0 else 0.0

        try:
            ll_null = xlogy(t00 + t10, 1 - pi_hat) + xlogy(t01 + t11, pi_hat)
            ll_alt = (xlogy(t00, 1 - pi01) + xlogy(t01, pi01)
                      + xlogy(t10, 1 - pi11) + xlogy(t11, pi11))
            lr = max(-2 * (ll_null - ll_alt), 0.0)
            p_value = 1 - stats.chi2.cdf(lr, 1)
        except (ValueError, FloatingPointError):
            lr, p_value = 0.0, 1.0
        return lr, p_value

    @staticmethod
    def verdict(p_value: float, level: float = 0.05) -> str:
        return "Accepted" if p_value > level else "Rejected"

    @classmethod
    def run(cls, returns: pd.DataFrame, var_table: pd.DataFrame, alpha: float) -> pd.DataFrame:
        rows = []
        for _, row in var_table.iterrows():
            idx, method, var = row["Index"], row["Method"], row["VaR"]
            if idx not in returns.columns or pd.isna(var):
                continue
            r = returns[idx].dropna().to_numpy(dtype=float)
            viol, expected, lr_uc, p_uc = cls.kupiec(r, var, alpha)
            lr_ind, p_ind = cls.christoffersen(r, var)
            p_joint = 1 - stats.chi2.cdf(lr_uc + lr_ind, 2)
            rows.append({
                "Index": idx, "Method": method,
                "Violations": viol, "Expected": round(expected, 1),
                "Kupiec p": round(p_uc, 4), "Kupiec": cls.verdict(p_uc),
                "Christoffersen p": round(p_ind, 4), "Christoffersen": cls.verdict(p_ind),
                "Joint p": round(p_joint, 4), "Joint": cls.verdict(p_joint),
            })
        return pd.DataFrame(rows)


# ==============================================================================
# 8. PLOTS
# ==============================================================================
class Plotter:
    """All project plots. Saves to PNG and (optionally) displays them."""

    def __init__(self, config: Config):
        self.cfg = config
        sns.set_theme(style="whitegrid")
        os.makedirs(config.output_dir, exist_ok=True)

    # ---------- utilities ----------
    def _finish(self, fig, filename: str):
        fig.tight_layout()
        try:
            fig.savefig(os.path.join(self.cfg.output_dir, filename), dpi=140, bbox_inches="tight")
        except OSError as err:
            print(f"  Could not save {filename}: {err}")
        if self.cfg.show_plots:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _grid(n: int, figsize=(16, 10)):
        nrows = math.ceil(n / 2)
        fig, axes = plt.subplots(nrows, 2, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        for ax in axes[n:]:                 # hides unused axes
            ax.set_visible(False)
        return fig, axes

    def _color(self, name):
        return self.cfg.colors.get(name, "steelblue")

    # ---------- prices and returns ----------
    def prices(self, prices: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(14, 7))
        for col in prices.columns:
            ax.plot(prices.index, prices[col], label=col, color=self._color(col))
        ax.set(title="Stock Market Indices", ylabel="Closing Price")
        ax.legend()
        self._finish(fig, "01_prices.png")

    def log_returns(self, returns: pd.DataFrame):
        fig, axes = self._grid(returns.shape[1], (16, 9))
        for ax, col in zip(axes, returns.columns):
            ax.plot(returns.index, returns[col], color=self._color(col), linewidth=0.8)
            ax.set(title=f"Log Returns - {col}", ylabel="Log Return")
        self._finish(fig, "02_log_returns.png")

    def distributions(self, returns: pd.DataFrame):
        fig, axes = self._grid(returns.shape[1], (16, 9))
        for ax, col in zip(axes, returns.columns):
            series = returns[col]
            sns.histplot(series, kde=True, bins=60, stat="density", color="royalblue", ax=ax)
            mu, sd = stats.norm.fit(series)
            x = np.linspace(series.min(), series.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, sd), "r", linewidth=2, label="Theoretical Normal")
            ax.set(title=f"Distribution of Log Returns - {col}", xlabel="Log Return")
            ax.legend()
        self._finish(fig, "03_distributions.png")

    def stats_overview(self, prices: pd.DataFrame, returns: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        normalized = prices / prices.iloc[0] * 100
        for col in normalized.columns:
            axes[0].plot(normalized.index, normalized[col], label=col, color=self._color(col))
        axes[0].set(title="Normalized Evolution (base 100)", ylabel="Points")
        axes[0].legend()
        for col in returns.columns:
            sns.kdeplot(returns[col], ax=axes[1], label=col, fill=True, alpha=0.3,
                        color=self._color(col))
        axes[1].set(title="Returns Density (Fat Tails)", xlabel="Daily Return", xlim=(-0.08, 0.08))
        axes[1].legend()
        self._finish(fig, "04_overview.png")

    def qq_plots(self, returns: pd.DataFrame):
        fig, axes = self._grid(returns.shape[1], (16, 12))
        for ax, col in zip(axes, returns.columns):
            std = (returns[col] - returns[col].mean()) / returns[col].std()
            stats.probplot(std, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {col} vs Standard Normal")
            ax.get_lines()[0].set_markerfacecolor("blue")
            ax.get_lines()[0].set_markeredgecolor("blue")
        self._finish(fig, "05_qqplots.png")

    def rolling_correlation(self, returns: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(16, 7))
        pairs = list(itertools.combinations(returns.columns, 2))
        palette = sns.color_palette("husl", len(pairs))
        for color, (c1, c2) in zip(palette, pairs):
            corr = returns[c1].rolling(self.cfg.corr_window).corr(returns[c2])
            ax.plot(corr.index, corr, label=f"{c1} x {c2}", color=color, alpha=0.8)
        ax.set(title=f"Rolling Correlation ({self.cfg.corr_window}-day Window)",
               ylabel="Pearson Correlation")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        self._finish(fig, "06_rolling_correlation.png")

    # ---------- AR(1) and EGARCH ----------
    def ar_residuals(self, residuals: pd.DataFrame):
        fig, axes = self._grid(residuals.shape[1], (16, 9))
        for ax, col in zip(axes, residuals.columns):
            ax.plot(residuals.index, residuals[col], color="steelblue", linewidth=0.8)
            ax.axhline(0, color="red", linestyle="--", alpha=0.7)
            ax.set(title=f"AR(1) Residuals - {col}", ylabel="Residuals")
        self._finish(fig, "07_ar_residuals.png")

    def egarch_panel(self, name: str, returns_pct: pd.Series, cond_vol: pd.Series,
                     std_resid: pd.Series, nic_z, nic_vol):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1.5], hspace=0.4)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(returns_pct.index, returns_pct, label="Log Returns (%)", color="blue", linewidth=0.7)
        ax1.plot(cond_vol.index, cond_vol, color="red", label="Conditional Volatility (EGARCH)")
        ax1.plot(cond_vol.index, -cond_vol, color="red")
        ax1.set(title=f"Conditional Volatility - {name}", ylabel="Return / Volatility")
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(std_resid.index, std_resid, color="steelblue", linewidth=0.7,
                 label="Standardized Residuals")
        ax2.set(title=f"Standardized Residuals - {name}", ylabel="Std. Residuals")
        ax2.legend(loc="upper right")

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(nic_z, nic_vol, color="purple", linewidth=2.5)
        ax3.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax3.set(title=f"News Impact Curve - {name}", xlabel="Standardized Shock",
                ylabel="Impact on Volatility")
        self._finish(fig, f"08_egarch_{name.replace(' ', '_').replace('&', 'and')}.png")

    # ---------- EVT ----------
    def mean_excess(self, data: dict):
        """data: {nome: (thresholds, mean_excess, u90)}"""
        fig, axes = self._grid(len(data), (16, 12))
        for ax, (name, (thr, me, u90)) in zip(axes, data.items()):
            ax.plot(thr, me, color="darkblue", linewidth=2)
            ax.axvline(u90, color="red", linestyle="--", label=f"90th percentile (u={u90:.2f})")
            ax.set(title=f"Mean Excess Plot - {name}", xlabel="Loss threshold (u)",
                   ylabel="Mean Excess")
            ax.legend()
        self._finish(fig, "09_mean_excess.png")

    def hill(self, data: dict):
        """data: {nome: (k_values, xi_values)}"""
        fig, axes = self._grid(len(data), (16, 12))
        for ax, (name, (k, xi)) in zip(axes, data.items()):
            ax.plot(k, xi, color="purple", linewidth=2)
            if len(xi) >= 3:                                  # stability zone
                third = len(xi) // 3
                zone = np.median(xi[third: 2 * third])
                ax.axhline(zone, color="red", linestyle="--", alpha=0.7,
                           label=f"Stability zone (~{zone:.3f})")
                ax.legend()
            ax.set(title=f"Hill Plot - {name}", xlabel="Number of extremes (k)",
                   ylabel="Shape estimate (xi)")
        self._finish(fig, "10_hill.png")

    def pit(self, pseudo_obs: pd.DataFrame):
        fig, axes = self._grid(pseudo_obs.shape[1], (16, 10))
        for ax, col in zip(axes, pseudo_obs.columns):
            sns.histplot(pseudo_obs[col], bins=50, stat="density", color="teal", ax=ax)
            ax.set(title=f"Pseudo-observations (PIT) - {col}", xlim=(0, 1))
        self._finish(fig, "11_pit.png")

    def gpd(self, excess_data: dict, gpd_params: pd.DataFrame):
        fig, axes = self._grid(len(excess_data), (16, 10))
        for ax, (name, exc) in zip(axes, excess_data.items()):
            xi, sigma = gpd_params.loc[name, "Shape (xi)"], gpd_params.loc[name, "Scale (sigma)"]
            sns.histplot(exc, bins=30, stat="density", color="lightcoral", alpha=0.6,
                         ax=ax, label="Empirical excesses")
            x = np.linspace(0, exc.max(), 200)
            ax.plot(x, stats.genpareto.pdf(x, c=xi, loc=0, scale=sigma), color="darkred",
                    linewidth=2.5, label=f"GPD fit (xi={xi:.3f})")
            ax.set(title=f"GPD Fit on Extreme Losses - {name}", xlabel="Loss exceedance")
            ax.legend()
        self._finish(fig, "15_gpd.png")

    # ---------- VaR / CVaR ----------
    def traditional_var(self, returns: pd.DataFrame, table: pd.DataFrame):
        fig, axes = self._grid(returns.shape[1], (16, 12))
        styles = {"Normal": ("royalblue", "-"), "Historical": ("firebrick", "--")}
        for ax, col in zip(axes, returns.columns):
            r = returns[col].dropna().to_numpy(dtype=float)
            sns.histplot(x=r, bins=80, stat="density", color="lightgray", edgecolor="none", ax=ax)
            sub = table[table["Index"] == col].set_index("Method")
            for method, (color, ls) in styles.items():
                if method not in sub.index:
                    continue
                var, cvar = sub.loc[method, "VaR"], sub.loc[method, "CVaR"]
                ax.axvline(var, color=color, linestyle=ls, linewidth=2.2,
                           label=f"VaR {method}: {var*100:.2f}%")
                if not pd.isna(cvar):
                    ax.axvline(cvar, color=color, linestyle=":", linewidth=2.2,
                               label=f"CVaR {method}: {cvar*100:.2f}%")
            ax.set(title=f"Traditional Risk Metrics (99%) - {col}",
                   xlabel="Daily Returns", xlim=(r.min(), 0.05))
            ax.legend(fontsize=8)
        self._finish(fig, "12_var_traditional.png")

    def var_comparison(self, pivot: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(14, 7))
        pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=1.2)
        ax.set(title="VaR (99%): Normal vs Alternative Models vs Historical",
               ylabel="VaR (%) - negative = loss", xlabel="Index")
        plt.setp(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Methods", loc="lower left")
        self._finish(fig, "13_var_comparison.png")

    def heatmap(self, backtest: pd.DataFrame, column: str, title: str, filename: str):
        order = [m for m in self.cfg.var_methods if m in backtest["Method"].unique()]
        labels = backtest.pivot(index="Index", columns="Method", values=column)[order]
        numeric = labels.replace({"Accepted": 1, "Rejected": 0}).astype(float)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric, annot=labels, fmt="", cmap=sns.color_palette(["#e74c3c", "#2ecc71"]),
                    cbar=False, linewidths=1, ax=ax, annot_kws={"weight": "bold"})
        ax.set(title=title, ylabel="Index", xlabel="VaR Model")
        self._finish(fig, filename)


# ==============================================================================
# 9. ORCHESTRATOR
# ==============================================================================
class RiskPipeline:
    """Runs all steps in order and stores the results."""

    def __init__(self, config: Config = None):
        self.cfg = config or Config()
        self.data = MarketData(self.cfg)
        self.diag = SeriesDiagnostics()
        self.ar = ARFilter()
        self.vol = VolatilityModeler(self.cfg)
        self.evt = EVTAnalyzer(self.cfg)
        self.risk = RiskMetrics(self.cfg)
        self.plot = Plotter(self.cfg)
        self.res = {}          # results (tables) available after run()

    @staticmethod
    def _show(title: str, df: pd.DataFrame, digits: int = 4):
        print(f"\n=== {title} ===")
        print(df.round(digits).to_string())

    # ---------- steps ----------
    def step_data(self):
        prices = self.data.fetch()
        returns = self.data.compute_log_returns()
        self.plot.prices(prices)
        self.plot.log_returns(returns)

    def step_statistics(self):
        returns = self.data.log_returns
        self.res["descriptive"] = self.diag.descriptive(returns)
        self._show("Descriptive statistics", self.res["descriptive"], 6)
        self.res["raw_tests"] = self.diag.raw_return_tests(returns)
        self._show("Tests: KS (normality) | ADF (stationarity) | ARCH (clustering)",
                   self.res["raw_tests"])
        self.plot.distributions(returns)
        self.plot.stats_overview(self.data.prices, returns)
        self.plot.qq_plots(returns)
        self.plot.rolling_correlation(returns)

    def step_ar_filter(self):
        self.res["ar_resid"] = self.ar.residuals(self.data.log_returns)
        self.plot.ar_residuals(self.res["ar_resid"])

    def step_egarch(self):
        std_resid = self.vol.fit_all(self.res["ar_resid"])
        self.res["std_resid"] = std_resid
        for name in self.vol.results:
            z, nic = self.vol.news_impact_curve(name)
            self.plot.egarch_panel(name, self.vol.pct_returns[name], self.vol.cond_vol[name],
                                   self.vol.std_resid[name], z, nic)
        self.res["resid_tests"] = self.diag.residual_tests(std_resid)
        self._show("Standardized residuals diagnostics (p < 0.05 => issue)",
                   self.res["resid_tests"])

    def step_evt_diagnostics(self):
        std_resid = self.res["std_resid"]
        me_data, hill_data = {}, {}
        for col in std_resid.columns:
            losses = self.evt.left_tail_losses(std_resid[col])
            if losses.size < 50:                      # sample too small
                print(f"  {col}: insufficient losses for EVT.")
                continue
            thr, me = self.evt.mean_excess(losses)
            me_data[col] = (thr, me, np.percentile(losses, 90))
            hill_data[col] = self.evt.hill(losses)
        self.plot.mean_excess(me_data)
        self.plot.hill(hill_data)
        self.plot.pit(self.evt.pseudo_observations(std_resid))

    def step_var(self):
        returns = self.data.log_returns
        table = self.risk.calculate_all(returns)
        self.res["var_table"] = table
        pivot = (table.pivot(index="Index", columns="Method", values="VaR") * 100)
        pivot = pivot[[m for m in self.cfg.var_methods if m in pivot.columns]]
        self._show("VaR 99% by method (%)", pivot, 2)
        cvar = (table.pivot(index="Index", columns="Method", values="CVaR") * 100)
        self._show("CVaR 99% by method (%)", cvar[pivot.columns], 2)
        self.plot.traditional_var(returns, table)
        self.plot.var_comparison(pivot)

    def step_backtest(self):
        bt = BacktestEngine.run(self.data.log_returns, self.res["var_table"], self.cfg.alpha)
        self.res["backtest"] = bt
        self._show("Backtesting (Kupiec / Christoffersen / Joint)", bt, 4)
        self.plot.heatmap(bt, "Kupiec", "Kupiec POF Test", "14a_kupiec.png")
        self.plot.heatmap(bt, "Joint", "Joint Conditional Coverage", "14b_joint.png")

    def step_gpd(self):
        params, excesses = self.evt.fit_gpd(self.res["std_resid"])
        self.res["gpd"] = params
        if params.empty:
            print("No GPD fitted.")
        else:
            self._show("GPD parameters (left tail)", params, 4)
            self.plot.gpd(excesses, params)

    # ---------- execution ----------
    def run(self):
        steps = [
            ("1. Data and returns",            self.step_data,            True),
            ("2. Statistics and tests",       self.step_statistics,      False),
            ("3. AR(1) filter",                self.step_ar_filter,       True),
            ("4. EGARCH",                      self.step_egarch,          True),
            ("5. EVT diagnostics (ME/Hill/PIT)", self.step_evt_diagnostics, False),
            ("6. VaR / CVaR",                  self.step_var,             True),
            ("7. Backtesting",                 self.step_backtest,        False),
            ("8. GPD",                         self.step_gpd,             False),
        ]
        for label, func, critical in steps:
            print("\n" + "=" * 80 + f"\n{label}\n" + "=" * 80)
            try:
                func()
            except Exception as err:
                print(f"[ERROR] {label}: {type(err).__name__}: {err}")
                if critical:
                    print("Critical step failed; stopping the pipeline.")
                    break
        print(f"\nPlots saved to: {os.path.abspath(self.cfg.output_dir)}")
        return self.res


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    config = Config(show_plots=True)       # show_plots=False to only save the PNGs
    RiskPipeline(config).run()