import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from pathlib import Path

from metrics.stats import compute_metrics

SRC_DIR = Path(__file__).resolve().parent.parent

# ── Colour helpers ─────────────────────────────────────────────────────────────
_STR_PALETTE = [
    'darkorange', 'mediumseagreen', 'mediumpurple',
    'sienna', 'teal', 'steelblue', 'goldenrod',
]

def _model_colors(models: dict) -> dict:
    """Int keys → Blues palette; str keys → distinct colours."""
    int_keys = sorted(k for k in models if isinstance(k, (int, float)))
    str_keys = [k for k in models if isinstance(k, str)]
    out = {}
    if int_keys:
        blues = cm.Blues(np.linspace(0.30, 0.85, max(len(int_keys), 1)))
        for k, c in zip(int_keys, blues):
            out[k] = c
    for i, k in enumerate(str_keys):
        out[k] = _STR_PALETTE[i % len(_STR_PALETTE)]
    return out

def _model_label(key) -> str:
    return f'L={key}' if isinstance(key, (int, float)) else str(key)

def _save(fig, path):
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        kw = {'bbox_inches': 'tight'}
        if p.suffix.lower() != '.pdf':
            kw['dpi'] = 300
        fig.savefig(str(p), **kw)
        print(f'[INFO]: Plot saved → {p}')


# ── Generalised notebook plot functions ────────────────────────────────────────

def plot_market_context(benchmark_norm: pd.Series, benchmark_ret: pd.Series,
                        benchmark_name: str = 'SP500', save_path=None):
    """Normalised benchmark price + daily return distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    axes[0].plot(benchmark_norm.index, benchmark_norm.values, color='steelblue', linewidth=2)
    axes[0].axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
    axes[0].set_title(f'{benchmark_name} — Performance OOS (normalizado)', fontsize=14)
    axes[0].set_xlabel('Data')
    axes[0].set_ylabel('Valor normalizado')
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].hist(benchmark_ret.values * 100, bins=30,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='gray', linewidth=1, linestyle='--')
    axes[1].axvline(benchmark_ret.mean() * 100, color='tomato', linewidth=1.5,
                    label=f'Média = {benchmark_ret.mean()*100:.3f}%')
    axes[1].set_title(f'Distribuição dos retornos diários {benchmark_name}', fontsize=14)
    axes[1].set_xlabel('Retorno diário (%)')
    axes[1].set_ylabel('Frequência')
    axes[1].legend()

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_performance(models: dict, benchmark: pd.Series,
                     title: str = None, save_path=None):
    """
    Normalised portfolio value for any number of models vs benchmark.

    models  : {key: pd.Series}  — int keys get Blues palette; str keys get
              distinct colours.  Int keys are labelled 'L={k}'; str keys as-is.
    benchmark: pd.Series — normalised benchmark (plotted in tomato dashed).
    """
    colors = _model_colors(models)
    fig, ax = plt.subplots(figsize=(18, 8))

    for key, series in models.items():
        color = colors[key]
        lw    = 1.4 if isinstance(key, (int, float)) else 2.8
        alpha = 0.85 if isinstance(key, (int, float)) else 1.0
        label = _model_label(key)
        ax.plot(series.index, series.values, color=color, linewidth=lw,
                alpha=alpha, label=label)
        ax.annotate(label, xy=(series.index[-1], series.iloc[-1]),
                    xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=9 if isinstance(key, (int, float)) else 11,
                    color=color,
                    fontweight='normal' if isinstance(key, (int, float)) else 'bold')

    ax.plot(benchmark.index, benchmark.values,
            color='tomato', linewidth=2.0, linestyle='--', label='SP500', zorder=5)
    ax.annotate('SP500', xy=(benchmark.index[-1], benchmark.iloc[-1]),
                xytext=(5, 0), textcoords='offset points',
                va='center', fontsize=10, color='tomato', fontweight='bold')

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.10)
    ax.set_title(title or 'Comparação de performance OOS', fontsize=16)
    ax.set_xlabel('Data', fontsize=13)
    ax.set_ylabel('Valor normalizado', fontsize=13)
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_dominance_overview(dominance: dict, detail_keys=None,
                             benchmark_name: str = 'SP500',
                             title: str = None, save_path=None):
    """
    Two-panel dominance plot.

    Left : SSD curves for all models.
    Right: SSD + FSD with fill for `detail_keys` (defaults to str-keyed models).

    dominance : {key: (s, fsd, ssd)}
    """
    colors = _model_colors(dominance)
    if detail_keys is None:
        detail_keys = [k for k in dominance if isinstance(k, str)]

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # ── Left: all models ──────────────────────────────────────────────────────
    ax = axes[0]
    for key, (s, fsd, ssd) in dominance.items():
        color = colors[key]
        lw    = 1.4 if isinstance(key, (int, float)) else 2.5
        label = _model_label(key)
        ax.plot(s, ssd, color=color, linewidth=lw,
                alpha=0.8 if isinstance(key, (int, float)) else 1.0, label=label)
        if isinstance(key, str):
            ax.annotate(label, xy=(s[-1], ssd[-1]), xytext=(5, 0),
                        textcoords='offset points', va='center',
                        fontsize=10, color=color, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=1.0, linestyle=':')
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.10)
    ax.set_title(f'Dominância SSD OOS vs {benchmark_name} — todos os modelos', fontsize=13)
    ax.set_xlabel('Rank do cenário (s)')
    ax.set_ylabel(f'SSD cumulativa vs {benchmark_name}')
    ax.legend(fontsize=9)

    # ── Right: detail for selected models ────────────────────────────────────
    ax = axes[1]
    for key in detail_keys:
        s, fsd, ssd = dominance[key]
        color = colors[key]
        label = _model_label(key)
        ax.plot(s, ssd, color=color, linewidth=2.0, label=f'SSD ({label})')
        ax.plot(s, fsd, color=color, linewidth=1.0, linestyle='--',
                alpha=0.6, label=f'FSD ({label})')
        ax.fill_between(s, fsd, ssd, color=color, alpha=0.12)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    detail_labels = ', '.join(_model_label(k) for k in detail_keys)
    ax.set_title(f'Detalhe dominância — {detail_labels}', fontsize=13)
    ax.set_xlabel('Rank do cenário (s)')
    ax.set_ylabel(f'Dominância vs {benchmark_name}')
    ax.legend(fontsize=9)

    _title = title or f'Curvas de dominância SSD (OOS vs {benchmark_name})'
    plt.suptitle(_title, fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_dominance_panels(dominance: dict, n_cols: int = 3,
                          benchmark_name: str = 'SP500',
                          title: str = None, save_path=None):
    """
    One subplot per model showing SSD + FSD dominance with violation shading.

    dominance : {key: (s, fsd, ssd)}
    n_cols    : columns in the subplot grid (rows computed automatically).
    """
    colors  = _model_colors(dominance)
    n       = len(dominance)
    n_cols  = min(n_cols, n)
    n_rows  = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 6 * n_rows), sharey=True)
    axes = np.array(axes).flatten()

    for ax, (key, (s, fsd, ssd)) in zip(axes, dominance.items()):
        color     = colors[key]
        label     = _model_label(key)
        dominates = ssd.min() >= 0
        status    = 'SSD ✓' if dominates else f'NÃO domina (min={ssd.min():.4f})'

        ax.plot(s, ssd, color=color, linewidth=2.2, label='SSD')
        ax.plot(s, fsd, color=color, linewidth=1.0, linestyle='--', alpha=0.6, label='FSD')
        ax.fill_between(s, fsd, ssd, color=color, alpha=0.14)
        ax.fill_between(s, np.minimum(ssd, 0), 0, color='tomato', alpha=0.25,
                        label='Violação SSD')
        ax.axhline(0, color='gray', linewidth=1.0, linestyle=':')
        ax.set_title(f'{label}\n{status}', fontsize=12,
                     color='green' if dominates else 'crimson')
        ax.set_xlabel(f'Rank do cenário (s)', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.tick_params(axis='x', rotation=20)

    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].set_ylabel(f'Dominância cumulativa vs {benchmark_name}', fontsize=11)
    plt.suptitle(title or f'Dominância SSD OOS por modelo vs {benchmark_name}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_risk_return_scatter(models: dict, benchmark: pd.Series,
                              rf_annual: float = 0.05,
                              title: str = None, save_path=None):
    """
    Two scatter panels: Volatility vs Sharpe  and  Max Drawdown vs CAGR.

    models    : {key: pd.Series} — normalised price series.
    benchmark : pd.Series        — normalised benchmark.
    """
    colors  = _model_colors(models)
    metrics = {k: compute_metrics(s, benchmark=benchmark, rf_annual=rf_annual)
               for k, s in models.items()}
    m_sp    = compute_metrics(benchmark, benchmark=benchmark, rf_annual=rf_annual)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (x_key, y_key, x_lbl, y_lbl, ax_title) in zip(axes, [
        ('Volatility (%)',   'Sharpe Ratio',    'Volatilidade anualizada (%)', 'Sharpe Ratio',
         'Risco × Retorno — Volatilidade vs Sharpe'),
        ('Max Drawdown (%)', 'CAGR (%)',         'Max Drawdown (%)',            'CAGR (%)',
         'Risco × Retorno — Drawdown vs CAGR'),
    ]):
        for key, m in metrics.items():
            color = colors[key]
            label = _model_label(key)
            size  = 80  if isinstance(key, (int, float)) else 160
            mark  = 'o' if isinstance(key, (int, float)) else 'D'
            kw    = dict(zorder=6, label=label) if isinstance(key, str) else dict(zorder=4)
            ax.scatter(m[x_key], m[y_key], color=color, s=size, marker=mark, **kw)
            ax.annotate(label, (m[x_key], m[y_key]),
                        xytext=(5, 3), textcoords='offset points',
                        fontsize=8 if isinstance(key, (int, float)) else 10,
                        color=color,
                        fontweight='normal' if isinstance(key, (int, float)) else 'bold')

        ax.scatter(m_sp[x_key], m_sp[y_key], color='tomato', s=120,
                   zorder=5, marker='*', label='SP500')
        ax.annotate('SP500', (m_sp[x_key], m_sp[y_key]),
                    xytext=(5, -10), textcoords='offset points',
                    fontsize=9, color='tomato')
        ax.set_xlabel(x_lbl, fontsize=12)
        ax.set_ylabel(y_lbl, fontsize=12)
        ax.set_title(ax_title, fontsize=13)
        ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        # ax.legend(fontsize=9)

    plt.suptitle(title or 'Fronteira de eficiência — todos os modelos (OOS)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_return_distribution(models: dict, benchmark: pd.Series,
                              highlight_keys=None, save_path=None):
    """
    Boxplot of daily returns for all models + overlapping histogram for key models.

    models        : {key: pd.Series} — normalised price series.
    benchmark     : pd.Series        — daily returns of benchmark (not normalised).
    highlight_keys: list of keys to include in the histogram panel; defaults to
                    str-keyed models + first int-keyed model.
    """
    colors  = _model_colors(models)
    sp_ret  = benchmark if benchmark.abs().max() < 1 else benchmark.pct_change().dropna()
    sp_pct  = sp_ret * 100

    ret_labels, ret_series_list = [], []
    for key, series in models.items():
        r = series.pct_change().dropna() * 100
        ret_labels.append(_model_label(key))
        ret_series_list.append(r.values)

    if highlight_keys is None:
        str_keys = [k for k in models if isinstance(k, str)]
        int_keys = sorted(k for k in models if isinstance(k, (int, float)))
        highlight_keys = str_keys + (int_keys[:1] if int_keys else [])

    box_colors = [colors[k] for k in models]
    bins = np.linspace(sp_pct.min() * 1.2, sp_pct.max() * 1.2, 60)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ── Boxplot ───────────────────────────────────────────────────────────────
    ax = axes[0]
    # `labels=` was removed from Axes.boxplot in matplotlib 3.11; set the ticks directly
    bp = ax.boxplot(ret_series_list, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker='.', markersize=3, alpha=0.4))
    ax.set_xticks(range(1, len(ret_labels) + 1))
    ax.set_xticklabels(ret_labels)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.axhline(sp_pct.mean(), color='tomato', linewidth=1.2, linestyle='--',
               label=f'SP500 média={sp_pct.mean():.3f}%')
    ax.set_ylabel('Retorno diário (%)', fontsize=11)
    ax.set_title('Distribuição de retornos diários', fontsize=12)
    ax.tick_params(axis='x', rotation=35)
    ax.legend(fontsize=9)

    # ── Histogram (key models only) ───────────────────────────────────────────
    ax = axes[1]
    ax.hist(sp_pct.values, bins=bins, color='tomato', alpha=0.35,
            label='SP500', density=True)
    for key in highlight_keys:
        r = models[key].pct_change().dropna() * 100
        ax.hist(r.values, bins=bins, color=colors[key], alpha=0.45,
                label=_model_label(key), density=True)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Retorno diário (%)', fontsize=11)
    ax.set_ylabel('Densidade', fontsize=11)
    ax.set_title('Histograma de retornos (modelos vs SP500)', fontsize=12)
    ax.legend(fontsize=9)

    plt.suptitle('Distribuição de retornos — modelos vs SP500',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_correlation_matrix(models: dict, benchmark: pd.Series,
                             benchmark_name: str = 'SP500', save_path=None):
    """
    Pearson correlation heatmap of daily returns across all models and benchmark.

    models    : {key: pd.Series} — normalised price series.
    benchmark : pd.Series        — daily returns of benchmark.
    """
    sp_ret = benchmark if benchmark.abs().max() < 1 else benchmark.pct_change().dropna()

    all_rets = {benchmark_name: sp_ret}
    for key, series in models.items():
        all_rets[_model_label(key)] = series.pct_change().dropna()

    ret_df   = pd.DataFrame(all_rets).dropna()
    corr_mat = ret_df.corr()
    labels   = corr_mat.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 1)))
    im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='Correlação de Pearson')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_mat.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color='black' if abs(val) < 0.8 else 'white')

    ax.set_title(f'Correlação de retornos diários entre modelos (OOS)', fontsize=14)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_drawdown(models: dict, benchmark: pd.Series,
                  fill_keys=None, title: str = None, save_path=None):
    """
    Drawdown curves for all models vs benchmark.

    models    : {key: pd.Series} — normalised price series.
    benchmark : pd.Series        — normalised benchmark.
    fill_keys : keys to shade with fill_between; defaults to str-keyed models.
    """
    colors   = _model_colors(models)
    if fill_keys is None:
        fill_keys = [k for k in models if isinstance(k, str)]

    fig, ax = plt.subplots(figsize=(18, 6))

    dd_sp = (benchmark - benchmark.cummax()) / benchmark.cummax() * 100
    ax.fill_between(dd_sp.index, dd_sp.values, 0, color='tomato', alpha=0.20, label='SP500')
    ax.plot(dd_sp.index, dd_sp.values, color='tomato', linewidth=1.5, linestyle='--')

    for key, series in models.items():
        color = colors[key]
        lw    = 1.3 if isinstance(key, (int, float)) else 2.5
        alpha = 0.7 if isinstance(key, (int, float)) else 1.0
        dd    = (series - series.cummax()) / series.cummax() * 100
        if key in fill_keys:
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.12)
        ax.plot(dd.index, dd.values, color=color, linewidth=lw,
                alpha=alpha, label=_model_label(key),
                zorder=6 if key in fill_keys else 4)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_title(title or 'Drawdown OOS — todos os modelos vs SP500', fontsize=14)
    ax.set_xlabel('Data')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(fontsize=10, loc='lower left')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_rolling_returns(models: dict, benchmark: pd.Series,
                          window: int = 21, title: str = None, save_path=None):
    """
    Rolling `window`-day return curves for all models vs benchmark.

    models    : {key: pd.Series} — normalised price series.
    benchmark : pd.Series        — normalised benchmark.
    """
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(18, 6))

    roll_sp = benchmark.pct_change(window).dropna() * 100
    ax.fill_between(roll_sp.index, roll_sp.values, 0, color='tomato', alpha=0.12)
    ax.plot(roll_sp.index, roll_sp.values, color='tomato',
            linewidth=1.5, linestyle='--', label='SP500')

    for key, series in models.items():
        color = colors[key]
        lw    = 1.3 if isinstance(key, (int, float)) else 2.5
        alpha = 0.7 if isinstance(key, (int, float)) else 1.0
        roll  = series.pct_change(window).dropna() * 100
        ax.plot(roll.index, roll.values, color=color, linewidth=lw,
                alpha=alpha, label=_model_label(key),
                zorder=6 if isinstance(key, str) else 4)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_title(title or f'Retorno rolante {window} dias (%) — todos os modelos vs SP500',
                 fontsize=14)
    ax.set_xlabel('Data')
    ax.set_ylabel(f'Retorno {window} dias (%)')
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_portfolio_comparison(portfolio, force_background=False,
                              benchmark_cols=None, title=None, save_path=None):
    """
    portfolio      : DataFrame with one portfolio column + optional benchmark/reference columns.
    benchmark_cols : list of column names to draw as dashed benchmark lines.
                     Defaults to ['SP500'] when SP500 is present (backward-compatible).
    title          : chart title (auto-generated if None).
    """
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    if force_background:
        ax.set_facecolor('white')
        ax.figure.set_facecolor('white')

    index_dt = pd.to_datetime(portfolio.index)

    # Resolve which columns are benchmarks (dashed) vs portfolio (solid)
    if benchmark_cols is None:
        _bench = {'SP500'} if 'SP500' in portfolio.columns else set()
    else:
        _bench = set(benchmark_cols)

    _always_skip  = {'Risk free rate'}
    portfolio_cols = [c for c in portfolio.columns if c not in _always_skip and c not in _bench]

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx   = 0

    line_labels = []  # (x_end, y_end, label, color) for end-of-line annotation

    for col in portfolio_cols:
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1
        series = portfolio[col].dropna()
        plt.plot(index_dt[:len(series)], series.values, linewidth=2.2, color=color)
        line_labels.append((index_dt[len(series) - 1], series.iloc[-1], col, color, False))

    for i, col in enumerate([c for c in portfolio.columns if c in _bench]):
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1
        series = portfolio[col].dropna()
        plt.plot(index_dt[:len(series)], series.values,
                 linestyle='--', linewidth=1.6, color=color, alpha=0.85)
        line_labels.append((index_dt[len(series) - 1], series.iloc[-1], col, color, True))

    if 'Risk free rate' in portfolio.columns:
        series = portfolio['Risk free rate'].dropna()
        plt.plot(index_dt[:len(series)], series.values,
                 linestyle=':', linewidth=1.2, color='gray')
        line_labels.append((index_dt[len(series) - 1], series.iloc[-1], 'Risk free rate', 'gray', True))

    # Expand x-axis by ~8% to give room for end-of-line labels
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.08)

    for x_end, y_end, label, color, is_bench in line_labels:
        ax.annotate(label, xy=(x_end, y_end), xytext=(6, 0),
                    textcoords='offset points', va='center',
                    fontsize=13, color=color,
                    fontweight='normal' if is_bench else 'bold')

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    _title = title or 'SSD — Comparativo de Performance'
    plt.title(_title, fontsize=20)
    plt.xlabel('Data', fontsize=16)
    plt.ylabel('Valor normalizado', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', colors='gray', labelsize=14)

    plt.tight_layout()
    _save(plt.gcf(), save_path)
    plt.show()


def plot_sweep(portfolios, sp500_norm, force_background=False):
    """
    Plot normalised portfolio value for each L value alongside SP500.
    Each portfolio line is annotated at its right end with the L value.

    Parameters
    ----------
    portfolios  : dict {L: pd.Series}  — from run_sweep()
    sp500_norm  : pd.Series            — normalised SP500 prices
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    if force_background:
        ax.set_facecolor('white')
        fig.set_facecolor('white')

    n      = len(portfolios)
    colors = cm.Blues(np.linspace(0.35, 0.9, n))

    for color, (L, series) in zip(colors, sorted(portfolios.items())):
        label      = f'GSSD  L={L}' + (' (SSD)' if L == 1 else '')
        x_vals     = pd.to_datetime(series.index)
        y_vals     = series.values
        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=1.8)

        # End-of-line annotation with the L value
        ax.annotate(
            f'L={L}',
            xy=(x_vals[-1], y_vals[-1]),
            xytext=(6, 0),
            textcoords='offset points',
            va='center',
            fontsize=11,
            color=color,
            fontweight='bold',
        )

    # SP500 reference in dashed red
    x_sp  = pd.to_datetime(sp500_norm.index)
    y_sp  = sp500_norm.values
    ax.plot(x_sp, y_sp, label='SP500', color='tomato', linewidth=2.2,
            linestyle='--', zorder=5)
    ax.annotate(
        'SP500',
        xy=(x_sp[-1], y_sp[-1]),
        xytext=(6, 0),
        textcoords='offset points',
        va='center',
        fontsize=11,
        color='tomato',
        fontweight='bold',
    )

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title('GSSD Parameter — L vs SP500', fontsize=20)
    plt.xlabel('Data', fontsize=16)
    plt.ylabel('Valor normalizado', fontsize=16)
    plt.legend(fontsize=13, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', colors='gray', labelsize=13)

    # Extra right margin so end-of-line labels aren't clipped
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.05)

    plt.tight_layout()

    output_dir = SRC_DIR / 'output' / 'gssd'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path   = output_dir / 'sweep_comparison.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f'[INFO]: Plot saved → {fig_path}')
    plt.show()


def plot_ssd_dominance_sweep(dominance_data, benchmark_name='SP500',
                              title=None, force_background=False):
    """
    Plot in-sample SSD (and FSD) dominance curves for each L value.

    For each L, the dominance relation is computed from the last training
    window's portfolio and benchmark returns (sorted ascending):
      FSD_s = r_p_(s) - r_b_(s)
      SSD_s = cumsum(r_p_(s) - r_b_(s))   for s = 1 .. S

    SSD ≥ 0 everywhere means the portfolio second-order stochastically
    dominates the benchmark.

    Parameters
    ----------
    dominance_data : dict {L: (s_indices, fsd, ssd)}  — from run_sweep()
    benchmark_name : str   — shown in axis label and title
    title          : str   — custom title (auto-generated if None)
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    if force_background:
        ax.set_facecolor('white')
        fig.set_facecolor('white')

    n      = len(dominance_data)
    colors = cm.Blues(np.linspace(0.35, 0.9, n))

    for color, (L, (s, fsd, ssd)) in zip(colors, sorted(dominance_data.items())):
        label = f'L={L}' + (' (SSD)' if L == 1 else '')
        ax.plot(s, ssd, label=label, color=color, linewidth=1.8)

        # FSD as a thin dashed line in the same colour
        ax.plot(s, fsd, color=color, linewidth=0.8, linestyle='--', alpha=0.5)

        # Shade between FSD and SSD to mirror the document's style
        ax.fill_between(s, fsd, ssd, color=color, alpha=0.08)

        # End-of-line annotation
        ax.annotate(
            f'L={L}',
            xy=(s[-1], ssd[-1]),
            xytext=(6, 0),
            textcoords='offset points',
            va='center',
            fontsize=11,
            color=color,
            fontweight='bold',
        )

    ax.axhline(0.0, color='gray', linewidth=1.0, linestyle=':')

    _title = title or f'In-sample SSD Dominance vs {benchmark_name} — GSSD '
    plt.title(_title, fontsize=20)
    plt.xlabel('Scenario rank (s)', fontsize=16)
    plt.ylabel(f'Cumulative dominance over {benchmark_name}', fontsize=16)
    plt.legend(fontsize=13, loc='best')
    plt.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', colors='gray', labelsize=13)

    # Extra right margin for end-of-line labels
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.05)

    plt.tight_layout()

    output_dir = SRC_DIR / 'output' / 'gssd'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path   = output_dir / 'ssd_dominance_sweep.png'
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f'[INFO]: Plot saved → {fig_path}')
    plt.show()
