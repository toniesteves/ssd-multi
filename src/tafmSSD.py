
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blotter.plots import plot_portfolio_comparison
from metrics.stats import compute_ssd_dominance

from docplex.mp.model import Model
from cplex.callbacks import LazyConstraintCallback, SolveCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin

np.set_printoptions(precision=8, suppress=True)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)

AVAILABLE_BENCHMARKS = ['SP500', 'CMA', 'HML', 'MARKET', 'RMW', 'SMB']


def sort_benchmark(benchmark: pd.DataFrame) -> list:
    """Sort each benchmark column ascending; returns list of np.ndarray."""
    return [np.sort(benchmark[col].values) for col in benchmark.columns]


def compute_synthetic_cum(benchmarks_sorted: list) -> tuple:
    """
    Synthetic benchmark computations for TAF-MultiSSD.

    Returns
    -------
    synth_cum : (S,) ndarray — max_k( cumsum(sorted_b_k)[s] )
        Cumulative synthetic target used in SSD constraints (eq. 9).
    synth_abs : (S,) ndarray — max_k( sorted_b_k[s] )
        Absolute per-rank return: the distribution of the synthetic benchmark.
        Each element is the maximum return across all K benchmarks at rank s.
        Saved to CSV as the "absolute points" of the synthetic benchmark distribution.
    """
    stacked   = np.array(benchmarks_sorted)            # shape (K, S)
    synth_cum = np.max([np.cumsum(b) for b in benchmarks_sorted], axis=0)
    synth_abs = np.max(stacked, axis=0)                # pointwise max of sorted returns
    return synth_cum, synth_abs


def missing_percentage(df, threshold=0):
    missing_pct = df.isnull().mean() * 100
    result = pd.DataFrame({
        'missing_count':      df.isnull().sum(),
        'missing_percentage': missing_pct,
    })
    result = result[result['missing_count'] > 0]
    if result.empty:
        return 'Nenhuma coluna possui dados faltantes.'
    result = result[result['missing_percentage'] > threshold]
    return result.sort_values(by='missing_percentage', ascending=False)


def remove_empty_samples(df, threshold):
    return df[df.columns[df.isnull().mean() <= threshold]]


def fill_missing_data(df):
    return df.ffill().bfill()


# ── Lazy cut callback ──────────────────────────────────────────────────────────

class SSDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback, SolveCallback):
    """
    Cutting-plane separation for TAF-MultiSSD.

    Uses a single synthetic benchmark (the element-wise max across all K benchmarks),
    so the separation is identical to the single-benchmark scSSD callback.
    """

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.n_calls   = 0
        self.tolerance = 1e-6

    def __call__(self):

        if self.get_cplex_status() == self.status.optimal:
            self.n_calls += 1

            n_assets      = self.scenarios.shape[1]
            curr_solution = self.make_complete_solution()
            V_value       = curr_solution[self.model.get_var_by_name('V')]
            w_values      = [self.get_values(w.index) for w in self.w_vars]

            if self.logging_mode > 3:
                print(f"\n{'V':<5}: {V_value:.5f}")

            portfolio_returns  = self.scenarios @ w_values
            idx_returns        = list(enumerate(portfolio_returns))
            sorted_idx_returns = sorted(idx_returns, key=lambda x: x[1])
            sorted_p_returns   = sorted(portfolio_returns)
            cum_p_returns      = np.cumsum(sorted_p_returns)

            # synth_cum is already precomputed and sorted ascending
            Vs_values = [self.get_values(var.index) for var in self.Vs_vars]

            max_diff  = -np.inf
            max_index = None

            for s, ret in enumerate(cum_p_returns):
                tau_s    = self.synth_cum[s]
                lhs_expr = Vs_values[s]
                rhs_expr = ret - tau_s
                diff     = lhs_expr - rhs_expr

                if self.logging_mode > 3:
                    print(f"\t|Js|={s+1:2d} C:{lhs_expr:.5f} ({ret:.5f}) <= ({tau_s:.5f}) Viol:{diff:.5f}")

                if lhs_expr > rhs_expr + self.tolerance:
                    if diff > max_diff:
                        max_diff  = diff
                        max_index = s

            if max_index is not None:
                if self.logging_mode > 3:
                    print(f"\n[INFO]: SSD constraint violated at Vs_{max_index}, violation={max_diff}\n")

                Vs_var                 = self.model.get_var_by_name(f'Vs_{max_index}')
                worst_scenario_indices = [idx for idx, _ in sorted_idx_returns[:max_index + 1]]
                cumulative_scenario    = np.sum(self.scenarios[worst_scenario_indices, :], axis=0)
                scenario_return        = self.model.sum(
                    cumulative_scenario[i] * self.w_vars[i] for i in range(n_assets)
                )
                rhs = self.synth_cum[max_index]

                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(Vs_var <= scenario_return - rhs)
                self.add(cpx_lhs, sense, cpx_rhs)

                if self.logging_mode > 3:
                    print(f"\t\033[92m> Violated constraint added.\033[0m\n")

            if self.logging_mode > 3:
                print('-' * 40)


# ── Optimisation model ─────────────────────────────────────────────────────────

def solve_ssd(scenarios, benchmarks, logging_mode=0, callback=False):
    """
    TAF-MultiSSD solver.

    Key difference from scmSSD: instead of K separate SSD constraint sets, a single
    synthetic benchmark is built as the element-wise max of all benchmark cumulative
    sorted returns.  The portfolio must SSD-dominate this synthetic target.

    Parameters
    ----------
    scenarios  : (S, N) array-like — in-sample asset returns
    benchmarks : list of np.ndarray — one sorted return array per benchmark (K total)

    Returns
    -------
    V_value    : float
    obj        : float
    weights    : (N,) np.ndarray
    synth_cum  : (S,) np.ndarray — cumulative synthetic benchmark used in this solve
    """
    print('\n=== STARTING tafmSSD MODEL ===')
    scenarios  = np.array(scenarios)
    S, N       = scenarios.shape
    MODEL_NAME = 'tafmSSD'

    if isinstance(benchmarks, np.ndarray) and benchmarks.ndim == 1:
        benchmarks = [benchmarks]
    elif not isinstance(benchmarks, list):
        benchmarks = [np.array(benchmarks)]

    # ── Synthetic benchmark ───────────────────────────────────────────────────
    synth_cum, synth_abs = compute_synthetic_cum(benchmarks)   # shapes (S,)

    model = Model(name='tafmSSD')
    model.parameters.preprocessing.presolve         = 0
    model.parameters.mip.display                    = 4
    model.parameters.threads                        = 1
    model.parameters.emphasis.numerical             = 1
    model.parameters.randomseed                     = 0
    model.parameters.emphasis.mip                   = 2
    model.parameters.simplex.tolerances.optimality  = 1e-9
    model.parameters.simplex.tolerances.feasibility = 1e-9
    model.parameters.timelimit                      = 900
    model.parameters.mip.tolerances.absmipgap       = 0
    model.parameters.mip.tolerances.mipgap          = 0
    model.parameters.mip.tolerances.integrality     = 0

    w  = model.continuous_var_list(N, lb=0, ub=1, name='wl')
    Vs = model.continuous_var_list(S, name='Vs', lb=-model.infinity, ub=model.infinity)
    V  = model.continuous_var(name='V', lb=-model.infinity, ub=model.infinity)
    cb = model.binary_var(name='cb_temp')

    model.add_constraint(model.sum(w) == 1, ctname='budget')

    # ── SSD constraints against synthetic benchmark ───────────────────────────
    for t in range(S):
        model.add_constraint((t + 1) * V <= Vs[t], ctname=f'maxmin_{t}')
        soma_t          = np.sum(scenarios[:t + 1, :], axis=0)
        scenario_return = model.sum(soma_t[i] * w[i] for i in range(N))
        model.add_constraint(Vs[t] <= scenario_return - synth_cum[t], ctname=f'ssd_{t}')

    model.maximize(V + 0.0 * cb)

    if logging_mode == 3:
        for ct in model.iter_constraints():
            print(f'Name: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}')

    lp_path = Path(__file__).resolve().parent / 'models' / f'{MODEL_NAME}.lp'
    lp_path.parent.mkdir(parents=True, exist_ok=True)
    model.export_as_lp(str(lp_path))

    if logging_mode == 2:
        model.print_information()

    if callback:
        lazy_cb              = model.register_callback(SSDLazyCallback)
        lazy_cb.scenarios    = scenarios
        lazy_cb.synth_cum    = synth_cum
        lazy_cb.w_vars       = w
        lazy_cb.Vs_vars      = Vs
        lazy_cb.logging_mode = logging_mode

    print('\nIniciando processo de otimização...')
    sol = model.solve(log_output=(logging_mode > 1))

    if sol is None:
        raise RuntimeError('No solution found for the optimization problem!')

    weights = np.array([sol[wi] for wi in w])
    print('\n[INFO]: Solução encontrada:')
    print(f'\tV              = {sol[V]}')
    print(f'\tBest Solution  = {sol.get_objective_value()}')
    print(f'\tNº of Cuts    = {model.get_cuts()["user"]}')

    return sol[V], sol.get_objective_value(), weights, synth_cum, synth_abs


# ── Walk-forward validation ────────────────────────────────────────────────────

def WFV(prices, returns, benchmark, logging_mode=0, sliding=True, output_dir=None):
    """
    Walk-forward validation for TAF-MultiSSD.

    At each rebalancing step the synthetic benchmark is computed from the training
    window and saved to CSV.  After the loop the OOS synthetic benchmark series
    (max of actual benchmark returns at each OOS time step) is returned.

    Returns
    -------
    norm_portfolio_df  : DataFrame  — normalised portfolio value (column 'TAF-SSD')
    weights_invested   : DataFrame
    rebalances_df      : DataFrame
    valuation_df       : DataFrame
    shares_df          : DataFrame
    optInfoData        : DataFrame
    synth_bench_oos    : pd.Series  — OOS normalised synthetic benchmark
    """
    INITIAL_CAPITAL   = 1_000_000

    weights_invested  = pd.DataFrame()
    rebalances_df     = pd.DataFrame()
    optInfoData       = pd.DataFrame()
    valuation_df      = pd.DataFrame()
    shares_df         = pd.DataFrame()
    norm_portfolio_df = pd.DataFrame(columns=['TAF-SSD'])

    print('---------------------------------------------------------------')
    print('[INFO]: Simulation Started...\n')

    initial_train_days = 200
    rebalance_freq     = 21
    max_steps          = len(prices)
    last_share         = None

    for step in range(0, max_steps):

        train_start = step if sliding else 0
        train_end   = train_start + initial_train_days

        train       = returns.iloc[train_start:train_end]
        train_bench = benchmark.iloc[train_start:train_end]
        valid       = returns.iloc[train_end - 1:train_end]

        train_bench_sorted = sort_benchmark(train_bench)

        v_value, func_obj, weights_arr, synth_cum, synth_abs = solve_ssd(
            train, train_bench_sorted, logging_mode=logging_mode, callback=True
        )

        v_df        = pd.DataFrame([v_value], columns=['obj'], index=valid.index)
        weights_opt = pd.DataFrame([weights_arr], columns=valid.columns, index=valid.index)

        print(f'Day: {step + 1} ({valid.index.min()})')
        print(f'\tSum proportions: {np.sum(weights_arr) * 100:.4f}%')
        print(f'\tFUNÇÃO OBJETIVO: {func_obj}')
        print(f'\tSynthetic benchmark at last rank: {synth_cum[-1]:.6f}')

        price = np.array(prices.loc[prices.index[step], :])

        if step == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            valuation     = weights_arr * INITIAL_CAPITAL
            share         = valuation / price
            rebalances_df = pd.concat([rebalances_df, weights_opt])
            optInfoData   = pd.concat([optInfoData, v_df])

            if output_dir is not None:
                date_str = valid.index.min().strftime('%Y-%m-%d')
                train.to_csv(output_dir / f'insample-tafmssd-{date_str}.csv', index=False)
                _save_synthetic_bench(synth_cum, synth_abs, train_bench.columns.tolist(),
                                      output_dir / f'synthetic_bench-{date_str}.csv')

        elif step % rebalance_freq == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            date_str = valid.index.min().strftime('%Y-%m-%d')

            if output_dir is not None:
                train.to_csv(output_dir / f'insample-tafmssd-{date_str}.csv', index=False)
                _save_synthetic_bench(synth_cum, synth_abs, train_bench.columns.tolist(),
                                      output_dir / f'synthetic_bench-{date_str}.csv')

            optInfoData      = pd.concat([optInfoData, v_df])
            valuation_normal = last_share * price
            total_value      = np.sum(valuation_normal)
            print(f'\tValuation before rebalance: {total_value}')
            valuation     = weights_arr * total_value
            share         = valuation / price
            rebalances_df = pd.concat([rebalances_df, weights_opt])

        else:
            share     = last_share
            valuation = share * price

        date_idx = valid.index[0]

        valuation_df     = pd.concat([valuation_df,
                                       pd.DataFrame([valuation], index=[date_idx], columns=valid.columns)])
        shares_df        = pd.concat([shares_df,
                                       pd.DataFrame([share], index=[date_idx], columns=valid.columns)])
        weights_invested = pd.concat([weights_invested, weights_opt])

        norm_portfolio_df.loc[date_idx, 'TAF-SSD'] = np.sum(valuation) / INITIAL_CAPITAL
        last_share = share

        print(f'\tPortfolio Value: {np.sum(valuation):.2f}')
        print(f'\tNorm Portfolio:  {np.sum(valuation) / INITIAL_CAPITAL:.6f}')
        print('---------------------------------------------------------------\n')

    # ── OOS synthetic benchmark: pointwise max of normalized individual benchmarks ──
    # At each day t: synth[t] = max_k( prod_{u<=t} (1+r_k[u]) )
    # = the best-performing benchmark at that point in time (all starting at 1.0)
    # This is "normalizado pelo seu respectivo benchmark no respectivo cenário":
    # each point is the value of whichever benchmark is dominant on that day.
    oos_bench       = benchmark.reindex(prices.index)
    oos_bench_norm  = (1 + oos_bench).cumprod()
    oos_bench_norm  = oos_bench_norm.div(oos_bench_norm.iloc[0])  # all start at 1.0
    synth_bench_oos = oos_bench_norm.max(axis=1)                  # best benchmark each day

    return (norm_portfolio_df, weights_invested, rebalances_df,
            valuation_df, shares_df, optInfoData, synth_bench_oos)


def _save_synthetic_bench(synth_cum: np.ndarray, synth_abs: np.ndarray,
                          bench_names: list, path: Path):
    """Save per-window synthetic benchmark details to CSV.

    synth_abs[s] = max_k(sorted_b_k[s])       — absolute distribution point at rank s
    synth_cum[s] = max_k(cumsum(sorted_b_k)[s]) — cumulative, used in SSD constraints
    """
    pd.DataFrame({
        'rank':       np.arange(1, len(synth_cum) + 1),
        'synth_abs':  synth_abs,   # absolute per-rank max return (distribution point)
        'synth_cum':  synth_cum,   # cumulative synthetic target
        'benchmarks': ', '.join(bench_names),
    }).to_csv(path, index=False)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='tafmSSD — TAF Multi-benchmark SSD (synthetic benchmark via max)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmarks: {', '.join(AVAILABLE_BENCHMARKS)}

The synthetic benchmark at each scenario rank s is:
    tau_synth[s] = max_k( cumsum(sorted_returns_k)[s] )

Examples:
  python tafmSSD.py --benchmarks all
  python tafmSSD.py --benchmarks SP500 HML CMA
  python tafmSSD.py --benchmarks MARKET RMW --oos-start 2024-01-01 --oos-end 2024-06-01
        """,
    )
    parser.add_argument('--benchmarks', nargs='+', default=['all'], metavar='BENCHMARK')
    parser.add_argument('--oos-start',    default='2024-01-01')
    parser.add_argument('--oos-end',      default='2024-06-01')
    parser.add_argument('--n-cols',       type=int, default=600)
    parser.add_argument('--logging-mode', type=int, default=1, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--no-sliding',   action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    args = parse_args()

    # ── Resolve benchmarks ────────────────────────────────────────────────────
    if any(b.lower() == 'all' for b in args.benchmarks):
        selected_benchmarks = AVAILABLE_BENCHMARKS[:]
    else:
        selected_benchmarks = []
        for b in args.benchmarks:
            b_upper = b.upper()
            if b_upper not in AVAILABLE_BENCHMARKS:
                print(f'[ERROR]: Unknown benchmark "{b}". Available: {AVAILABLE_BENCHMARKS}')
                sys.exit(1)
            selected_benchmarks.append(b_upper)

    print(f'[INFO]: Algorithm          : tafmSSD (TAF Multi-benchmark SSD)')
    print(f'[INFO]: Selected benchmarks: {selected_benchmarks}')
    print(f'[INFO]: OOS period         : {args.oos_start} → {args.oos_end}')
    print(f'[INFO]: Asset columns      : {args.n_cols}')
    print(f'[INFO]: Window             : {"sliding" if not args.no_sliding else "expanding"}')

    SRC_DIR   = Path(__file__).resolve().parent
    DATA_DIR  = SRC_DIR / 'data'
    OOS_START = args.oos_start
    OOS_END   = args.oos_end
    N_COLS    = args.n_cols

    # ── Load market data ──────────────────────────────────────────────────────
    marketData = pd.read_csv(DATA_DIR / 'marketData.csv', parse_dates=['Date'])
    marketData.set_index('Date', inplace=True)

    cleanData        = remove_empty_samples(marketData, 0.15)
    cleanData        = fill_missing_data(cleanData)
    marketDataPrices = cleanData.copy().abs()

    print(f'[INFO]: Clean data shape   : {marketDataPrices.shape}')

    # ── Returns + inject Fama-French factors ──────────────────────────────────
    marketDataReturns = marketDataPrices.diff() / marketDataPrices.shift(1)

    factors = pd.read_csv(DATA_DIR / 'factors.csv')
    marketDataReturns[factors.columns] = factors.to_numpy()
    marketDataReturns = marketDataReturns.iloc[1:]

    # ── Split assets and benchmarks ───────────────────────────────────────────
    ALL_BENCH_COLS = ['SP500'] + factors.columns.tolist()

    benchReturns = marketDataReturns[selected_benchmarks]
    assetReturns = marketDataReturns.drop(columns=ALL_BENCH_COLS, errors='ignore').iloc[:, :N_COLS]

    # ── OOS asset prices ──────────────────────────────────────────────────────
    oosPrices = marketDataPrices.loc[OOS_START:OOS_END]
    oosAssets = oosPrices.drop(columns=['SP500']).iloc[:, :N_COLS]

    print(f'[INFO]: Asset returns      : {assetReturns.shape}')
    print(f'[INFO]: Benchmark returns  : {benchReturns.shape}  {list(benchReturns.columns)}')
    print(f'[INFO]: OOS prices         : {oosAssets.shape}')

    output_dir = SRC_DIR / 'output' / 'tafmssd'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run walk-forward validation ───────────────────────────────────────────
    (portfolio, weightsInvested, weightsRebalance,
     portfolioValuation, portfolioShares, optInfoData,
     synth_bench_oos) = WFV(
        oosAssets,
        assetReturns,
        benchReturns,
        logging_mode=args.logging_mode,
        sliding=not args.no_sliding,
        output_dir=output_dir,
    )

    print(optInfoData)

    # ── Label and save portfolio ──────────────────────────────────────────────
    bench_label = '+'.join(selected_benchmarks)
    port_label  = f'TAF-SSD {bench_label}'
    portfolio.rename(columns={'TAF-SSD': port_label}, inplace=True)

    portfolio.to_csv(output_dir / 'portfolio_tafmssd.csv')
    print(f'[INFO]: Portfolio saved → {output_dir / "portfolio_tafmssd.csv"}')

    # ── Add benchmark series for plotting ─────────────────────────────────────
    for bench in selected_benchmarks:
        if bench == 'SP500':
            series = marketDataPrices.loc[OOS_START:OOS_END, 'SP500'].abs()
            portfolio['SP500'] = (series / series.iloc[0]).reindex(portfolio.index)
        elif bench in factors.columns:
            bench_ret = marketDataReturns.loc[OOS_START:OOS_END, bench]
            bench_cum = (1 + bench_ret).cumprod()
            portfolio[bench] = (bench_cum / bench_cum.iloc[0]).reindex(portfolio.index)

    # ── Add synthetic benchmark (OOS: max of actual returns) ──────────────────
    portfolio['Synthetic'] = synth_bench_oos.reindex(portfolio.index)
    portfolio['Risk free rate'] = 1.0

    # Save synthetic benchmark separately for later analysis
    synth_bench_oos.to_csv(output_dir / 'synthetic_benchmark_oos.csv', header=['Synthetic'])
    print(f'[INFO]: OOS synthetic benchmark saved → {output_dir / "synthetic_benchmark_oos.csv"}')

    # ── In-sample SSD dominance vs synthetic benchmark ────────────────────────
    last_w      = weightsRebalance.iloc[-1].to_numpy()
    last_idx    = len(assetReturns) - len(oosAssets) + len(weightsRebalance) - 1
    train_start = last_idx
    train_end   = train_start + 200
    train_ret   = assetReturns.iloc[train_start:train_end].to_numpy()
    train_bench = benchReturns.iloc[train_start:train_end]
    synth_cum_last, synth_abs_last = compute_synthetic_cum(sort_benchmark(train_bench))
    r_port      = train_ret @ last_w

    s_dom, fsd_dom, ssd_dom = compute_ssd_dominance(np.sort(r_port), synth_abs_last)
    min_ssd = ssd_dom.min()
    print(f'\n[INFO]: In-sample SSD dominance vs synthetic benchmark — '
          f'min(SSD) = {min_ssd:.6f}  '
          f'({"dominates" if min_ssd >= 0 else "does NOT dominate"})')

    # ── Plot ──────────────────────────────────────────────────────────────────
    benchmark_cols = selected_benchmarks + ['Synthetic']
    plot_portfolio_comparison(
        portfolio,
        force_background=True,
        benchmark_cols=benchmark_cols,
        title=f'tafmSSD — {port_label} vs benchmarks + sintético ({OOS_START} → {OOS_END})',
        save_path=output_dir / 'performance_comparison.png',
    )

    print('\n==========================================================')
    print('Finalising simulation...')
