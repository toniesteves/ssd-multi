
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

SRC_DIR     = Path(__file__).resolve().parent
DATA_DIR    = SRC_DIR / 'data'
SECTORS_DIR = DATA_DIR / 'sectors'

AVAILABLE_BENCHMARKS = sorted([f.stem for f in SECTORS_DIR.glob('*.csv')])


def load_sector_returns(sectors_dir: Path, selected: list, index: pd.Index) -> pd.DataFrame:
    """
    Load sector CSVs and compute equal-weighted daily returns for each sector.

    Each sector CSV contains adjusted-close prices for all stocks in that sector.
    The benchmark return on day t = mean(daily returns of all stocks in that sector).
    Missing values are forward-filled before computing returns; stocks with > 15% missing
    are dropped from the sector universe.
    """
    frames = {}
    for sector in selected:
        prices = pd.read_csv(sectors_dir / f'{sector}.csv', parse_dates=['Date'])
        prices.set_index('Date', inplace=True)
        prices = prices[prices.columns[prices.isnull().mean() <= 0.15]]
        prices = prices.ffill().bfill().abs()
        ret    = prices.diff() / prices.shift(1)
        ret    = ret.iloc[1:]
        frames[sector] = ret.mean(axis=1)
    df = pd.DataFrame(frames)
    return df.reindex(index)


def sort_benchmark(benchmark: pd.DataFrame) -> list:
    return [np.sort(benchmark[col].values) for col in benchmark.columns]


def compute_synthetic_tau(benchmarks_sorted: list) -> tuple:
    """
    Synthetic benchmark tail profile (taf-MultiSSDapproach.pdf, Sec. 2.1).

    tau_k[s]  = Tail_{(s+1)/S_k}(I^k) = (1/S_k) * cumsum(sorted_b_k)[s]
                — unconditional expectation of the s+1 worst benchmark outcomes
    synth_tau : max_k( tau_k[s] )     — used in the SSD constraints
    synth_cum : S * synth_tau         — cumulative envelope (CSV compatibility)
    synth_abs : diff of synth_cum     — implied per-rank outcome of the synthetic benchmark
    """
    tails     = [np.cumsum(b) / len(b) for b in benchmarks_sorted]
    synth_tau = np.max(tails, axis=0)
    S         = len(benchmarks_sorted[0])
    synth_cum = synth_tau * S
    synth_abs = np.diff(synth_cum, prepend=0.0)
    return synth_tau, synth_cum, synth_abs


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
    Cutting-plane separation for TAF-MultiSSD (sector benchmarks).

    Identical to tafmSSD — single synthetic benchmark via element-wise max.
    """

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.n_calls   = 0
        self.n_cuts    = 0
        self.tolerance = 1e-6

    def __call__(self):

        if self.get_cplex_status() == self.status.optimal:
            self.n_calls += 1

            n_scen, n_assets = self.scenarios.shape
            curr_solution = self.make_complete_solution()
            V_value       = curr_solution[self.model.get_var_by_name('V')]
            w_values      = [self.get_values(w.index) for w in self.w_vars]

            if self.logging_mode > 3:
                print(f"\n{'V':<5}: {V_value:.5f}")

            portfolio_returns  = self.scenarios @ w_values
            idx_returns        = list(enumerate(portfolio_returns))
            sorted_idx_returns = sorted(idx_returns, key=lambda x: x[1])
            sorted_p_returns   = sorted(portfolio_returns)
            tail_p_returns     = np.cumsum(sorted_p_returns) / n_scen

            Vs_values = [self.get_values(var.index) for var in self.Vs_vars]

            max_diff  = -np.inf
            max_index = None

            for s, ret in enumerate(tail_p_returns):
                tau_s    = self.synth_tau[s]
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
                cumulative_scenario    = np.sum(self.scenarios[worst_scenario_indices, :], axis=0) / n_scen
                scenario_return        = self.model.sum(
                    cumulative_scenario[i] * self.w_vars[i] for i in range(n_assets)
                )
                rhs = self.synth_tau[max_index]

                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(Vs_var <= scenario_return - rhs)
                self.add(cpx_lhs, sense, cpx_rhs)
                self.n_cuts += 1

                if self.logging_mode > 3:
                    print(f"\t\033[92m> Violated constraint added.\033[0m\n")

            if self.logging_mode > 3:
                print('-' * 40)


# ── Optimisation model ─────────────────────────────────────────────────────────

def solve_ssd(scenarios, benchmarks, logging_mode=0, callback=False, unscaled=False):
    """
    TAF-MultiSSD solver against sector synthetic benchmark.

    Parameters
    ----------
    scenarios  : (S, N) array-like — in-sample asset returns
    benchmarks : list of np.ndarray — one sorted return array per sector (K total)
    unscaled   : max-min constraint V <= Vs[t] (Tail objective) instead of
                 ((t+1)/S)·V <= Vs[t] (CVaR objective)

    Returns
    -------
    V_value     : float
    obj         : float
    weights     : (N,) np.ndarray
    synth_tau   : (S,) np.ndarray — max_k Tail_{s/S_k}(I^k), used in constraints
    synth_cum   : (S,) np.ndarray
    synth_abs   : (S,) np.ndarray
    solver_info : dict — solver diagnostics (gap, bound, cuts, time, ...)
    """
    print('\n=== STARTING tafmSSD-sectors MODEL ===')
    scenarios  = np.array(scenarios)
    S, N       = scenarios.shape
    MODEL_NAME = 'tafmssd-sectors-unscaled' if unscaled else 'tafmssd-sectors'

    if isinstance(benchmarks, np.ndarray) and benchmarks.ndim == 1:
        benchmarks = [benchmarks]
    elif not isinstance(benchmarks, list):
        benchmarks = [np.array(benchmarks)]

    synth_tau, synth_cum, synth_abs = compute_synthetic_tau(benchmarks)

    model = Model(name='tafmssd-sectors')
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

    # Scaled model (Fábián et al., 2011b): (s/S)·V <= Tail_{s/S}(Rx) - tau_s,
    # with Tail_{s/S}(Rx) = (1/S) * sum of the s worst portfolio returns.
    # Unscaled model: V <= Tail_{s/S}(Rx) - tau_s (same Tail scale, no s/S factor).
    for t in range(S):
        coef = 1.0 if unscaled else (t + 1) / S
        model.add_constraint(coef * V <= Vs[t], ctname=f'maxmin_{t}')
        soma_t          = np.sum(scenarios[:t + 1, :], axis=0) / S
        scenario_return = model.sum(soma_t[i] * w[i] for i in range(N))
        model.add_constraint(Vs[t] <= scenario_return - synth_tau[t], ctname=f'ssd_{t}')

    model.maximize(V + 0.0 * cb)

    if logging_mode == 3:
        for ct in model.iter_constraints():
            print(f'Name: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}')

    lp_path = Path(__file__).resolve().parent / 'models' / f'{MODEL_NAME}.lp'
    lp_path.parent.mkdir(parents=True, exist_ok=True)
    model.export_as_lp(str(lp_path))

    if logging_mode == 2:
        model.print_information()

    lazy_cb = None
    if callback:
        lazy_cb              = model.register_callback(SSDLazyCallback)
        lazy_cb.scenarios    = scenarios
        lazy_cb.synth_tau    = synth_tau
        lazy_cb.w_vars       = w
        lazy_cb.Vs_vars      = Vs
        lazy_cb.logging_mode = logging_mode

    print('\nIniciando processo de otimização...')
    sol = model.solve(log_output=(logging_mode > 1))

    if sol is None:
        raise RuntimeError('No solution found for the optimization problem!')

    weights = np.array([sol[wi] for wi in w])

    details = sol.solve_details
    try:
        cplex_cuts = model.get_cuts()
    except Exception:
        cplex_cuts = {}

    solver_info = {
        'status':        details.status,
        'status_code':   details.status_code,
        'objective':     sol.get_objective_value(),
        'V':             sol[V],
        'best_bound':    details.best_bound,
        'mip_gap':       details.mip_relative_gap,
        'solve_time_s':  details.time,
        'n_iterations':  details.nb_iterations,
        'n_nodes':       details.nb_nodes_processed,
        'n_lazy_calls':  lazy_cb.n_calls if lazy_cb is not None else 0,
        'n_lazy_cuts':   lazy_cb.n_cuts if lazy_cb is not None else 0,
        'cplex_cuts':    sum(cplex_cuts.values()) if cplex_cuts else 0,
        'n_variables':   model.number_of_variables,
        'n_constraints': model.number_of_constraints,
    }

    print('\n[INFO]: Solução encontrada:')
    print(f'\tV              = {sol[V]}')
    print(f'\tBest Solution  = {sol.get_objective_value()}')
    print(f'\tBest Bound     = {details.best_bound}')
    print(f'\tMIP Gap        = {details.mip_relative_gap}')
    print(f'\tStatus         = {details.status}')
    print(f'\tSolve time     = {details.time:.2f}s')
    print(f'\tLazy cuts      = {solver_info["n_lazy_cuts"]} (in {solver_info["n_lazy_calls"]} callback calls)')

    return (sol[V], sol.get_objective_value(), weights,
            synth_tau, synth_cum, synth_abs, solver_info)


# ── Walk-forward validation ────────────────────────────────────────────────────

def WFV(prices, returns, benchmark, logging_mode=0, sliding=True, output_dir=None,
        unscaled=False):
    """
    Walk-forward validation for tafmSSD-sectors.

    Parameters
    ----------
    prices    : OOS asset prices (N assets)
    returns   : full asset returns (train + OOS)
    benchmark : sector benchmark returns aligned to returns.index
    """
    INITIAL_CAPITAL   = 1_000_000

    weights_invested  = pd.DataFrame()
    rebalances_df     = pd.DataFrame()
    optInfoData       = pd.DataFrame()
    valuation_df      = pd.DataFrame()
    shares_df         = pd.DataFrame()
    solver_stats_df   = pd.DataFrame()
    norm_portfolio_df = pd.DataFrame(columns=['TAF-SSD-sectors'])

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

        (v_value, func_obj, weights_arr, synth_tau, synth_cum, synth_abs,
         solver_info) = solve_ssd(
            train, train_bench_sorted, logging_mode=logging_mode, callback=True,
            unscaled=unscaled,
        )

        v_df        = pd.DataFrame([v_value], columns=['obj'], index=valid.index)
        weights_opt = pd.DataFrame([weights_arr], columns=valid.columns, index=valid.index)

        solver_stats_df = pd.concat([solver_stats_df,
                                     pd.DataFrame([solver_info], index=valid.index)])

        print(f'Day: {step + 1} ({valid.index.min()})')
        print(f'\tSum proportions: {np.sum(weights_arr) * 100:.4f}%')
        print(f'\tFUNÇÃO OBJETIVO: {func_obj}')
        print(f'\tSynthetic tau at last rank: {synth_tau[-1]:.6f}')

        price = np.array(prices.loc[prices.index[step], :])

        if step == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            valuation     = weights_arr * INITIAL_CAPITAL
            share         = valuation / price
            rebalances_df = pd.concat([rebalances_df, weights_opt])
            optInfoData   = pd.concat([optInfoData, v_df])

            if output_dir is not None:
                date_str = valid.index.min().strftime('%Y-%m-%d')
                train.to_csv(output_dir / f'insample-tafmssd-sectors-{date_str}.csv', index=False)
                _save_synthetic_bench(synth_tau, synth_cum, synth_abs, train_bench.columns.tolist(),
                                      output_dir / f'synthetic_bench-{date_str}.csv')

        elif step % rebalance_freq == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            date_str = valid.index.min().strftime('%Y-%m-%d')

            if output_dir is not None:
                train.to_csv(output_dir / f'insample-tafmssd-sectors-{date_str}.csv', index=False)
                _save_synthetic_bench(synth_tau, synth_cum, synth_abs, train_bench.columns.tolist(),
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

        norm_portfolio_df.loc[date_idx, 'TAF-SSD-sectors'] = np.sum(valuation) / INITIAL_CAPITAL
        last_share = share

        print(f'\tPortfolio Value: {np.sum(valuation):.2f}')
        print(f'\tNorm Portfolio:  {np.sum(valuation) / INITIAL_CAPITAL:.6f}')
        print('---------------------------------------------------------------\n')

    # OOS synthetic benchmark — same formulation as in-sample:
    #   tau_oos[s] = max_k( Tail_{s/S_k}(I^k) ) over the OOS sector returns.
    # The implied sorted returns diff(S·tau) are mapped back to dates by quantile
    # matching: rank s is assigned to the date holding the s-th smallest
    # cross-sector mean return, so the series has exactly the synthetic
    # distribution with a market-consistent chronology.
    oos_bench = benchmark.reindex(prices.index)
    _, _, synth_abs_oos = compute_synthetic_tau(sort_benchmark(oos_bench))
    rank_of_date  = oos_bench.mean(axis=1).rank(method='first').astype(int) - 1
    synth_ret_oos = pd.Series(synth_abs_oos[rank_of_date.values], index=oos_bench.index)
    synth_norm      = (1 + synth_ret_oos).cumprod()
    synth_bench_oos = synth_norm.div(synth_norm.iloc[0])

    return (norm_portfolio_df, weights_invested, rebalances_df,
            valuation_df, shares_df, optInfoData, synth_bench_oos, solver_stats_df)


def _save_synthetic_bench(synth_tau: np.ndarray, synth_cum: np.ndarray,
                          synth_abs: np.ndarray, bench_names: list, path: Path):
    pd.DataFrame({
        'rank':       np.arange(1, len(synth_tau) + 1),
        'synth_abs':  synth_abs,
        'synth_tau':  synth_tau,
        'synth_cum':  synth_cum,
        'benchmarks': ', '.join(bench_names),
    }).to_csv(path, index=False)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='tafmSSD-sectors — TAF Multi-benchmark SSD com benchmarks setoriais',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmarks (GICS sectors): {', '.join(AVAILABLE_BENCHMARKS)}

The synthetic benchmark at each scenario rank s is the tail expectation envelope:
    tau_synth[s] = max_k( Tail_{{s/S_k}}(I^k) ) = max_k( (1/S_k) * cumsum(sorted_sector_returns_k)[s] )

Each sector return is the equal-weighted mean of all stocks in that sector.

Examples:
  python tafmSSD-sectors.py --benchmarks all
  python tafmSSD-sectors.py --benchmarks TECHNOLOGY FINANCIALS HEALTHCARE
  python tafmSSD-sectors.py --benchmarks ENERGY UTILITIES --oos-start 2024-01-01
        """,
    )
    parser.add_argument('--benchmarks', nargs='+', default=['all'], metavar='BENCHMARK')
    parser.add_argument('--oos-start',    default='2024-01-01')
    parser.add_argument('--oos-end',      default='2024-06-01')
    parser.add_argument('--n-cols',       type=int, default=600)
    parser.add_argument('--logging-mode', type=int, default=1, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--no-sliding',   action='store_true')
    parser.add_argument('--unscaled',     action='store_true',
                        help='Unscaled (Tail) max-min: V <= Vs[t]. Outputs go to output/tafmssd-sectors-unscaled/')
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

    print(f'[INFO]: Algorithm          : tafmSSD-sectors (TAF Multi-benchmark SSD — setor GICS)')
    print(f'[INFO]: Max-min scaling    : {"unscaled (Tail)" if args.unscaled else "scaled (CVaR)"}')
    print(f'[INFO]: Selected benchmarks: {selected_benchmarks}')
    print(f'[INFO]: OOS period         : {args.oos_start} → {args.oos_end}')
    print(f'[INFO]: Asset columns      : {args.n_cols}')
    print(f'[INFO]: Window             : {"sliding" if not args.no_sliding else "expanding"}')

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

    # ── Asset returns (no factor injection) ───────────────────────────────────
    marketDataReturns = marketDataPrices.diff() / marketDataPrices.shift(1)
    marketDataReturns = marketDataReturns.iloc[1:]

    ALL_BENCH_COLS = ['SP500']
    assetReturns   = marketDataReturns.drop(columns=ALL_BENCH_COLS, errors='ignore').iloc[:, :N_COLS]

    # ── Sector benchmark returns ──────────────────────────────────────────────
    benchReturns = load_sector_returns(SECTORS_DIR, selected_benchmarks, marketDataReturns.index)

    # ── OOS asset prices ──────────────────────────────────────────────────────
    oosPrices = marketDataPrices.loc[OOS_START:OOS_END]
    oosAssets = oosPrices.drop(columns=ALL_BENCH_COLS, errors='ignore').iloc[:, :N_COLS]

    print(f'[INFO]: Asset returns      : {assetReturns.shape}')
    print(f'[INFO]: Benchmark returns  : {benchReturns.shape}  {list(benchReturns.columns)}')
    print(f'[INFO]: OOS prices         : {oosAssets.shape}')

    output_dir = SRC_DIR / 'output' / ('tafmssd-sectors-unscaled' if args.unscaled else 'tafmssd-sectors')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run walk-forward validation ───────────────────────────────────────────
    (portfolio, weightsInvested, weightsRebalance,
     portfolioValuation, portfolioShares, optInfoData,
     synth_bench_oos, solverStats) = WFV(
        oosAssets,
        assetReturns,
        benchReturns,
        logging_mode=args.logging_mode,
        sliding=not args.no_sliding,
        output_dir=output_dir,
        unscaled=args.unscaled,
    )

    print(optInfoData)

    # ── Save solver diagnostics ───────────────────────────────────────────────
    solverStats.index.name = 'date'
    solverStats.to_csv(output_dir / 'solver_stats_tafmssd_sectors.csv')
    print(f'[INFO]: Solver stats saved → {output_dir / "solver_stats_tafmssd_sectors.csv"}')
    print(solverStats[['status', 'objective', 'best_bound', 'mip_gap',
                       'n_lazy_cuts', 'solve_time_s']])

    # ── Label and save portfolio ──────────────────────────────────────────────
    bench_label = '+'.join(selected_benchmarks)
    port_label  = f'TAF-SSD-sectors{"-unscaled" if args.unscaled else ""} {bench_label}'
    portfolio.rename(columns={'TAF-SSD-sectors': port_label}, inplace=True)

    portfolio.to_csv(output_dir / 'portfolio_tafmssd_sectors.csv')
    print(f'[INFO]: Portfolio saved → {output_dir / "portfolio_tafmssd_sectors.csv"}')

    # ── Add sector benchmark series for plotting ──────────────────────────────
    for sector in selected_benchmarks:
        bench_ret = benchReturns.loc[OOS_START:OOS_END, sector]
        bench_cum = (1 + bench_ret).cumprod()
        portfolio[sector] = (bench_cum / bench_cum.iloc[0]).reindex(portfolio.index)

    # ── Add synthetic benchmark ───────────────────────────────────────────────
    portfolio['Synthetic'] = synth_bench_oos.reindex(portfolio.index)
    portfolio['Risk free rate'] = 1.0

    synth_bench_oos.to_csv(output_dir / 'synthetic_benchmark_oos.csv', header=['Synthetic'])
    print(f'[INFO]: OOS synthetic benchmark saved → {output_dir / "synthetic_benchmark_oos.csv"}')

    # ── In-sample SSD dominance vs synthetic benchmark ────────────────────────
    last_w      = weightsRebalance.iloc[-1].to_numpy()
    last_idx    = len(assetReturns) - len(oosAssets) + len(weightsRebalance) - 1
    train_start = last_idx
    train_end   = train_start + 200
    train_ret   = assetReturns.iloc[train_start:train_end].to_numpy()
    train_bench = benchReturns.iloc[train_start:train_end]
    _, _, synth_abs_last = compute_synthetic_tau(sort_benchmark(train_bench))
    r_port = train_ret @ last_w

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
        title=f'tafmSSD-sectors — {port_label} vs benchmarks setoriais + sintético ({OOS_START} → {OOS_END})',
        save_path=output_dir / 'performance_comparison.pdf',
    )

    print('\n==========================================================')
    print('Finalising simulation...')
