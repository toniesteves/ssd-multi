
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

    Same routine as tafmSSD-sectors.py so both models see identical benchmarks.
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


def remove_empty_samples(df, threshold):
    return df[df.columns[df.isnull().mean() <= threshold]]


def fill_missing_data(df):
    return df.ffill().bfill()


# ── Lazy cut callback ──────────────────────────────────────────────────────────

class SSDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback, SolveCallback):
    """
    Cutting-plane separation for multi-benchmark scaled SSD (sector benchmarks).

    One separation loop per benchmark k: the most violated constraint of each
    benchmark is added at every callback call (same scheme as scmSSD).
    Tail scale (/S) as in tafmSSD-sectors, so both models share tolerances.
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

            for k in range(len(self.tau_all)):
                tau_k       = self.tau_all[k]
                Vs_values_k = [self.get_values(var.index) for var in self.Vs_all[k]]

                max_diff  = -np.inf
                max_index = None

                for s, ret in enumerate(tail_p_returns):
                    tau_s    = tau_k[s]
                    lhs_expr = Vs_values_k[s]
                    rhs_expr = ret - tau_s
                    diff     = lhs_expr - rhs_expr

                    if self.logging_mode > 3:
                        print(f"\t[Bench {k}] |Js|={s+1:2d} C:{lhs_expr:.5f} ({ret:.5f}) <= ({tau_s:.5f}) Viol:{diff:.5f}")

                    if lhs_expr > rhs_expr + self.tolerance:
                        if diff > max_diff:
                            max_diff  = diff
                            max_index = s

                if max_index is not None:
                    if self.logging_mode > 3:
                        print(f"\n[INFO]: SSD constraint violated at Vs_{k}_{max_index}, violation={max_diff}\n")

                    Vs_var                 = self.model.get_var_by_name(f'Vs_{k}_{max_index}')
                    worst_scenario_indices = [idx for idx, _ in sorted_idx_returns[:max_index + 1]]
                    cumulative_scenario    = np.sum(self.scenarios[worst_scenario_indices, :], axis=0) / n_scen
                    scenario_return        = self.model.sum(
                        cumulative_scenario[i] * self.w_vars[i] for i in range(n_assets)
                    )
                    rhs = tau_k[max_index]

                    cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(Vs_var <= scenario_return - rhs)
                    self.add(cpx_lhs, sense, cpx_rhs)
                    self.n_cuts += 1

                    if self.logging_mode > 3:
                        print(f"\t\033[92m> Violated constraint added.\033[0m\n")

            if self.logging_mode > 3:
                print('-' * 40)


# ── Optimisation model ─────────────────────────────────────────────────────────

def solve_ssd(scenarios, benchmarks, logging_mode=0, callback=False):
    """
    Multi-benchmark scaled SSD (original formulation, all benchmarks jointly).

    For every benchmark k and rank t:
        ((t+1)/S) · V <= Vs_{k,t}
        Vs_{k,t}      <= Tail_{(t+1)/S}(R_port) - tau_k[t]      [lazy]
    with tau_k[t] = (1/S) * cumsum(sorted_b_k)[t].

    This is the formulation that tafmSSD-sectors.py collapses into a single
    synthetic benchmark via max_k(tau_k); both must yield the same optimal V.

    Parameters
    ----------
    scenarios  : (S, N) array-like — in-sample asset returns
    benchmarks : list of np.ndarray — one sorted return array per sector (K total)

    Returns
    -------
    V_value, obj, weights (N,), solver_info (dict)
    """
    print('\n=== STARTING tafscmSSD-sectors MODEL ===')
    scenarios  = np.array(scenarios)
    S, N       = scenarios.shape
    MODEL_NAME = 'tafscmssd-sectors'

    if isinstance(benchmarks, np.ndarray) and benchmarks.ndim == 1:
        benchmarks = [benchmarks]
    elif not isinstance(benchmarks, list):
        benchmarks = [np.array(benchmarks)]

    n_benchmarks = len(benchmarks)
    tau_all      = [np.cumsum(np.sort(b)) / len(b) for b in benchmarks]

    model = Model(name='tafscmssd-sectors')
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

    w      = model.continuous_var_list(N, lb=0, ub=1, name='wl')
    Vs_all = {k: model.continuous_var_list(S, name=f'Vs_{k}',
                                           lb=-model.infinity, ub=model.infinity)
              for k in range(n_benchmarks)}
    V  = model.continuous_var(name='V', lb=-model.infinity, ub=model.infinity)
    cb = model.binary_var(name='cb_temp')

    model.add_constraint(model.sum(w) == 1, ctname='budget')

    for k in range(n_benchmarks):
        for t in range(S):
            model.add_constraint(((t + 1) / S) * V <= Vs_all[k][t], ctname=f'maxmin_{k}_{t}')
            soma_t          = np.sum(scenarios[:t + 1, :], axis=0) / S
            scenario_return = model.sum(soma_t[i] * w[i] for i in range(N))
            model.add_constraint(Vs_all[k][t] <= scenario_return - tau_all[k][t],
                                 ctname=f'ssd_{k}_{t}')

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
        lazy_cb.tau_all      = tau_all
        lazy_cb.w_vars       = w
        lazy_cb.Vs_all       = Vs_all
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

    return sol[V], sol.get_objective_value(), weights, solver_info


# ── Walk-forward validation ────────────────────────────────────────────────────

def WFV(prices, returns, benchmark, logging_mode=0, sliding=True, output_dir=None):
    """
    Walk-forward validation for tafscmSSD-sectors.

    Parameters
    ----------
    prices    : OOS asset prices (N assets)
    returns   : full asset returns (train + OOS)
    benchmark : sector benchmark returns aligned to returns.index (K columns, all jointly)
    """
    INITIAL_CAPITAL   = 1_000_000

    weights_invested  = pd.DataFrame()
    rebalances_df     = pd.DataFrame()
    optInfoData       = pd.DataFrame()
    valuation_df      = pd.DataFrame()
    shares_df         = pd.DataFrame()
    solver_stats_df   = pd.DataFrame()
    norm_portfolio_df = pd.DataFrame(columns=['SCM-SSD-sectors'])

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

        v_value, func_obj, weights_arr, solver_info = solve_ssd(
            train, train_bench_sorted, logging_mode=logging_mode, callback=True
        )

        v_df        = pd.DataFrame([v_value], columns=['obj'], index=valid.index)
        weights_opt = pd.DataFrame([weights_arr], columns=valid.columns, index=valid.index)

        solver_stats_df = pd.concat([solver_stats_df,
                                     pd.DataFrame([solver_info], index=valid.index)])

        print(f'Day: {step + 1} ({valid.index.min()})')
        print(f'\tSum proportions: {np.sum(weights_arr) * 100:.4f}%')
        print(f'\tFUNÇÃO OBJETIVO: {func_obj}')

        price = np.array(prices.loc[prices.index[step], :])

        if step == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            valuation     = weights_arr * INITIAL_CAPITAL
            share         = valuation / price
            rebalances_df = pd.concat([rebalances_df, weights_opt])
            optInfoData   = pd.concat([optInfoData, v_df])

            if output_dir is not None:
                date_str = valid.index.min().strftime('%Y-%m-%d')
                train.to_csv(output_dir / f'insample-tafscmssd-sectors-{date_str}.csv', index=False)

        elif step % rebalance_freq == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            date_str = valid.index.min().strftime('%Y-%m-%d')

            if output_dir is not None:
                train.to_csv(output_dir / f'insample-tafscmssd-sectors-{date_str}.csv', index=False)

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

        norm_portfolio_df.loc[date_idx, 'SCM-SSD-sectors'] = np.sum(valuation) / INITIAL_CAPITAL
        last_share = share

        print(f'\tPortfolio Value: {np.sum(valuation):.2f}')
        print(f'\tNorm Portfolio:  {np.sum(valuation) / INITIAL_CAPITAL:.6f}')
        print('---------------------------------------------------------------\n')

    return (norm_portfolio_df, weights_invested, rebalances_df,
            valuation_df, shares_df, optInfoData, solver_stats_df)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='tafscmSSD-sectors — Multi-benchmark scaled SSD com benchmarks setoriais (formulação original)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmarks (GICS sectors): {', '.join(AVAILABLE_BENCHMARKS)}

All selected sectors enter the SAME model: one set of Vs variables and one
set of SSD constraints per sector (K x S). This is the reference formulation
against which tafmSSD-sectors.py (single synthetic benchmark) is checked for
equivalence — both must reach the same optimal V.

Each sector return is the equal-weighted mean of all stocks in that sector.

Examples:
  python tafscmSSD-sectors.py --benchmarks all
  python tafscmSSD-sectors.py --benchmarks TECHNOLOGY FINANCIALS HEALTHCARE
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

    print(f'[INFO]: Algorithm          : tafscmSSD-sectors (Multi-benchmark scaled SSD — setor GICS)')
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

    output_dir = SRC_DIR / 'output' / 'tafscmssd-sectors'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run walk-forward validation (all sectors jointly) ─────────────────────
    (portfolio, weightsInvested, weightsRebalance,
     portfolioValuation, portfolioShares, optInfoData, solverStats) = WFV(
        oosAssets,
        assetReturns,
        benchReturns,
        logging_mode=args.logging_mode,
        sliding=not args.no_sliding,
        output_dir=output_dir,
    )

    print(optInfoData)

    # ── Save solver diagnostics and rebalance weights ─────────────────────────
    solverStats.index.name = 'date'
    solverStats.to_csv(output_dir / 'solver_stats_tafscmssd_sectors.csv')
    print(f'[INFO]: Solver stats saved → {output_dir / "solver_stats_tafscmssd_sectors.csv"}')
    print(solverStats[['status', 'objective', 'best_bound', 'mip_gap',
                       'n_lazy_cuts', 'solve_time_s']])

    weightsRebalance.index.name = 'date'
    weightsRebalance.to_csv(output_dir / 'rebalances_tafscmssd_sectors.csv')
    print(f'[INFO]: Rebalance weights saved → {output_dir / "rebalances_tafscmssd_sectors.csv"}')

    # ── Label and save portfolio ──────────────────────────────────────────────
    bench_label = '+'.join(selected_benchmarks)
    port_label  = f'SCM-SSD-sectors {bench_label}'
    portfolio.rename(columns={'SCM-SSD-sectors': port_label}, inplace=True)

    portfolio.to_csv(output_dir / 'portfolio_tafscmssd_sectors.csv')
    print(f'[INFO]: Portfolio saved → {output_dir / "portfolio_tafscmssd_sectors.csv"}')

    # ── Add sector benchmark series for plotting ──────────────────────────────
    for sector in selected_benchmarks:
        bench_ret = benchReturns.loc[OOS_START:OOS_END, sector]
        bench_cum = (1 + bench_ret).cumprod()
        portfolio[sector] = (bench_cum / bench_cum.iloc[0]).reindex(portfolio.index)
    portfolio['Risk free rate'] = 1.0

    # ── In-sample SSD dominance vs each sector (last rebalancing window) ──────
    last_w      = weightsRebalance.iloc[-1].to_numpy()
    last_idx    = len(assetReturns) - len(oosAssets) + len(weightsRebalance) - 1
    train_start = last_idx
    train_end   = train_start + 200
    train_ret   = assetReturns.iloc[train_start:train_end].to_numpy()
    train_bench = benchReturns.iloc[train_start:train_end]
    r_port      = train_ret @ last_w

    print()
    for sector in selected_benchmarks:
        _, _, ssd_dom = compute_ssd_dominance(r_port, train_bench[sector].to_numpy())
        min_ssd = ssd_dom.min()
        print(f'[INFO]: In-sample SSD dominance vs {sector:<28} — min(SSD) = {min_ssd:+.6f}  '
              f'({"dominates" if min_ssd >= 0 else "does NOT dominate"})')

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_portfolio_comparison(
        portfolio,
        force_background=True,
        benchmark_cols=selected_benchmarks,
        title=f'tafscmSSD-sectors — {port_label} vs benchmarks setoriais ({OOS_START} → {OOS_END})',
        save_path=output_dir / 'performance_comparison.pdf',
    )

    print('\n==========================================================')
    print('Finalising simulation...')
