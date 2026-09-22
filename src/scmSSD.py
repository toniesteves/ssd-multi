
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


class SSDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback, SolveCallback):

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.n_calls    = 0
        self.tolerance  = 1e-6
        self.debug      = False

    def __call__(self):

        if self.get_cplex_status() == self.status.optimal:
            self.n_calls += 1

            n_assets      = self.scenarios.shape[1]
            curr_solution = self.make_complete_solution()
            V_value       = curr_solution[self.model.get_var_by_name('V')]
            w_values      = [self.get_values(w.index) for w in self.w_vars]

            if self.logging_mode > 3:
                print(f"\n{'V':<5}: {V_value:.5f}")

            portfolio_returns   = self.scenarios @ w_values
            idx_returns         = list(enumerate(portfolio_returns))
            sorted_idx_returns  = sorted(idx_returns, key=lambda x: x[1])
            sorted_p_returns    = sorted(portfolio_returns)
            cum_p_returns       = np.cumsum(sorted_p_returns)

            for k in range(len(self.benchmarks)):
                cum_b_returns = np.cumsum(self.benchmarks[k])
                Vs_vars_k     = self.Vs_all[k]
                Vs_values_k   = [self.get_values(var.index) for var in Vs_vars_k]

                max_diff  = -np.inf
                max_index = None

                for s, ret in enumerate(cum_p_returns):
                    tau_s    = cum_b_returns[s]
                    lhs_expr = Vs_values_k[s]
                    rhs_expr = ret - tau_s
                    diff     = lhs_expr - rhs_expr

                    if self.logging_mode > 3:
                        print(f"\t[Bench {k}] |Js|={s+1:2d} C:{lhs_expr:.5f} ({ret:.5f})<= ({tau_s:.5f}) Violated:{diff:.5f}")

                    if lhs_expr > rhs_expr + self.tolerance:
                        if diff > max_diff:
                            max_diff  = diff
                            max_index = s

                if max_index is not None:
                    if self.logging_mode > 3:
                        print(f"\n[INFO]: SSD constraint violated at Vs_{k}_{max_index}, violation={max_diff}\n")

                    Vs                     = self.model.get_var_by_name(f'Vs_{k}_{max_index}')
                    worst_scenario_indices = [idx for idx, _ in sorted_idx_returns[:max_index + 1]]
                    cumulative_scenario    = np.sum(self.scenarios[worst_scenario_indices, :], axis=0)
                    scenario_return        = self.model.sum(cumulative_scenario[i] * self.w_vars[i] for i in range(n_assets))
                    rhs                    = cum_b_returns[max_index]

                    cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(Vs <= scenario_return - rhs)
                    self.add(cpx_lhs, sense, cpx_rhs)

                    if self.logging_mode > 3:
                        print(f"\t\033[92m> Violated constraint added.\033[0m\n")

            if self.logging_mode > 3:
                print('-' * 40)


def solve_ssd(scenarios, benchmarks, logging_mode=0, callback=False):

    print('\n=== STARTING scmSSD MODEL ===')
    scenarios  = np.array(scenarios)
    S, N       = scenarios.shape
    MODEL_NAME = 'scmSSD'

    if isinstance(benchmarks, np.ndarray) and benchmarks.ndim == 1:
        benchmarks = [benchmarks]
    elif not isinstance(benchmarks, list):
        benchmarks = [np.array(benchmarks)]

    n_benchmarks = len(benchmarks)

    model = Model(name='scmSSD')
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
        benchmark_sorted = np.sort(benchmarks[k])
        cum_b_returns    = np.cumsum(benchmark_sorted)
        for t in range(S):
            model.add_constraint((t + 1) * V <= Vs_all[k][t], ctname=f'maxmin_{k}_{t}')
            soma_t          = np.sum(scenarios[:t + 1, :], axis=0)
            scenario_return = model.sum(soma_t[i] * w[i] for i in range(N))
            model.add_constraint(Vs_all[k][t] <= scenario_return - cum_b_returns[t],
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

    if callback:
        lazy_cb              = model.register_callback(SSDLazyCallback)
        lazy_cb.scenarios    = scenarios
        lazy_cb.benchmarks   = [np.sort(b) for b in benchmarks]
        lazy_cb.w_vars       = w
        lazy_cb.Vs_all       = Vs_all
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

    return sol[V], sol.get_objective_value(), weights


def WFV(prices, returns, benchmark, logging_mode=0, sliding=True, output_dir=None):

    INITIAL_CAPITAL   = 1_000_000

    weights_invested  = pd.DataFrame()
    rebalances_df     = pd.DataFrame()
    optInfoData       = pd.DataFrame()
    valuation_df      = pd.DataFrame()
    shares_df         = pd.DataFrame()
    norm_portfolio_df = pd.DataFrame(columns=['SSD'])

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

        if logging_mode == 4:
            print(f'\nRef:\n{np.array(train_bench).flatten()}')
            for i, col in enumerate(train.columns):
                print(f'\nAsset: {i} - {col}\n{train[col].to_numpy()}')
            print('------------- END INSTANCE --------------')

        v_value, func_obj, weights_arr = solve_ssd(
            train, train_bench_sorted, logging_mode=logging_mode, callback=True
        )

        v_df        = pd.DataFrame([v_value], columns=['obj'], index=valid.index)
        weights_opt = pd.DataFrame([weights_arr], columns=valid.columns, index=valid.index)

        print(f'Day: {step + 1} ({valid.index.min()})')
        print(f'\tSum proportions: {np.sum(weights_arr) * 100}%')
        print(f'\tFUNÇÃO OBJETIVO: {func_obj}')

        price = np.array(prices.loc[prices.index[step], :])

        if step == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            valuation     = weights_arr * INITIAL_CAPITAL
            share         = valuation / price
            rebalances_df = pd.concat([rebalances_df, weights_opt])
            optInfoData   = pd.concat([optInfoData, v_df])
            if output_dir is not None:
                train.to_csv(
                    output_dir / f'insample-scmssd-{valid.index.min().strftime("%Y-%m-%d")}.csv',
                    index=False,
                )

        elif step % rebalance_freq == 0:
            print(f'\n\t\033[92mToday is a rebalancing day.\033[0m')
            data_str = valid.index.min().strftime('%Y-%m-%d')
            if output_dir is not None:
                train.to_csv(output_dir / f'insample-scmssd-{data_str}.csv', index=False)
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

        norm_portfolio_df.loc[date_idx, 'SSD'] = np.sum(valuation) / INITIAL_CAPITAL
        last_share = share

        print(f'\tPortfolio Value: {np.sum(valuation)}')
        print(f'\tNorm Portfolio: {np.sum(valuation) / INITIAL_CAPITAL}')
        print('---------------------------------------------------------------\n')

    return norm_portfolio_df, weights_invested, rebalances_df, valuation_df, shares_df, optInfoData


def parse_args():
    parser = argparse.ArgumentParser(
        description='scmSSD — Multi-benchmark SSD Portfolio Optimization (Walk-Forward)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available benchmarks: {', '.join(AVAILABLE_BENCHMARKS)}

SP500 comes from marketData.csv (price column).
CMA, HML, MARKET, RMW, SMB come from factors.csv (Fama-French factor returns).

Examples:
  python scmSSD.py --benchmarks SP500
  python scmSSD.py --benchmarks SP500 HML CMA
  python scmSSD.py --benchmarks all
  python scmSSD.py --benchmarks MARKET RMW --oos-start 2024-01-01 --oos-end 2024-06-01
        """,
    )
    parser.add_argument(
        '--benchmarks', nargs='+', default=['all'], metavar='BENCHMARK',
        help=(f'Benchmarks to use. '
              f'Options: {", ".join(AVAILABLE_BENCHMARKS)}, all  (default: all)'),
    )
    parser.add_argument(
        '--oos-start', default='2024-01-01',
        help='OOS period start date (default: 2024-01-01)',
    )
    parser.add_argument(
        '--oos-end', default='2024-06-01',
        help='OOS period end date (default: 2024-06-01)',
    )
    parser.add_argument(
        '--n-cols', type=int, default=600,
        help='Number of asset columns to use (default: 600)',
    )
    parser.add_argument(
        '--logging-mode', type=int, default=1, choices=[0, 1, 2, 3, 4],
        help='Verbosity level 0–4 (default: 1)',
    )
    parser.add_argument(
        '--no-sliding', action='store_true',
        help='Use expanding window instead of sliding window',
    )
    return parser.parse_args()


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    args = parse_args()

    # ── Resolve selected benchmarks ───────────────────────────────────────────
    if any(b.lower() == 'all' for b in args.benchmarks):
        selected_benchmarks = AVAILABLE_BENCHMARKS[:]
    else:
        selected_benchmarks = []
        for b in args.benchmarks:
            b_upper = b.upper()
            if b_upper not in AVAILABLE_BENCHMARKS:
                print(f'[ERROR]: Unknown benchmark "{b}".')
                print(f'         Available: {AVAILABLE_BENCHMARKS}')
                sys.exit(1)
            selected_benchmarks.append(b_upper)

    print(f'[INFO]: Selected benchmarks : {selected_benchmarks}')
    print(f'[INFO]: OOS period          : {args.oos_start} → {args.oos_end}')
    print(f'[INFO]: Asset columns       : {args.n_cols}')
    print(f'[INFO]: Window              : {"sliding" if not args.no_sliding else "expanding"}')

    SRC_DIR   = Path(__file__).resolve().parent
    DATA_DIR  = SRC_DIR / 'data'
    OOS_START = args.oos_start
    OOS_END   = args.oos_end
    N_COLS    = args.n_cols

    # ── Load and clean market data ────────────────────────────────────────────
    marketData = pd.read_csv(DATA_DIR / 'marketData.csv', parse_dates=['Date'])
    marketData.set_index('Date', inplace=True)

    cleanData        = remove_empty_samples(marketData, 0.15)
    cleanData        = fill_missing_data(cleanData)
    marketDataPrices = cleanData.copy().abs()

    print(f'[INFO]: Clean data shape    : {marketDataPrices.shape}')

    # ── Compute returns and inject factor returns ─────────────────────────────
    marketDataReturns = marketDataPrices.diff() / marketDataPrices.shift(1)

    factors = pd.read_csv(DATA_DIR / 'factors.csv')
    marketDataReturns[factors.columns] = factors.to_numpy()

    marketDataReturns = marketDataReturns.iloc[1:]  # drop first NaN row

    # ── Split assets and benchmarks ───────────────────────────────────────────
    ALL_BENCH_COLS = ['SP500'] + factors.columns.tolist()

    benchReturns = marketDataReturns[selected_benchmarks]
    assetReturns = marketDataReturns.drop(columns=ALL_BENCH_COLS, errors='ignore').iloc[:, :N_COLS]

    # ── OOS asset prices ──────────────────────────────────────────────────────
    oosPrices = marketDataPrices.loc[OOS_START:OOS_END]
    oosAssets = oosPrices.drop(columns=['SP500']).iloc[:, :N_COLS]

    print(f'[INFO]: Asset returns       : {assetReturns.shape}')
    print(f'[INFO]: Benchmark returns   : {benchReturns.shape}  {list(benchReturns.columns)}')
    print(f'[INFO]: OOS prices          : {oosAssets.shape}')

    # ── Run walk-forward validation per benchmark ─────────────────────────────
    combined = pd.DataFrame()  # aggregates all SSD series for the final combined plot

    for bench in selected_benchmarks:
        print(f'\n{"="*60}')
        print(f'[INFO]: Running scmSSD optimised against benchmark: {bench}')
        print(f'{"="*60}')

        output_dir = SRC_DIR / 'output' / 'scmssd' / bench.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        bench_returns_single = marketDataReturns[[bench]]

        portfolio, weightsInvested, weightsRebalance, portfolioValuation, portfolioShares, optInfoData = WFV(
            oosAssets,
            assetReturns,
            bench_returns_single,
            logging_mode=args.logging_mode,
            sliding=not args.no_sliding,
            output_dir=output_dir,
        )

        print(optInfoData)

        port_label = f'SSD {bench}'
        portfolio.rename(columns={'SSD': port_label}, inplace=True)

        portfolio.to_csv(output_dir / f'portfolio_scmssd_{bench.lower()}.csv')
        print(f'[INFO]: Portfolio saved → {output_dir / f"portfolio_scmssd_{bench.lower()}.csv"}')

        # ── Normalised benchmark series for per-benchmark plot ────────────────
        port_vs_bench = portfolio[[port_label]].copy()
        if bench == 'SP500':
            series = marketDataPrices.loc[OOS_START:OOS_END, 'SP500'].abs()
            port_vs_bench['SP500'] = (series / series.iloc[0]).reindex(portfolio.index)
        elif bench in factors.columns:
            bench_ret_norm = marketDataReturns.loc[OOS_START:OOS_END, bench]
            bench_cum      = (1 + bench_ret_norm).cumprod()
            port_vs_bench[bench] = (bench_cum / bench_cum.iloc[0]).reindex(portfolio.index)
        port_vs_bench['Risk free rate'] = 1.0

        # ── In-sample SSD dominance (last rebalancing window vs this benchmark) ──
        last_w       = weightsRebalance.iloc[-1].to_numpy()
        last_idx     = len(assetReturns) - len(oosAssets) + len(weightsRebalance) - 1
        train_start  = last_idx
        train_end    = train_start + 200
        train_ret    = assetReturns.iloc[train_start:train_end].to_numpy()
        bench_ret_is = marketDataReturns.iloc[train_start:train_end][bench].to_numpy()
        r_port       = train_ret @ last_w

        s_dom, fsd_dom, ssd_dom = compute_ssd_dominance(r_port, bench_ret_is)
        min_ssd = ssd_dom.min()
        print(f'\n[INFO]: In-sample SSD dominance vs {bench} — min(SSD) = {min_ssd:.6f}  '
              f'({"dominates" if min_ssd >= 0 else "does NOT dominate"})')

        combined[port_label] = portfolio[port_label]

    # ── Combined plot: all SSD portfolios + all benchmarks ───────────────────
    for bench in selected_benchmarks:
        if bench == 'SP500':
            series = marketDataPrices.loc[OOS_START:OOS_END, 'SP500'].abs()
            combined['SP500'] = (series / series.iloc[0]).reindex(combined.index)
        elif bench in factors.columns:
            bench_ret_norm = marketDataReturns.loc[OOS_START:OOS_END, bench]
            bench_cum      = (1 + bench_ret_norm).cumprod()
            combined[bench] = (bench_cum / bench_cum.iloc[0]).reindex(combined.index)
    combined['Risk free rate'] = 1.0

    combined_output = SRC_DIR / 'output' / 'scmssd'
    combined_output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_output / 'portfolio_scmssd_combined.csv')

    bench_label = '+'.join(selected_benchmarks)
    plot_portfolio_comparison(
        combined,
        force_background=True,
        benchmark_cols=selected_benchmarks,
        title=f'scmSSD — {bench_label} ({OOS_START} → {OOS_END})',
        save_path=combined_output / 'performance_comparison_combined.png',
    )

    print('\n==========================================================')
    print('Finalising simulation...')
