
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

import cplex
from docplex.mp.model import Model
from cplex.callbacks import UserCutCallback, LazyConstraintCallback, SolveCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin

# blotter / metrics are siblings of this file's directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blotter.plots import plot_portfolio_comparison
from metrics.stats import compute_ssd_dominance

np.set_printoptions(precision=8, suppress=True)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


# ── Utilities ──────────────────────────────────────────────────────────────────

def sort_benchmark(benchmark: pd.DataFrame) -> np.ndarray:
    """Aceita um DataFrame de uma coluna e devolve os valores ordenados ascendentemente."""
    if benchmark.shape[1] != 1:
        raise ValueError("O DataFrame deve conter exatamente uma coluna.")
    col = benchmark.columns[0]
    return np.sort(benchmark[col].values)


def missing_percentage(df, threshold=0):
    missing_pct = df.isnull().mean() * 100
    result = pd.DataFrame({
        'missing_count':      df.isnull().sum(),
        'missing_percentage': missing_pct,
    })
    result = result[result['missing_count'] > 0]
    if result.empty:
        return "Nenhuma coluna possui dados faltantes."
    return result[result['missing_percentage'] > threshold].sort_values(
        by='missing_percentage', ascending=False
    )


def remove_empty_samples(df, threshold):
    return df[df.columns[df.isnull().mean() <= threshold]]


def fill_missing_data(df):
    return df.ffill().bfill()


# ── Lazy-constraint callback ───────────────────────────────────────────────────
# Copied verbatim from 09.1 - mSSD + marketData.ipynb (cell 13)

class SSDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback, SolveCallback):

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.n_calls    = 0
        self.tolerance  = 1e-6
        self.debug      = False
        self.cuts_debug = False

    def __call__(self):

        if self.get_cplex_status() == self.status.optimal:
            self.n_calls += 1

            if self.cuts_debug: print('-' * 40)
            if self.cuts_debug: print(f"\n [INFO]: Running separation algorithm #{self.n_calls} ")

            n_scenarios   = len(self.scenarios)
            n_assets      = self.scenarios.shape[1]
            curr_solution = self.make_complete_solution()
            curr_w_values = {var.name: self.get_values(var.index) for var in self.w_vars}
            V_value       = curr_solution[self.model.get_var_by_name('V')]
            w_values      = [self.get_values(w.index) for w in self.w_vars]

            if self.logging_mode > 3:
                print(f"\n{'V':<5}: {V_value:.5f}")
                print(f"Weights:")
                my_array = np.array([w for w in w_values])
                print(f"\n{my_array}")

            # Calculando Tail(R'w) para todo t.
            portfolio_returns = self.scenarios @ w_values

            # Ordenar pelo valor do retorno e Imprimir formatado
            idx_returns        = list(enumerate(portfolio_returns))
            sorted_idx_returns = sorted(idx_returns, key=lambda x: x[1])
            if self.logging_mode > 3:
                for rank, (original_index, value) in enumerate(sorted_idx_returns, start=1):
                    print(f"\t{rank:2}-th worst return: {value: .5f} (index {original_index})")
                print()

            # Ordena os retornos do portfólio e do benchmark
            sorted_p_returns = sorted(portfolio_returns)
            sorted_b_returns = self.benchmark

            # Soma acumulada
            cum_p_returns = np.cumsum(sorted_p_returns)
            cum_b_returns = np.cumsum(sorted_b_returns)

            max_diff  = -np.inf
            max_index = None
            max_ret   = None

            for s, ret in enumerate(cum_p_returns):

                tau_s    = cum_b_returns[s]
                rhs_expr = ret - tau_s
                lhs_expr = V_value

                diff = lhs_expr - rhs_expr
                if self.logging_mode > 3:
                    print(f"\t|Js| = {s + 1:2d}, C: {V_value:.5f} - ({ret:.5f}) <= ({tau_s:.5f}) "
                          f"(Violated by {diff:.5f})")

                if lhs_expr > rhs_expr + self.tolerance:
                    if diff > max_diff:
                        max_diff  = diff
                        max_index = s
                        max_ret   = ret

            if max_index is not None:

                if self.logging_mode > 3:
                    print(f"\n[INFO]: SSD constraint violated to Vs_{max_index}\n")
                    print(f"[INFO]: Violation: {max_diff}\n")

                Vs = self.model.get_var_by_name(f"Vs_{max_index}")

                worst_scenario_indices = [idx for idx, _ in sorted_idx_returns[:max_index + 1]]
                if self.debug: print(f"\tWorst Scenario Indices: {worst_scenario_indices}")

                cumulative_scenario = np.sum(self.scenarios[worst_scenario_indices, :], axis=0)
                scenario_return     = self.model.sum(
                    cumulative_scenario[i] * self.w_vars[i] for i in range(n_assets)
                )
                rhs = cum_b_returns[max_index]

                if self.logging_mode > 3:
                    formatted_values = [f"{ret:.5f}" for ret in cumulative_scenario]
                    if self.debug: print(f"\tWorst Scenario Returns: {formatted_values}")
                    print(f"\t{'Cut to be added':<22}: ", end="")
                    for i in range(n_assets):
                        print(f"{cumulative_scenario[i]:.5f} {self.w_vars[i].name} + ", end="")
                    print(f"1 {Vs} <= {rhs:.5f}\n")

                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(Vs <= scenario_return - rhs)
                self.add(cpx_lhs, sense, cpx_rhs)

                if self.logging_mode > 3:
                    print(f"\t\033[92m> Violated constraint added.\033[0m\n")
                    print(f"\t> End of Separation Algorithm.")
                    print('-' * 40)


# ── Optimisation model ─────────────────────────────────────────────────────────
# Copied verbatim from 09.1 - mSSD + marketData.ipynb (cell 14), path only changed

def solve_ssd(scenarios, benchmark, logging_mode=0, callback=False):
    """
    usSSD: maximises V = min_s Vs[s] subject to budget and a single full-sum
    SSD constraint Vs[S-1] <= sum_portfolio - sum_benchmark.
    All intermediate constraints are generated lazily by SSDLazyCallback.
    """
    print("\n=== STARTING usSSD MODEL ===")
    model         = Model(name='SSD')
    n_scenar      = scenarios.shape[0]
    n_assets      = scenarios.shape[1]
    model_verbose = False
    MODEL_NAME    = "usSSD"

    scenarios = np.array(scenarios)
    benchmark = np.array(benchmark)

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

    w  = model.continuous_var_list(n_assets, lb=0, ub=1, name='wl')
    Vs = model.continuous_var_list(n_scenar, name='Vs', lb=-model.infinity, ub=model.infinity)
    V  = model.continuous_var(name='V', lb=-model.infinity, ub=model.infinity)
    cb = model.binary_var(name='cb_temp')

    model.add_constraint(model.sum(w) == 1, ctname='budget')

    for t in range(n_scenar):
        model.add_constraint(V <= Vs[t], ctname=f'maxmin_{t}')

    # Only one SSD constraint at build time: the full-sum (s = S)
    total_p_return = model.sum(w[i] * sum(scenarios[:, i]) for i in range(n_assets))
    total_b_return = sum(benchmark)
    model.add_constraint(Vs[n_scenar - 1] <= total_p_return - total_b_return, ctname='ssd_base_S')

    model.maximize(V + 0.0 * cb)

    if logging_mode == 3:
        for ct in model.iter_constraints():
            print(f"Name: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}")

    model.export_as_lp(str(Path(__file__).resolve().parent / 'models' / f'{MODEL_NAME}.lp'))

    if logging_mode == 2:
        model.print_information()

    if callback:
        lazy_cb              = model.register_callback(SSDLazyCallback)
        lazy_cb.scenarios    = scenarios
        lazy_cb.benchmark    = benchmark
        lazy_cb.w_vars       = w
        lazy_cb.logging_mode = logging_mode

    print("\nIniciando processo de otimização...")
    if logging_mode > 0:
        model_verbose = True
    sol = model.solve(log_output=model_verbose)

    if sol:
        weights = np.array([sol[wi] for wi in w])
        print("\n[INFO]: Solução encontrada:")
        print(f"\tV              {sol[V]}")
        print(f"\tBest Solution = {sol.get_objective_value()}")
        print(f"\tNº of Cuts    = {model.get_cuts()['user']}")
        print()
        return sol[V], sol.get_objective_value(), weights
    else:
        raise RuntimeError("No solution found for the optimization problem!")


# ── Walk-Forward Validation ────────────────────────────────────────────────────
# Copied verbatim from 09.1 - mSSD + marketData.ipynb (cell 15), path/output_dir added

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
    validation_days    = 1
    rebalance_freq     = 21
    max_steps          = len(prices)
    last_share         = None

    for step in range(0, max_steps, validation_days):

        train_start = step if sliding else 0
        train_end   = train_start + initial_train_days

        train       = returns.iloc[train_start:train_end]
        train_bench = benchmark.iloc[train_start:train_end]
        valid       = returns.iloc[train_end - 1:train_end]

        train_bench_sorted = sort_benchmark(train_bench)

        if logging_mode == 4:
            print(f"\nRef:\n{np.array(train_bench).flatten()}")
            for i, col in enumerate(train.columns):
                print(f"\nAsset: {i} - {col}\n{train[col].to_numpy()}")
            print("------------- END INSTANCE --------------")
            print(f"\nSorted Ref:\n{train_bench_sorted}")

        v_value, func_obj, weights_arr = solve_ssd(
            train, train_bench_sorted, logging_mode=logging_mode, callback=True
        )

        v_df        = pd.DataFrame([v_value], columns=['obj'], index=valid.index)
        weights_opt = pd.DataFrame([weights_arr], columns=valid.columns, index=valid.index)

        print(f"Day: {step + 1} ({valid.index.min()})")
        print(f"\tSum proportions: {np.sum(weights_arr)*100}%")
        print(f"\tFUNÇÃO OBJETIVO: {func_obj}")

        price = np.array(prices.loc[prices.index[step], :])

        # Day 0
        if step == 0:
            print(f"\n\t\033[92mToday is a rebalancing day.\033[0m")
            valuation     = weights_arr * INITIAL_CAPITAL
            share         = valuation / price
            rebalances_df = pd.concat([rebalances_df, weights_opt])
            optInfoData   = pd.concat([optInfoData, v_df])
            if output_dir is not None:
                train.to_csv(
                    output_dir / f'insample-usssd-{valid.index.min().strftime("%Y-%m-%d")}.csv',
                    index=False,
                )

        # Rebalancing day
        elif step % rebalance_freq == 0:
            print(f"\n\t\033[92mToday is a rebalancing day.\033[0m")
            data_str = valid.index.min().strftime('%Y-%m-%d')
            if output_dir is not None:
                train.to_csv(output_dir / f'insample-usssd-{data_str}.csv', index=False)
            optInfoData = pd.concat([optInfoData, v_df])

            total_value = np.sum(last_share * price)
            print(f"\tValuation before rebalance: {total_value}")
            valuation     = weights_arr * total_value
            share         = valuation / price
            rebalances_df = pd.concat([rebalances_df, weights_opt])

        # Hold day
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

        print(f"\tPortfolio Value: {np.sum(valuation)}")
        print(f"\tNorm Portfolio:  {np.sum(valuation)/INITIAL_CAPITAL}")
        print('---------------------------------------------------------------\n')

    return norm_portfolio_df, weights_invested, rebalances_df, valuation_df, shares_df, optInfoData


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    SRC_DIR       = Path(__file__).resolve().parent
    DATA_DIR      = SRC_DIR / 'data'
    BENCHMARK_COL = 'SP500'
    N_COLS        = 600          # same universe as scSSD.py / tafmSSD-sectors.py (2x2 grid)
    oos_start     = '2024-01-01'
    oos_end       = '2024-06-01'

    output_dir = SRC_DIR / 'output' / 'usssd'
    os.makedirs(output_dir, exist_ok=True)

    # ── Data loading ── (no .abs() — matches reference exactly) ──────────────
    marketData = pd.read_csv(DATA_DIR / 'marketData.csv', parse_dates=['Date'])
    marketData.set_index('Date', inplace=True)

    cleanData        = remove_empty_samples(marketData, 0.15)
    cleanData        = fill_missing_data(cleanData)
    marketDataPrices = cleanData.copy()          # NOTE: no .abs() — same as reference
    print(f'Clean data shape: {cleanData.shape}')

    # ── Returns & benchmark split ─────────────────────────────────────────────
    marketDataReturns = marketDataPrices.diff()[1:] / marketDataPrices.shift(1)[1:]
    benchReturns      = marketDataReturns[[BENCHMARK_COL]]
    assetReturns      = marketDataReturns.drop(columns=[BENCHMARK_COL]).iloc[:, :N_COLS]

    # ── OOS asset prices ──────────────────────────────────────────────────────
    oosPrices = marketDataPrices.loc[oos_start:oos_end]
    oosAssets = oosPrices.drop(columns=[BENCHMARK_COL]).iloc[:, :N_COLS]

    print(f'Asset returns : {assetReturns.shape}')
    print(f'Bench returns : {benchReturns.shape}')
    print(f'OOS prices    : {oosAssets.shape}')

    # ── Run simulation ────────────────────────────────────────────────────────
    portfolio, weightsInvested, weightsRebalance, portfolioValuation, portfolioShares, optmizationInfoData = WFV(
        oosAssets,
        assetReturns,
        benchReturns,
        logging_mode=1,
        sliding=True,
        output_dir=output_dir,
    )

    print(optmizationInfoData)

    # ── Save portfolio CSV (same format as scSSD / gSSD) ─────────────────────
    portfolio.to_csv(output_dir / 'portfolio_usssd.csv')
    print(f'[INFO]: Portfolio saved → {output_dir / "portfolio_usssd.csv"}')

    # ── Normalised SP500 ──────────────────────────────────────────────────────
    sp500_slice = marketDataPrices.loc[oos_start:oos_end, BENCHMARK_COL]
    sp500       = sp500_slice / sp500_slice.iloc[0]
    portfolio['SP500']          = sp500
    portfolio['Risk free rate'] = 1.0

    # ── In-sample SSD dominance (last rebalancing window) ────────────────────
    last_w      = weightsRebalance.iloc[-1].to_numpy()
    last_idx    = len(assetReturns) - len(oosAssets) + len(weightsRebalance) - 1
    train_start = last_idx
    train_end   = train_start + 200
    train_ret   = assetReturns.iloc[train_start:train_end].to_numpy()
    bench_ret   = benchReturns.iloc[train_start:train_end].iloc[:, 0].to_numpy()
    r_port      = train_ret @ last_w

    s_dom, fsd_dom, ssd_dom = compute_ssd_dominance(r_port, bench_ret)
    min_ssd = ssd_dom.min()
    print(f'\n[INFO]: In-sample SSD dominance — min(SSD) = {min_ssd:.6f}  '
          f'({"dominates" if min_ssd >= 0 else "does NOT dominate"})')

    plot_portfolio_comparison(portfolio, force_background=True)

    print('\n==========================================================')
    print('Finalising usSSD simulation...')
