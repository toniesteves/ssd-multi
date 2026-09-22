
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np


# blotter / metrics are siblings of this file's directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blotter.plots import plot_portfolio_comparison
from metrics.stats import compute_ssd_dominance

import cplex
from docplex.mp.model import Model
from cplex.callbacks import UserCutCallback, LazyConstraintCallback, SolveCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin

np.set_printoptions(precision=8, suppress=True)

pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


def sort_benchmark(benchmark: pd.Series) -> np.ndarray:
    """Ordena os retornos do benchmark em ordem crescente."""
    return np.sort(benchmark.values)


def missing_percentage(df, threshold=0):
    """
    Retorna a quantidade e porcentagem de valores ausentes por coluna,
    filtrando colunas com porcentagem maior que o threshold.

    Se nenhuma coluna tiver valores ausentes, retorna uma mensagem.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame analisado.

    threshold : float
        Percentual mínimo de dados ausentes para exibir a coluna.
        Exemplo: threshold=10 mostra apenas colunas com mais de 10% de NaN.
    """

    missing_pct = df.isnull().mean() * 100

    result = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percentage': missing_pct
    })

    # mantém apenas colunas com pelo menos 1 NaN
    result = result[result['missing_count'] > 0]

    # verifica se existe alguma coluna com NaN
    if result.empty:
        return "Nenhuma coluna possui dados faltantes."

    # aplica threshold
    result = result[result['missing_percentage'] > threshold]

    return result.sort_values(by='missing_percentage', ascending=False)


def remove_empty_samples(df, threshold):
    """
    Remove colunas onde a porcentagem de valores ausentes (NaN) 
    é maior que o threshold fornecido.
    
    Parâmetros:
    df (pd.DataFrame): O dataframe original.
    threshold (float): Valor entre 0 e 1 (ex: 0.30 para remover colunas com >30% de NaN).
    """
    # Calcula a média de valores nulos por coluna
    # Como True=1 e False=0, a média representa a porcentagem de nulos
    colunas_para_manter = df.columns[df.isnull().mean() <= threshold]
    
    return df[colunas_para_manter]


def fill_missing_data(df):
    """
    Preenche dados ausentes usando ffill seguido de bfill.
    
    O ffill preenche os nulos com o valor anterior.
    O bfill preenche os nulos restantes com o próximo valor.
    """
    return df.ffill().bfill()





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
            idx_returns = list(enumerate(portfolio_returns))
            sorted_idx_returns  = sorted(idx_returns, key=lambda x: x[1])
            if self.logging_mode > 3:
                for rank, (original_index, value) in enumerate(sorted_idx_returns, start=1):
                    print(f"\t{rank:2}-th worst return: {value: .5f} (index {original_index})")
                print()

            # Ordena os retornos do portfólio
            sorted_p_returns = sorted(portfolio_returns)
            cum_p_returns    = np.cumsum(sorted_p_returns)
            cum_b_returns    = np.cumsum(self.benchmark)  # já ordenado ascendente

            Vs_values = [self.get_values(var.index) for var in self.Vs_vars]

            max_diff  = -np.inf
            max_index = None

            for s, ret in enumerate(cum_p_returns):
                tau_s    = cum_b_returns[s]
                lhs_expr = Vs_values[s]
                rhs_expr = ret - tau_s
                diff     = lhs_expr - rhs_expr

                if self.logging_mode > 3:
                    print(f"\t|Js| = {s + 1:2d}, C: {lhs_expr:.5f} - ({ret:.5f}) <= ({tau_s:.5f}) (Violated by {diff:.5f})")

                if lhs_expr > rhs_expr + self.tolerance:
                    if diff > max_diff:
                        max_diff  = diff
                        max_index = s

            if max_index is not None:

                if self.logging_mode > 3:
                    print(f"\n[INFO]: SSD constraint violated at Vs_{max_index}\n")
                    print(f"[INFO]: Violation: {max_diff}\n")

                Vs = self.model.get_var_by_name(f"Vs_{max_index}")

                worst_scenario_indices = [idx for idx, _ in sorted_idx_returns[:max_index + 1]]
                if self.debug: print(f"\tWorst Scenario Indices: {worst_scenario_indices}")

                cumulative_scenario = np.sum(self.scenarios[worst_scenario_indices, :], axis=0)
                scenario_return     = self.model.sum(cumulative_scenario[i] * self.w_vars[i] for i in range(n_assets))
                rhs                 = cum_b_returns[max_index]

                if self.logging_mode > 3:
                    if self.debug: print(f"\tWorst Scenario Returns: {[f'{r:.5f}' for r in cumulative_scenario]}")
                    print(f"\t{'Cut to be added':<22}: ", end="")
                    for i in range(n_assets):
                        print(f"{cumulative_scenario[i]:.5f} {self.w_vars[i].name} + ", end="")
                    print(f"1 {Vs} <= {rhs:.5f}\n")

                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(Vs <= scenario_return - rhs)
                self.add(cpx_lhs, sense, cpx_rhs)

                if self.logging_mode > 3:
                    print(f"\t\033[92m> Violated constraint added.\033[0m\n")

            if self.logging_mode > 3:
                print(f"\t> End of Separation Algorithm.")
                print('-' * 40)


def solve_ssd(scenarios, benchmark: np.ndarray, logging_mode=0, callback=False):

    print("\n=== STARTING SSD MODEL ===")
    scenarios  = np.array(scenarios)
    S, N       = scenarios.shape
    MODEL_NAME = "scmSSD"

    model = Model(name='SSD')
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

    # ── Decision variables ────────────────────────────────────────────────
    w  = model.continuous_var_list(N, lb=0, ub=1, name='wl')
    Vs = model.continuous_var_list(S, lb=-model.infinity, ub=model.infinity, name='Vs')
    V  = model.continuous_var(name='V', lb=-model.infinity, ub=model.infinity)
    cb = model.binary_var(name='cb_temp')   # forces MIP so lazy callbacks are active

    # ── Budget constraint ─────────────────────────────────────────────────
    model.add_constraint(model.sum(w) == 1, ctname='budget')

    # ── SSD constraints ───────────────────────────────────────────────────
    benchmark_sorted = np.sort(benchmark)
    cum_b_returns    = np.cumsum(benchmark_sorted)

    for t in range(S):
        model.add_constraint((t + 1) * V <= Vs[t], ctname=f'maxmin_{t}')
        soma_cenarios_t = np.sum(scenarios[:t + 1, :], axis=0)
        scenario_return = model.sum(soma_cenarios_t[i] * w[i] for i in range(N))
        model.add_constraint(Vs[t] <= scenario_return - cum_b_returns[t], ctname=f'ssd_{t}')

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
        lazy_cb.benchmark    = benchmark_sorted
        lazy_cb.w_vars       = w
        lazy_cb.Vs_vars      = Vs
        lazy_cb.logging_mode = logging_mode

    print("\nIniciando processo de otimização...")
    sol = model.solve(log_output=(logging_mode > 1))

    if sol is None:
        raise RuntimeError("No solution found for the optimization problem!")

    weights = np.array([sol[wi] for wi in w])
    print("\n[INFO]: Solução encontrada:")
    print(f"\tV              = {sol[V]}")
    print(f"\tBest Solution  = {sol.get_objective_value()}")
    print(f"\tNº of Cuts    = {model.get_cuts()['user']}")

    return sol[V], sol.get_objective_value(), weights


def WFV(prices, returns, benchmark, logging_mode=0, sliding=True, output_dir=None):

    INITIAL_CAPITAL    = 1000000

    weights_invested   = pd.DataFrame()
    rebalances_df      = pd.DataFrame()
    optInfoData        = pd.DataFrame()
    valuation_df       = pd.DataFrame()
    shares_df          = pd.DataFrame()
    norm_portfolio_df  = pd.DataFrame(columns=["SSD"])

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

        # Single benchmark: extract Series and sort ascending
        bench_col    = train_bench.iloc[:, 0] if isinstance(train_bench, pd.DataFrame) else train_bench
        bench_sorted = sort_benchmark(bench_col)

        if logging_mode == 4:
            print(f"\nRef:\n{bench_col.values}")
            print(f"\nSorted Ref:\n{bench_sorted}")
            for i, col in enumerate(train.columns):
                print(f"\nAsset: {i} - {col}\n{train[col].to_numpy()}")
            print("------------- END INSTANCE --------------")

        v_value, func_obj, weights_arr = solve_ssd(train, bench_sorted, logging_mode=logging_mode, callback=True)

        v_df        = pd.DataFrame([v_value], columns=['obj'], index=valid.index)
        weights_opt = pd.DataFrame([weights_arr], columns=valid.columns, index=valid.index)

        print(f"Day: {step + 1} ({valid.index.min()})")
        print(f"\tSum proportions: {np.sum(weights_arr)*100}%")
        print(f"\tFUNÇÃO OBJETIVO: {func_obj}")

        # ============================================================
        # SIMULATION
        # ============================================================

        price = np.array(prices.loc[prices.index[step], :])

        # ============================================================
        # DIA 0 — primeiro dia da simulação
        # ============================================================
        if step == 0:
            print(f"\n\t\033[92mToday is a rebalancing day.\033[0m")

            valuation     = (weights_arr * INITIAL_CAPITAL)
            share         = (valuation / price)
            rebalances_df = pd.concat([rebalances_df, weights_opt])

            optInfoData   = pd.concat([optInfoData, v_df])

            if output_dir is not None:
                train.to_csv(output_dir / f'insample-scssd-{valid.index.min().strftime("%Y-%m-%d")}.csv', index=False)

        # ============================================================
        # DIA DE REBALANCEAMENTO
        # ============================================================
        elif step % rebalance_freq == 0:

            print(f"\n\t\033[92mToday is a rebalancing day.\033[0m")

            data_str = valid.index.min().strftime('%Y-%m-%d')
            if output_dir is not None:
                train.to_csv(output_dir / f'insample-scssd-{data_str}.csv', index=False)
            optInfoData = pd.concat([optInfoData, v_df])
            
            valuation_normal = (last_share * price)
            total_value      = np.sum(valuation_normal)

            print(f"\tValuation before rebalance: {total_value}")

            valuation     = (weights_arr * total_value)
            share         = (valuation / price)
            rebalances_df = pd.concat([rebalances_df, weights_opt])

        # ============================================================
        # DIA SEM REBALANCEAMENTO
        # ============================================================
        else:
            share     = last_share
            valuation = share * price

        # ============================================================
        # Armazenamento
        # ============================================================
        # break

        date_idx = valid.index[0]

        valuation_df     = pd.concat([valuation_df,pd.DataFrame([valuation], index=[date_idx], columns=valid.columns)])

        shares_df        = pd.concat([shares_df, pd.DataFrame([share], index=[date_idx], columns=valid.columns)])
        weights_invested = pd.concat([weights_invested, weights_opt])

        norm_portfolio_df.loc[date_idx, "SSD"] = (np.sum(valuation) / INITIAL_CAPITAL)

        last_share = share

        print(f"\tPortfolio Value: {np.sum(valuation)}")
        print(f"\tNorm Portfolio: {np.sum(valuation)/INITIAL_CAPITAL}")
        print('---------------------------------------------------------------\n')
        
    return (norm_portfolio_df, weights_invested, rebalances_df, valuation_df, shares_df, optInfoData)



if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')

    SRC_DIR       = Path(__file__).resolve().parent
    DATA_DIR      = SRC_DIR / 'data'
    BENCHMARK_COL = 'SP500'
    N_COLS        = 600
    oos_start     = '2024-01-01'
    oos_end       = '2024-06-01'

    # ── Data loading ──────────────────────────────────────────────────────────
    marketData = pd.read_csv(DATA_DIR / 'marketData.csv', parse_dates=['Date'])
    marketData.set_index('Date', inplace=True)

    cleanData = remove_empty_samples(marketData, 0.15)
    cleanData = fill_missing_data(cleanData)
    print(f'Clean data shape: {cleanData.shape}')

    marketDataPrices = cleanData.copy().abs()

    # ── Returns & benchmark split ─────────────────────────────────────────────
    returns      = (marketDataPrices.diff() / marketDataPrices.shift(1)).iloc[1:]
    benchReturns = returns[[BENCHMARK_COL]]
    assetReturns = returns.drop(columns=[BENCHMARK_COL]).iloc[:, :N_COLS]

    # ── OOS asset prices ──────────────────────────────────────────────────────
    oosPrices  = marketDataPrices.loc[oos_start:oos_end]
    oosAssets  = oosPrices.drop(columns=[BENCHMARK_COL]).iloc[:, :N_COLS]

    print(f'Asset returns : {assetReturns.shape}')
    print(f'Bench returns : {benchReturns.shape}')
    print(f'OOS prices    : {oosAssets.shape}')

    output_dir = SRC_DIR / 'output' / 'scssd'
    os.makedirs(output_dir, exist_ok=True)

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

    # ── Save portfolio results CSV  ───────────
    portfolio.to_csv(output_dir / 'portfolio_scssd.csv')
    print(f'[INFO]: Portfolio saved → {output_dir / "portfolio_scssd.csv"}')


    weightsInvested.to_csv(output_dir / 'weightsInvested_scssd.csv')
    print(f'[INFO]: Weights invested saved → {output_dir / "weightsInvested_scssd.csv"}')


    weightsRebalance.to_csv(output_dir / 'weightsRebalance_scssd.csv')
    print(f'[INFO]: Weights rebalance saved → {output_dir / "weightsRebalance_scssd.csv"}')


    portfolioValuation.to_csv(output_dir / 'portfolioValuation_scssd.csv')
    print(f'[INFO]: Portfolio valuation saved → {output_dir / "portfolioValuation_scssd.csv"}')


    portfolioShares.to_csv(output_dir / 'portfolioShares_scssd.csv')
    print(f'[INFO]: Portfolio shares saved → {output_dir / "portfolioShares_scssd.csv"}')


    optmizationInfoData.to_csv(output_dir / 'optmizationInfoData_scssd.csv')
    print(f'[INFO]: Optimization info data saved → {output_dir / "optmizationInfoData_scssd.csv"}')


    # ── Normalised SP500 from marketData ──────────────────────────────────────
    sp500_slice = marketDataPrices.loc[oos_start:oos_end, BENCHMARK_COL].abs()
    sp500       = sp500_slice / sp500_slice.iloc[0]
    portfolio['SP500']          = sp500
    portfolio['Risk free rate'] = 1.0

    # ── In-sample SSD dominance (last rebalancing window) ────────────────────
    last_w   = weightsRebalance.iloc[-1].to_numpy()
    last_idx = len(returns) - len(oosAssets) + len(weightsRebalance) - 1
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
    print('Finalising simulation...')