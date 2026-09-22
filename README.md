# SSD-Multi — Portfolio Optimization via Second-Order Stochastic Dominance

A research implementation of SSD-based portfolio optimization models supporting single and multiple benchmarks, including a synthetic benchmark constructed from GICS sector indices.

## Models

| Script | Description |
|--------|-------------|
| `src/scSSD.py` | Single-benchmark (S&P 500), scaled. Reference implementation — no CLI. |
| `src/scmSSD.py` | Multi-benchmark, scaled. CLI via `argparse`. |
| `src/usSSD.py` | Single-benchmark, unscaled. |
| `src/tafmSSD.py` | TAF-MultiSSD — reduces K benchmarks to a single synthetic benchmark via the *k-sum* operator. |
| `src/tafmSSD-sectors.py` | TAF-MultiSSD against the 11 GICS sector equal-weighted indices. |

### Mathematical formulation (scmSSD)

```
maximize  V
s.t.      sum(w) = 1,  w >= 0
          (t+1)*V  <=  Vs_{k,t}                              ∀ k, t
          Vs_{k,t} <=  sum_{s=1}^{t}(R_port_s) − τ_{k,t}   [lazy]
```

`τ_{k,t}` is the cumulative sum of the *t* worst returns of benchmark *k*. Lazy constraints are added via CPLEX callbacks — only the most violated constraint is added per iteration.

### Synthetic benchmark (tafmSSD)

```
τ_synth[s] = max_k( cumsum(sorted_b_k)[s] )
```

A single set of `Vs` variables is used against this upper envelope, yielding a pure LP.

## Quick start

All scripts must be run from `src/`:

```bash
cd src/

# Single-benchmark reference
python scSSD.py

# Multi-benchmark (all available benchmarks)
python scmSSD.py --benchmarks all

# Multi-benchmark (selected benchmarks)
python scmSSD.py --benchmarks SP500 HML CMA

# Custom OOS window, expanding window
python scmSSD.py --benchmarks SP500 --oos-start 2024-01-01 --oos-end 2024-06-01 --no-sliding

# TAF-MultiSSD (Fama-French + SP500)
python tafmSSD.py --benchmarks all

# TAF-MultiSSD (GICS sectors)
python tafmSSD-sectors.py --benchmarks all
```

### CLI flags (scmSSD, tafmSSD, tafmSSD-sectors)

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmarks` | required | Space-separated list or `all` |
| `--oos-start` | `2024-01-01` | OOS start date |
| `--oos-end` | `2024-06-01` | OOS end date |
| `--n-cols` | all assets | Max number of assets to use |
| `--no-sliding` | off | Use expanding window instead of sliding |
| `--logging-mode` | `1` | `1` = summary, `2` = verbose |

## Data

| File | Contents |
|------|----------|
| `src/data/marketData.csv` | Daily prices — ~505 S&P 500 constituents + SP500 index (305 rows) |
| `src/data/factors.csv` | Fama-French daily returns: CMA, HML, MARKET, RMW, SMB |
| `src/data/sectors/` | 11 GICS sector equal-weighted return series |

Available benchmarks: `SP500`, `CMA`, `HML`, `MARKET`, `RMW`, `SMB` (and all 11 GICS sectors for the sectors variant).

## Outputs

| Model | Portfolio file | Extras |
|-------|---------------|--------|
| scSSD | `output/scssd/portfolio_scssd.csv` (`SSD` column) | `insample-scssd-*.csv` |
| scmSSD | `output/scmssd/{bench}/portfolio_scmssd_{bench}.csv` (`SSD {BENCH}` column) | `insample-scmssd-*.csv` |
| tafmSSD | `output/tafmssd/portfolio_tafmssd.csv` (`TAF-SSD {bench_label}` column) | `synthetic_benchmark_oos.csv`, `synthetic_bench-{date}.csv` |
| tafmSSD-sectors | `output/tafmssd-sectors/` | same structure as tafmSSD |

## Walk-forward validation

- Training window: 200 days (sliding by default)
- Rebalancing frequency: 21 days
- Default OOS: 2024-01-01 → 2024-06-01 (104 return days)
- Initial capital: R$ 1,000,000

## Project structure

```
ssd-multi/
├── src/
│   ├── scSSD.py
│   ├── scmSSD.py
│   ├── usSSD.py
│   ├── tafmSSD.py
│   ├── tafmSSD-sectors.py
│   ├── blotter/plots.py       # reusable plot library
│   ├── metrics/stats.py       # compute_metrics, compute_ssd_dominance
│   ├── data/
│   │   ├── marketData.csv
│   │   ├── factors.csv
│   │   └── sectors/           # 11 GICS sector CSV files
│   ├── models/                # CPLEX LP model exports (.lp)
│   ├── output/                # generated — not committed
│   └── scripts/
│       └── extract_sectors.py
├── notebooks/
│   ├── scmssd_analysis.ipynb
│   ├── tafmssd_analysis.ipynb
│   ├── tafmssd_sectors_analysis.ipynb
│   └── tafmssd_sectors_analysis_EN.ipynb
├── doc/
│   └── references/            # academic papers
└── requirements.txt
```

## Dependencies

- **Python 3.12**
- **CPLEX 22.1.2** + **docplex 2.31** — required; lazy constraint callbacks will not run without CPLEX
- See `requirements.txt` for the full pinned environment

```bash
pip install -r requirements.txt
```

> CPLEX must be installed separately (IBM academic or commercial license). The free Community Edition is limited to 1000 variables/constraints and will not handle the full dataset.

## Notebooks

```bash
cd /media/toni/Dados/Git/ssd-multi
jupyter lab
```

Notebooks use relative paths from `notebooks/`; the kernel must have `src/` on `sys.path` (handled internally via `sys.path.insert`).

| Notebook | Purpose |
|----------|---------|
| `scmssd_analysis.ipynb` | Full scmSSD analysis — market context, dominance, 6×6 cross-dominance |
| `tafmssd_analysis.ipynb` | Comparative: tafmSSD vs scmSSD vs scSSD — drawdown, dominance, metrics |
| `tafmssd_sectors_analysis.ipynb` | TAF-MultiSSD with 11 GICS sector benchmarks |
| `tafmssd_sectors_analysis_EN.ipynb` | English translation of the sectors analysis |
