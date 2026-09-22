# SSD-Multi — Portfolio Optimization via Second-Order Stochastic Dominance

A research implementation of SSD-based portfolio optimization models supporting single and multiple benchmarks, including a synthetic benchmark constructed from GICS sector indices.

## Models

| Script | Description |
|--------|-------------|
| `src/scSSD.py` | Single-benchmark (S&P 500), scaled. Reference implementation — no CLI. |
| `src/scmSSD.py` | Multi-benchmark, scaled. CLI via `argparse`. |
| `src/usSSD.py` | Single-benchmark, unscaled. |
| `src/tafmSSD.py` | TAF-MultiSSD — reduces K benchmarks to a single synthetic benchmark via the *k-sum* operator. |
| `src/tafmSSD-sectors.py` | TAF-MultiSSD against the 11 GICS sector equal-weighted indices. `--unscaled` switches the max-min constraint from `(s/S)·V <= Vs` to `V <= Vs` (Tail instead of CVaR). |
| `src/tafscmSSD-sectors.py` | Rome MultiSSD with sector benchmarks — all 11 sectors as K separate constraint sets, no synthetic envelope. Reference used for the equivalence test. |

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
| tafmSSD-sectors | `output/tafmssd-sectors/` | same structure as tafmSSD, plus `en/` with English-labelled figures |
| tafmSSD-sectors `--unscaled` | `output/tafmssd-sectors-unscaled/portfolio_tafmssd_sectors.csv` | same structure |
| tafscmSSD-sectors | `output/tafscmssd-sectors/portfolio_tafscmssd_sectors.csv` | `rebalances_*.csv`, `solver_stats_*.csv`, `timing_paired.csv` |

`insample-*.csv` and `src/models/*.lp` are regenerated on every run and are gitignored; portfolios, solver statistics and figures are committed.

## Walk-forward validation

- Training window: 200 days (sliding by default)
- Rebalancing frequency: 21 days
- Default OOS: 2024-01-01 → 2024-06-01 (104 return days)
- Initial capital: R$ 1,000,000

## Project structure

```
ssd-multi/
├── src/
│   ├── scSSD.py                      # Roman Model (Scaled)
│   ├── usSSD.py                      # Roman Model (Unscaled)
│   ├── scmSSD.py                     # Rome MultiSSD — factor benchmarks
│   ├── tafmSSD.py                    # TAF-MultiSSD — factor benchmarks
│   ├── tafmSSD-sectors.py            # TAF-MultiSSD — sector benchmarks (+ --unscaled)
│   ├── tafscmSSD-sectors.py          # Rome MultiSSD — sector benchmarks
│   ├── blotter/plots.py              # reusable plot library
│   ├── metrics/stats.py              # compute_metrics, compute_ssd_dominance
│   ├── data/
│   │   ├── marketData.csv            # daily prices + SP500
│   │   ├── factors.csv               # Fama-French factors
│   │   ├── sectors/                  # 11 GICS sector CSV files
│   │   └── marketDataUS.sqlite       # portSim database — gitignored (~600 MB)
│   ├── models/                       # CPLEX LP exports (.lp) — gitignored
│   ├── output/                       # one folder per model; results are committed,
│   │   │                             # insample-*.csv dumps are gitignored
│   │   ├── scssd/  usssd/  scmssd/  tafmssd/
│   │   ├── tafmssd-sectors/          # + en/ — figures with English labels
│   │   ├── tafmssd-sectors-unscaled/
│   │   └── tafscmssd-sectors/
│   └── scripts/
│       └── extract_sectors.py
├── notebooks/
│   ├── tafmssd_sectors_analysis_v2.ipynb      # current analysis (PT)
│   ├── tafmssd_sectors_analysis_v2_EN.ipynb   # current analysis (EN)
│   ├── tafmssd_sectors_analysis.ipynb         # v1 (PT, superseded)
│   ├── tafmssd_sectors_analysis_EN.ipynb      # v1 (EN, superseded)
│   ├── tafmssd_analysis.ipynb
│   ├── scmssd_analysis.ipynb
│   └── reports/                      # HTML exports without code, for sharing
│       ├── tafmssd_sectors_analysis_v2_report.html
│       └── tafmssd_sectors_analysis_v2_EN_report.html
├── doc/
│   └── references/                   # academic papers
├── media/                            # meeting recordings — gitignored
├── BACKLOG.txt                       # open items from the 2026-07-23 meeting
└── requirements.txt
```

Reports are generated from the notebooks with:

```bash
cd notebooks/
jupyter nbconvert --to html --no-input --output-dir reports \
    --output tafmssd_sectors_analysis_v2_report tafmssd_sectors_analysis_v2.ipynb
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
| `tafmssd_sectors_analysis_v2.ipynb` | **Current analysis (PT)** — sector benchmarks, equivalence test (14c), 2×2 scaled/unscaled grid (14d) |
| `tafmssd_sectors_analysis_v2_EN.ipynb` | **Current analysis (EN)** — same content, figures regenerated with English labels |
| `tafmssd_sectors_analysis.ipynb` | v1 of the sectors analysis (PT) — superseded by v2 |
| `tafmssd_sectors_analysis_EN.ipynb` | v1 of the sectors analysis (EN) — superseded by v2 |
| `tafmssd_analysis.ipynb` | Comparative: tafmSSD vs scmSSD vs scSSD — drawdown, dominance, metrics |
| `scmssd_analysis.ipynb` | Full scmSSD analysis — market context, dominance, 6×6 cross-dominance |

HTML exports without code live in `notebooks/reports/` and are the versions meant for sharing.
