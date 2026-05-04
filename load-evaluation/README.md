# Stochastic Routing Optimization (RS-SOR)

A two-layer reliability-aware stochastic routing system that jointly optimizes user-level and system-level reliability for urban road networks. Uses NYC TLC taxi trip data as input.

---

## Project Structure

```
stochasticRouting/
├── src/
│   ├── main.cpp                 # Main RS-SOR algorithm (TLC data)
│   ├── baseline_compare.cpp     # Multi-baseline comparison (SP / Random / Greedy)
│   ├── query_detail_compare.cpp # Per-query detail output and comparison
│   └── tntp_converter.py        # Converts SiouxFalls TNTP data (legacy)
├── data2026/
│   ├── process_tlc.py           # Processes TLC parquet files → network + queries
│   ├── *.parquet                # Raw TLC trip data (excluded from git, see .gitignore)
│   └── processed/
│       ├── network_tlc.txt      # Processed road network
│       ├── queries_tlc.txt      # Processed routing queries
│       └── zone_mapping.txt     # TLC zone ID mapping
├── documents/
│   └── FYP_2026_Apr.pdf         # Project report
├── plot_merged_results.py       # Plot experiment result figures
├── run_experiments.sh           # Batch experiment runner
├── experiment_results.png       # Sample result figure
├── .gitignore                   # Excludes binaries, parquet files, runtime output
└── README.md
```

---

## System Requirements

- **Compiler**: g++ with C++11 support (`g++ --version` ≥ 4.8)
- **Python**: 3.8+ with `pandas`, `numpy`, `pyarrow`, `matplotlib`
- **RAM**: ~500 MB for 50K queries on the TLC network (~260 nodes, ~50K edges)
- **Disk**: ~2 GB for raw TLC parquet files

Install Python dependencies:
```bash
pip install pandas pyarrow numpy matplotlib
```

---

## Step 1 — Prepare Data

The processed files already exist at `data2026/processed/`. **Skip this step** unless you need to regenerate them.

To regenerate from the raw TLC parquet files:
```bash
cd data2026
python3 process_tlc.py
```

This reads all `.parquet` files in `data2026/` (yellow, green, fhvhv, fhv taxi data) and outputs:
- `data2026/processed/network_tlc.txt` — road network (nodes, edges, travel time statistics)
- `data2026/processed/queries_tlc.txt` — routing queries (origin, destination, departure time)

**Alternatively**, to convert a SiouxFalls TNTP dataset instead:
```bash
# Place SiouxFalls_net.tntp and SiouxFalls_trips.tntp in data/raw/
python3 src/tntp_converter.py
# Outputs: data/processed/network.txt and data/processed/queries.txt
```

---

## Step 2 — Build

Compile all three binaries from the project root:

```bash
# Main RS-SOR algorithm
g++ -O2 -std=c++11 -o main src/main.cpp -lm

# Multi-baseline comparison
g++ -O2 -std=c++11 -o baseline_compare src/baseline_compare.cpp -lm

# Per-query detail comparison
g++ -O2 -std=c++11 -o query_detail src/query_detail_compare.cpp -lm
```

---

## Step 3 — Run

### Main experiment (`main`)

```bash
./main [queries] [alpha] [theta] [K] [detour]
```

| Argument | Meaning | Default |
|----------|---------|---------|
| `queries` | Number of queries to route | 10000 |
| `alpha` | System reliability level α ∈ (0.5, 1) | 0.95 |
| `theta` | Linear sensitivity parameter θ > 0 | 0.1 |
| `K` | K-shortest candidate paths per query | 5 |
| `detour` | Maximum detour factor (0.3 = 30% longer) | 0.50 |

Examples:
```bash
# Default: 10K queries, α=0.95, θ=0.1, K=5, detour=50%
./main

# 30K queries with tighter detour constraint
./main 30000 0.95 0.1 5 0.3

# Vary system reliability
./main 10000 0.99 0.1 5 0.5
```

Each run saves results to a timestamped file `results_YYYYMMDD_HHMMSS.txt` in the current directory.

### Baseline comparison (`baseline_compare`)

```bash
./baseline_compare [queries] [alpha] [theta] [K]
```

Runs Shortest Path, Random routing, and RS-SOR side-by-side and prints a comparison table.

### Per-query detail (`query_detail`)

```bash
./query_detail [queries] [alpha] [theta] [K]
```

Outputs per-query routing decisions (chosen path, μ, σ, system cost) for detailed analysis.

### Batch experiments (`run_experiments.sh`)

```bash
bash run_experiments.sh
```

Runs a predefined sweep over α and θ values and prints summarised results.

---

## Step 4 — Plot Results

```bash
python3 plot_merged_results.py
```

Generates a two-panel figure (`experiment_results.png`) comparing:
- **Left panel**: Maximum system load — Shortest Path vs RS-SOR across query counts
- **Right panel**: Load reduction percentage and optimal β* value

The script uses hard-coded result values; edit the arrays at the top of the file to update with new results.

---

## Data Formats

### Network file (`network_tlc.txt`)
```
V M
u v mean_time std_dev capacity
...
```
- `V` — number of nodes, `M` — number of edges
- `mean_time`, `std_dev` — travel time statistics in minutes
- `capacity` — number of historical trips on that OD pair (used as `c(e) = log(trips)`)

### Query file (`queries_tlc.txt`)
```
N
id start_t s d
...
```
- `N` — total number of queries
- `start_t` — discrete departure time step (1 step = 10 minutes)
- `s`, `d` — origin and destination node IDs

---

## Algorithm Overview

```
Load Network & Queries
        │
        ▼
Generate K-Shortest Paths per query  (Yen's / modified Dijkstra)
        │
        ▼
Algorithm 1: Analytical O(1) optimization of β*
  · Compute bottleneck edge under uniform routing
  · Derive network covariance constants Ω_µ, Ω_v
  · Closed-form β* = Φ(Z_β*)
        │
        ▼
Algorithm 2: RS-SOR online routing with β*
  · For each query: filter β*-feasible paths within detour bound
  · Choose path minimizing current multiplicative weights
  · Doubling trick if any edge weight exceeds threshold
  · Update weights: x(e,τ) *= (1 + 1/(2λc(e)))^Δμ
        │
        ▼
Output: max system load, avg travel time, comparison vs. shortest path
```

**Objective**: minimize peak system load $\max_{e,\tau} \hat{l}(e,\tau,\alpha) / c(e)$ while ensuring each routed path is β*-reliable.

---

## Key Parameters Reference

| Parameter | CLI arg | Default | Description |
|-----------|---------|---------|-------------|
| α (alpha) | `argv[2]` | 0.95 | System-level reliability: prob. peak load ≤ computed bound |
| θ (theta) | `argv[3]` | 0.1 | Linear sensitivity of path choice to reliability score |
| K | `argv[4]` | 5 | Candidate paths per query |
| Detour factor | `argv[5]` | 0.50 | Allow paths ≤ (1+factor)×optimal cost |
| β* | computed | — | Per-user reliability parameter (output of Algorithm 1) |

---

## Troubleshooting

**Binary not found / permission denied**
```bash
chmod +x main baseline_compare query_detail
```

**`data2026/processed/` files missing**
```bash
cd data2026 && python3 process_tlc.py
```

**Very slow candidate generation**  
Reduce K or increase the detour factor to prune the search space:
```bash
./main 10000 0.95 1.0 3 0.3
```

**β* always at minimum (0.51)**  
Try a lower α or higher θ to allow more path diversity:
```bash
./main 10000 0.85 2.0 5 0.5
```
