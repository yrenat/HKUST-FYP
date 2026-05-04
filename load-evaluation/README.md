# Stochastic Routing Optimization (RS-SOR)

A two-layer reliability-aware stochastic routing system that jointly optimizes user-level and system-level reliability for urban road networks. Uses NYC TLC taxi trip data as input.

---

## Files

```
stochasticRouting/
├── src/main.cpp                 # RS-SOR algorithm implementation
├── data2026/processed/
│   ├── network_tlc.txt          # Road network (259 nodes, 49223 edges)
│   └── queries_tlc.txt          # Routing queries (up to 50K)
└── README.md
```

---

## Build

```bash
g++ -O2 -std=c++17 -o main src/main.cpp -lm
```

---

## Run

```bash
./main [queries] [alpha] [theta] [K] [detour]
```

| Argument | Meaning | Default |
|----------|---------|---------|
| `queries` | Number of queries to route | 10000 |
| `alpha` | System reliability level α ∈ (0.5, 1) | 0.95 |
| `theta` | Linear choice model sensitivity θ > 0 | 0.1 |
| `K` | K-shortest candidate paths per query | 5 |
| `detour` | Max detour factor (0.3 = 30% longer allowed) | 0.30 |

Examples:
```bash
# Default parameters
./main

# 10K queries, α=0.95, θ=0.1, K=5, detour=30%
./main 10000 0.95 0.1 5 0.3

# 30K queries
./main 30000 0.95 0.1 5 0.3
```

Each run saves results to `results_YYYYMMDD_HHMMSS.txt`.

---

## Data Formats

### Network file (`network_tlc.txt`)
```
V M
u v mean_time std_dev capacity
...
```
- `V` — nodes, `M` — edges
- `mean_time`, `std_dev` — travel time in minutes
- `capacity` — trip count; used as `c(e) = log(trips)`

### Query file (`queries_tlc.txt`)
```
N
id start_t s d
...
```
- `start_t` — departure time step (1 step = 10 min)
- `s`, `d` — origin and destination node IDs

---

## Algorithm Overview

```
Load Network & Queries
        │
        ▼
Generate K-Shortest Paths per query
        │
        ▼
Algorithm 1: Gradient Descent optimization of β*
  · Softmax path distribution weighted by linear cost c_β(p) = μ_p + Z_β·σ_p
  · Minimize max reliable load via gradient descent on β
        │
        ▼
Algorithm 2: RS-SOR online routing with β*
  · Filter β*-feasible paths within detour bound
  · Choose path minimizing multiplicative edge weights
  · Doubling trick for capacity violations
  · Update: x(e,τ) *= (1 + 1/(2λc(e)))^Δμ
        │
        ▼
Output: β*, max system load, comparison vs. shortest path
```

**Objective**: minimize $\max_{e,\tau} \hat{l}(e,\tau,\alpha) / c(e)$ subject to each path being β*-reliable.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| α | 0.95 | System reliability confidence level |
| θ | 0.1 | Linear choice model sensitivity |
| K | 5 | Candidate paths per query |
| Detour | 0.30 | Max allowed path cost ratio above optimal |
| β* | computed | Per-user reliability (output of Algorithm 1) |

---

## Troubleshooting

**Slow candidate generation** — reduce K:
```bash
./main 10000 0.95 0.1 3 0.3
```

**β* stuck at 0.51** — try lower α or increase GD_ITERATIONS in source:
```bash
./main 10000 0.85 0.1 5 0.5
```
