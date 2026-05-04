# Overview

This code implements a stochastic and congestion-aware routing algorithm
using NYC TLC taxi trip data. The main purpose is to route many trip
queries while reducing system-wide congestion.

Instead of always sending users to the fastest route, the algorithm also
considers whether some roads or time periods are becoming too crowded.
In this way, it tries to balance two goals:

-   users should not get routes that are much longer than their normal
    route;

-   the whole traffic system should avoid heavy bottlenecks.

The code builds a directed graph from taxi trip records, where each node
is a TLC taxi zone and each edge is a historical trip connection between
two zones.

# Input and Output

## Input

The input should be one or more NYC TLC `.parquet` files. The code
supports different TLC formats, such as Yellow Taxi, Green Taxi, FHV,
and FHVHV data.

The useful columns are:

-   pickup time,

-   drop-off time,

-   pickup location ID,

-   drop-off location ID,

-   trip distance.


## Output

The output is saved to:

``` {.python language="Python"}
routing_results.txt
```

The file includes the chosen route for each query, the original trip
time, the historical mean time, the algorithm's route time, and whether
the route exceeds the detour threshold.

# Main Workflow

The code mainly has three steps:

1.  Read and clean NYC TLC trip data.

2.  Build a stochastic graph from the cleaned data.

3.  Sample trip queries and route them using the congestion-aware
    algorithm.

For each edge, the code stores:

-   mean travel time $\mu_e$,

-   standard deviation $\sigma_e$,

-   estimated capacity $c(e)$.

The reliable cost of a path is calculated as:

$$\hat{c}(p,\beta) = \mu_p + Z_{\beta}\sigma_p$$

where $Z_{\beta} = \Phi^{-1}(\beta)$. This means paths with larger
uncertainty are penalized more.

# Important Parameters

The main parameters are:

``` {.python language="Python"}
optimal_beta = 0.85
delta_detour = 0.3
T_time_steps = 200
```

-   $\beta$: user reliability level. A larger value means the algorithm
    is more careful about uncertain travel times.

-   $\delta$: allowed detour ratio. Here $\delta = 0.3$, so the route
    should be within $1.3$ times the baseline reliable route.

-   $T$: number of time steps used to record congestion.

# Data Processing

The function:

``` {.python language="Python"}
standardize_tlc_dataframe(df)
```

standardizes column names across different TLC datasets.

Then the code removes bad records, including:

-   trips with non-positive duration,

-   trips longer than 180 minutes,

-   zero-distance trips,

-   trips with the same pickup and drop-off zone.

After cleaning, the code groups trips by pickup and drop-off zones:

$$(u,v) = (\text{PULocationID}, \text{DOLocationID})$$

For each pair, it computes the mean, standard deviation, and count of
trip durations. These values become the edge statistics in the graph.

The capacity is estimated by:

$$c(e) = \max \left(1, \left\lfloor \frac{\text{count}(e)}{5} \right\rfloor \right)$$

This is only an approximation based on historical trip frequency.

# Routing Logic

First, the algorithm finds the baseline reliable path cost
$C_{\text{base}}$ using a modified Dijkstra search:

$$C_{\text{base}} = \min_p \left(\mu_p + Z_{\beta}\sigma_p\right)$$

Then it searches for a system-optimal path. The selected path must
satisfy:

$$\hat{c}(p,\beta) \leq (1+\delta)C_{\text{base}}$$

Among all paths satisfying this condition, the algorithm chooses the one
with the smallest system congestion weight.

Each edge-time pair $(e,\tau)$ has a weight $x(e,\tau)$. If many
previous routes use the same edge at the same time, its weight
increases. Therefore, later queries are encouraged to use less crowded
alternatives.

# Main Algorithm

The main function is:

``` {.python language="Python"}
real_algorithm_3_ssor(graph, queries, beta_star, delta, T)
```

For each query, it:

1.  finds the baseline reliable route cost;

2.  finds a congestion-aware route within the detour limit;

3.  records the route;

4.  updates the load and weights of used edge-time pairs.

The load update is:

$$\mu_{\text{sys}}(e,\tau) \leftarrow \mu_{\text{sys}}(e,\tau) + 1$$

The final reported system load is:

$$\max_{e,\tau} \frac{\mu_{\text{sys}}(e,\tau)}{c(e)}$$

A smaller value means the traffic is more evenly spread.

# How to Run

Install the required packages:

``` {.bash language="bash"}
pip install pandas numpy scipy tqdm pyarrow
```

Put the TLC Parquet files in the same folder as the script, or change
`DATA_DIRECTORY`.

Then run:

``` {.bash language="bash"}
python final-report.py
```

The program will generate `routing_results.txt` after finishing.

# Limitations

This implementation is mainly for experiment and testing. Some
limitations are:

-   The graph is based on TLC taxi zones, not real road intersections.

-   Capacity is estimated from historical trip counts.

-   The code uses expected arrival time for time-step assignment.
