import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import heapq
import random
from tqdm import tqdm

# ==========================================
# 1. Math Helpers & User's Original Logic
# ==========================================
def phi_inv(p):
    """Inverse CDF of the standard normal distribution."""
    return norm.ppf(p)

# ==========================================
# 2. Real Graph Data Structures
# ==========================================
class Edge:
    def __init__(self, u, v, edge_id, mu, sigma, capacity):
        self.u = u
        self.v = v
        self.id = edge_id
        self.mu = mu
        self.sigma = sigma
        self.capacity = capacity

class Graph:
    def __init__(self):
        self.edges = {}
        self.adj = {}

    def add_edge(self, u, v, edge_id, mu, sigma, capacity):
        edge = Edge(u, v, edge_id, mu, sigma, capacity)
        self.edges[edge_id] = edge
        if u not in self.adj:
            self.adj[u] = []
        self.adj[u].append(edge)

class DynamicQuery:
    def __init__(self, q_id, source, dest, start_time, original_duration, mean_duration):
        self.id = q_id
        self.source = source
        self.dest = dest
        self.start_time = start_time
        self.original_duration = original_duration
        self.mean_duration = mean_duration

# ==========================================
# 3. Real Graph Routing Algorithms
# ==========================================
def find_baseline_cost(graph, source, dest, beta):
    """
    Finds the minimum perceived cost C_base dynamically using Dijkstra.
    Cost = mu_path + Z_beta * sqrt(var_path)
    """
    Z_beta = phi_inv(beta)
    pq = [(0.0, 0.0, 0.0, source)]
    visited = set()
    
    while pq:
        cost, cur_mu, cur_var, u = heapq.heappop(pq)
        
        if u == dest:
            return cost
            
        if u in visited:
            continue
        visited.add(u)
        
        for edge in graph.adj.get(u, []):
            if edge.v not in visited:
                new_mu = cur_mu + edge.mu
                new_var = cur_var + (edge.sigma ** 2)
                new_cost = new_mu + Z_beta * math.sqrt(new_var)
                heapq.heappush(pq, (new_cost, new_mu, new_var, edge.v))
                
    return float('inf')

def find_optimal_system_path(graph, source, dest, beta, delta, C_base, x_weights, start_time):
    """
    Finds the path minimizing system weights x(e, tau) while strictly 
    adhering to the user detour constraint: cost <= (1 + delta) * C_base.
    """
    Z_beta = phi_inv(beta)
    pq = [(0.0, 0.0, 0.0, source, [])]
    best_costs = {} 
    
    best_path = None
    best_sys_weight = float('inf')
    best_path_time = 0.0  
    
    while pq:
        sys_weight, cur_mu, cur_var, u, path = heapq.heappop(pq)
        
        # Prune if it violates the user's reliability constraint
        current_beta_cost = cur_mu + Z_beta * math.sqrt(cur_var)
        if current_beta_cost > (1 + delta) * C_base:
            continue
            
        if u == dest:
            if sys_weight < best_sys_weight:
                best_sys_weight = sys_weight
                best_path = path
                best_path_time = current_beta_cost 
            continue 
            
        tau = int(start_time + cur_mu)
        state_key = (u, tau)
        
        if state_key in best_costs and best_costs[state_key] <= sys_weight:
            continue
        best_costs[state_key] = sys_weight
        
        for edge in graph.adj.get(u, []):
            new_mu = cur_mu + edge.mu
            new_var = cur_var + (edge.sigma ** 2)
            
            edge_weight = x_weights.get((edge.id, tau), 1e-5)
            new_sys_weight = sys_weight + edge_weight
            
            # Prevent cyclic loops in paths
            if edge.v not in [e.u for e in path]:
                heapq.heappush(pq, (new_sys_weight, new_mu, new_var, edge.v, path + [edge]))
            
    return best_path, best_path_time  

# ==========================================
# 4. Integrated Algorithm 3
# ==========================================
def real_algorithm_3_ssor(graph, queries, beta_star, delta, T):
    """
    Routes real queries on the TLC graph.
    """
    mu_sys = {}
    x_weights = {}
    
    capacities = {e_id: e.capacity for e_id, e in graph.edges.items()}
    min_cap = min(capacities.values()) if capacities else 1
    lambda_guess = 1.0 / min_cap
    m = len(queries)
    
    print("Initializing system weights...")
    for e_id, c in tqdm(capacities.items(), desc="Init Weights"):
        for tau in range(T):
            mu_sys[(e_id, tau)] = 0.0
            x_weights[(e_id, tau)] = 1.0 / (2 * m * T * c)
            
    routed_results = []
    
    for q in tqdm(queries, desc="Routing Queries"):
        c_base = find_baseline_cost(graph, q.source, q.dest, beta_star)
        if c_base == float('inf'):
            tqdm.write(f"Query {q.id}: No feasible path found.")
            continue
            
        p_star, p_star_time = find_optimal_system_path(
            graph, q.source, q.dest, beta_star, delta, c_base, x_weights, q.start_time
        )
        
        if not p_star:
            tqdm.write(f"Query {q.id}: No path satisfies the detour constraint.")
            continue
            
        routed_results.append((q.id, [e.id for e in p_star], c_base, p_star_time, q.original_duration, q.mean_duration))
        
        cur_time = q.start_time
        for edge in p_star:
            tau = int(cur_time)
            if tau < T:
                mu_sys[(edge.id, tau)] += 1.0 
                c_e = capacities.get(edge.id, 1)
                
                if x_weights[(edge.id, tau)] > math.exp(0.5) / c_e:
                    lambda_guess *= 2
                    for e_i, c_i in capacities.items():
                        for t_i in range(T):
                            x_weights[(e_i, t_i)] = (1.0 / (2 * m * T * c_i)) * math.exp(mu_sys[(e_i, t_i)] / (lambda_guess * c_i))
                
                x_weights[(edge.id, tau)] *= (1 + 1.0 / (2 * lambda_guess * c_e))
                
            cur_time += edge.mu
            
    if not mu_sys:
        return 0, []
        
    max_load = max([mu_sys[(e, tau)] / capacities.get(e, 1) for e, tau in mu_sys.keys()])
    return max_load, routed_results

# ==========================================
# 5. NYC TLC Data Processing Pipeline
# ==========================================
def standardize_tlc_dataframe(df):
    """
    Standardizes column names across Yellow, Green, FHV, and FHVHV datasets.
    """
    # 1. Standardize Datetime columns
    if 'tpep_pickup_datetime' in df.columns:
        df = df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime'})
    elif 'lpep_pickup_datetime' in df.columns:
        df = df.rename(columns={'lpep_pickup_datetime': 'pickup_datetime', 'lpep_dropoff_datetime': 'dropoff_datetime'})
        
    # 2. Standardize Location ID columns (handle case variations like PUlocationID)
    col_mapping = {}
    for col in df.columns:
        if col.lower() == 'pulocationid':
            col_mapping[col] = 'PULocationID'
        elif col.lower() == 'dolocationid':
            col_mapping[col] = 'DOLocationID'
    df = df.rename(columns=col_mapping)
    
    # 3. Ensure trip_distance exists (FHV might lack it, fill with dummy > 0 to pass filters)
    if 'trip_distance' not in df.columns:
        df['trip_distance'] = 1.0
        
    # Keep only necessary columns to save memory
    cols_to_keep = ['pickup_datetime', 'dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']
    
    # Drop rows missing critical routing data
    df = df.dropna(subset=cols_to_keep)
    return df[cols_to_keep]

def build_graph_and_queries_from_directory(directory_path, num_queries=10):
    """
    Reads all TLC Parquet files in a directory, builds the statistical graph, and samples routing queries.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, directory_path)
    
    # Find all parquet files in the directory
    parquet_files = glob.glob(os.path.join(target_dir, "*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in directory: {target_dir}")
        
    dataframes = []
    print(f"Found {len(parquet_files)} parquet files. Loading and standardizing...")
    
    for file in tqdm(parquet_files, desc="Reading Parquet Files"):
        try:
            temp_df = pd.read_parquet(file)
            temp_df = standardize_tlc_dataframe(temp_df)
            dataframes.append(temp_df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    print("Concatenating all datasets...")
    df = pd.concat(dataframes, ignore_index=True)
    
    print("Cleaning data...")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
    
    df['duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60.0
    
    # Filter out bad data
    df = df[(df['duration_min'] > 0) & (df['duration_min'] < 180)]
    df = df[df['trip_distance'] > 0]
    df = df[df['PULocationID'] != df['DOLocationID']]
    
    print("Aggregating edge statistics (mu, sigma)...")
    stats = df.groupby(['PULocationID', 'DOLocationID'])['duration_min'].agg(['mean', 'std', 'count']).reset_index()
    stats['std'] = stats['std'].fillna(stats['mean'] * 0.1)
    
    # Create a lookup dictionary for the mean duration between source and destination
    mean_lookup = {(int(row['PULocationID']), int(row['DOLocationID'])): row['mean'] for _, row in stats.iterrows()}
    
    print("Building the mathematical graph...")
    g = Graph()
    
    for _, row in tqdm(stats.iterrows(), total=stats.shape[0], desc="Building Graph"):
        u = int(row['PULocationID'])
        v = int(row['DOLocationID'])
        edge_id = f"{u}_{v}"
        mu = row['mean']
        sigma = row['std']
        capacity = max(1, int(row['count'] / 5)) 
        
        g.add_edge(u, v, edge_id, mu, sigma, capacity)
        
    print(f"Graph built with {len(g.edges)} edges.")
    
    print(f"Sampling {num_queries} queries from the dataset...")
    queries = []
    sampled_df = df.sample(n=num_queries, random_state=42)
    
    min_time = sampled_df['pickup_datetime'].min()
    
    for i, (_, row) in enumerate(sampled_df.iterrows()):
        q_id = i + 1
        source = int(row['PULocationID'])
        dest = int(row['DOLocationID'])
        start_time = (row['pickup_datetime'] - min_time).total_seconds() / 60.0
        
        original_duration = row['duration_min']
        mean_duration = mean_lookup.get((source, dest), original_duration)
        
        queries.append(DynamicQuery(q_id, source, dest, start_time, original_duration, mean_duration))
        
    return g, queries

# ==========================================
# 6. Execution Example
# ==========================================
if __name__ == "__main__":
    # Point this to the directory containing your parquet files. 
    # Use "." if the files are in the same directory as the script.
    DATA_DIRECTORY = "." 
    OUTPUT_FILE = "routing_results.txt" # Define the output file name
    
    try:
        g, tlc_queries = build_graph_and_queries_from_directory(DATA_DIRECTORY, num_queries=1000) 
        
        optimal_beta = 0.85      
        delta_detour = 0.3       # Note: This is often referred to as 'alpha' in routing literature for the (1 + alpha) constraint
        T_time_steps = 200       
        
        # --- ADDED PRINT STATEMENTS HERE ---
        print("\n" + "="*40)
        print(f"Optimal Beta (Reliability): {optimal_beta}")
        print(f"Alpha / Delta (Detour Constraint): {delta_detour}")
        print("="*40 + "\n")
        
        print("Running System-Optimal Routing Algorithm...")
        max_expected_load, routes = real_algorithm_3_ssor(
            g, tlc_queries, optimal_beta, delta_detour, T_time_steps
        )
        
        print(f"\nRouting complete. Saving results to {OUTPUT_FILE}...")
        
        # Open the file in write mode
        with open(OUTPUT_FILE, "w") as f:
            
            # Write summary statistics to the file
            summary_header = (
                f"--- Configuration Parameters ---\n"
                f"Optimal Beta (Reliability): {optimal_beta}\n"
                f"Alpha / Delta (Detour Constraint): {delta_detour}\n\n"
                f"--- Results ---\n"
                f"Maximum Expected System Load: {max_expected_load:.4f}\n"
                f"Routed queries:\n"
            )
            
            f.write(summary_header)
            print(f"\n{summary_header}") # Also print to console
            
            for q_id, route, fastest_time, new_time, orig_time, mean_time in routes:
                # Compare against the mean_time instead of orig_time
                ratio = new_time / mean_time if mean_time > 0 else float('inf')
                threshold = (1 + delta_detour) * mean_time
                is_greater = new_time > threshold
                
                # Construct the output string for the current query
                output_str = (
                    f"\nQuery {q_id} routed via zones: {route}\n"
                    f"   -> Original dataset time: {orig_time:.2f} mins\n"
                    f"   -> Mean dataset time (source to dest): {mean_time:.2f} mins\n"
                    f"   -> Algorithm's fastest possible time: {fastest_time:.2f} mins\n"
                    f"   -> Algorithm's system-optimal time:   {new_time:.2f} mins\n"
                    f"   -> System-optimal time is {ratio:.2f} times the mean dataset time\n"
                    f"   -> Is system-optimal time > {(1+delta_detour):.2f}x mean? {'Yes' if is_greater else 'No'} (Threshold: {threshold:.2f} mins)\n"
                )
                
                # Write to file and print to console
                f.write(output_str)
                print(output_str, end="")
                
        print(f"\nAll results successfully saved to {OUTPUT_FILE}.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")