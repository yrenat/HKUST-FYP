import math
import networkx as nx
from scipy.stats import norm
from collections import defaultdict

class ProbabilisticRouter:
    def __init__(self, graph, c_val=1.0, lambd=1.0, max_tau=15):
        self.graph = graph
        self.max_tau = max_tau
        
        self.expected_volumes = defaultdict(float)
        self.costs = defaultdict(float)
        
        self.edge_capacities = defaultdict(lambda: c_val)
        self.lambd = lambd
        
        self.m = graph.number_of_edges()
        self.U = max_tau
        
        for u, v in self.graph.edges():
            for tau in range(self.max_tau + 1):
                self.costs[((u, v), tau)] = self._calculate_cost((u, v), tau)

    def _calculate_cost(self, edge, tau):
        c_e = self.edge_capacities[edge]
        exp_vol = self.expected_volumes[(edge, tau)]
        
        base = 1.0 + (1.0 / (2.0 * self.lambd * c_e))
        numerator = math.pow(base, exp_vol)
        denominator = 2.0 * self.U * self.m * c_e
        
        return numerator / denominator

    def get_path_stats(self, path_nodes):
        path_stats = []
        cum_mean = 0.0
        cum_var = 0.0
        
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edge_data = self.graph.edges[u, v]
            
            cum_mean += edge_data['mean']
            cum_var += (edge_data['std'] ** 2)
            
            path_stats.append((cum_mean, math.sqrt(cum_var)))
            
        return path_stats

    def get_location_probability(self, path_stats, edge_index, tau, t_q):
        time_elapsed = tau - t_q
        
        cum_mean_i, cum_std_i = path_stats[edge_index]
        cdf_i = norm.cdf(time_elapsed, loc=cum_mean_i, scale=cum_std_i)
        
        if edge_index == 0:
            cdf_i_minus_1 = 1.0 if time_elapsed >= 0 else 0.0
        else:
            cum_mean_i_minus_1, cum_std_i_minus_1 = path_stats[edge_index - 1]
            cdf_i_minus_1 = norm.cdf(time_elapsed, loc=cum_mean_i_minus_1, scale=cum_std_i_minus_1)
            
        prob = cdf_i_minus_1 - cdf_i
        return max(0, prob)

    def get_expected_travel_time(self, path_nodes):
        time = 0.0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            time += self.graph.edges[u, v]['mean']
        return time

    def find_candidate_paths(self, start_node, end_node, detour_global):
        try:
            shortest_path_nodes = nx.shortest_path(
                self.graph, 
                source=start_node, 
                target=end_node, 
                weight='mean'
            )
        except nx.NetworkXNoPath:
            return [], 0, 0
            
        min_expected_time = self.get_expected_travel_time(shortest_path_nodes)
        
        time_cutoff = min_expected_time * (1.0 + detour_global)
        
        all_paths_nodes = nx.all_simple_paths(
            self.graph, 
            source=start_node, 
            target=end_node
        )
        
        valid_paths = [
            p for p in all_paths_nodes 
            if self.get_expected_travel_time(p) <= time_cutoff
        ]
        
        return valid_paths, min_expected_time, time_cutoff

    def get_expected_path_cost(self, path_nodes, t_q):
        path_stats = self.get_path_stats(path_nodes)
        total_cost = 0.0
        
        for tau in range(self.max_tau + 1):
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                edge = (u, v)
                
                prob = self.get_location_probability(path_stats, i, tau, t_q)
                cost = self.costs[(edge, tau)]
                total_cost += prob * cost
                
        return total_cost

    def route_query(self, t_q, start_node, end_node, detour_global=0.3):
        print(f"\n--- New Query at t_q = {t_q} (from {start_node} to {end_node}) ---")
        
        valid_paths, min_time, cutoff = self.find_candidate_paths(
            start_node, end_node, detour_global
        )
        
        if not valid_paths:
            print("No paths found satisfying the constraints.")
            return None
            
        print(f"Shortest path E[time]: {min_time:.2f}. Cutoff E[time]: {cutoff:.2f}")
        print(f"Found {len(valid_paths)} valid candidate paths.")

        path_costs = {}
        for path in valid_paths:
            cost = self.get_expected_path_cost(path, t_q)
            path_costs[tuple(path)] = cost
        
        for path_tuple, cost in path_costs.items():
            print(f"  Cost for {path_tuple}: {cost:.5f}")

        best_path_tuple = min(path_costs, key=path_costs.get)
        best_path_nodes = list(best_path_tuple)
        
        print(f"==> Decision: Chose {best_path_nodes}")

        self._update_global_state(best_path_nodes, t_q)
        return best_path_nodes

    def _update_global_state(self, path_nodes, t_q):
        path_stats = self.get_path_stats(path_nodes)
        
        for tau in range(self.max_tau + 1):
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                edge = (u, v)
                
                prob = self.get_location_probability(path_stats, i, tau, t_q)
                
                if prob > 0.001: 
                    self.expected_volumes[(edge, tau)] += prob
                    self.costs[(edge, tau)] = self._calculate_cost(edge, tau)
        
    def print_current_loads(self):
        print("\n--- Current Top 5 Expected Loads E[v(e, tau)] ---")
        sorted_loads = sorted(
            self.expected_volumes.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        for (edge, tau), load in sorted_loads[:5]:
            if load > 0.01:
                print(f"  (e={edge}, tau={tau}): {load:.3f}")

G = nx.DiGraph()

edges = [
    ('S', 'A', 2.0, 0.2), ('S', 'B', 3.0, 0.3),
    ('A', 'C', 3.0, 0.3), ('A', 'E', 1.0, 0.1),
    ('B', 'C', 1.0, 0.1), ('B', 'E', 3.0, 0.3),
    ('C', 'D', 2.0, 0.2),
    ('E', 'D', 4.0, 0.4),
]
for u, v, mean, std in edges:
    G.add_edge(u, v, mean=mean, std=std)

router = ProbabilisticRouter(G, max_tau=15)

DETOUR_GLOBAL = 0.3

router.route_query(t_q=0.0, start_node='S', end_node='D', detour_global=DETOUR_GLOBAL)
router.print_current_loads()

router.route_query(t_q=1.0, start_node='B', end_node='A', detour_global=DETOUR_GLOBAL)
router.print_current_loads()

router.route_query(t_q=2.0, start_node='S', end_node='D', detour_global=DETOUR_GLOBAL)
router.print_current_loads()

router.route_query(t_q=3.0, start_node='S', end_node='E', detour_global=DETOUR_GLOBAL)
router.print_current_loads()