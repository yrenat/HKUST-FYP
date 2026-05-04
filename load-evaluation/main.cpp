#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <unordered_map>
#include <ctime>
#include <sstream>

using namespace std;

// ==========================================
// 1. 全局配置与超参数
// ==========================================
const int MAX_V = 5000;
const int MAX_M = 50000;
const int MAX_T = 200;      // 离散时间步长
const double INF = 1e18;

// 算法超参数
double ALPHA_SYS = 0.95;    // 系统可靠性目标 alpha
double DETOUR_FACTOR = 0.30; // 允许绕路 50%
double THETA = 0.1;         // Linear choice model 分散度参数
double LEARNING_RATE = 0.01;
int GD_ITERATIONS = 100;    // 梯度下降最大迭代次数
double GD_TOLERANCE = 1e-6; // 梯度下降停止条件
int K_PATHS = 5;            // K-Shortest Paths 的 K 值

// ==========================================
// 2. 数学工具函数 (Math Utils)
// ==========================================
const double PI = 3.14159265358979323846;

double norm_pdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * PI);
}

double norm_cdf(double x) {
    return 0.5 * erfc(-x * sqrt(0.5));
}

// 极其精确的逆正态分布 CDF 近似 (Acklam's approximation)
double inv_norm_cdf(double p) {
    if (p <= 0.0) return -INF;
    if (p >= 1.0) return INF;
    // 简化的分段近似，实际工程可用 Boost 库
    double t = sqrt(-2.0 * log(min(p, 1.0 - p)));
    double num = 2.515517 + 0.802853 * t + 0.010328 * t * t;
    double den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t;
    double z = t - num / den;
    return (p < 0.5) ? -z : z;
}

// ==========================================
// 3. 数据结构定义
// ==========================================
struct Edge {
    int to, id;
    double mean, std_dev, cap;
};

struct Path {
    vector<int> edges;           // 路径中的边 ID 列表
    double mu = 0.0;             // 期望行驶时间
    double sigma = 0.0;          // 行驶时间标准差
    double w_prob = 0.0;         // Softmax 概率
    double cost_beta = 0.0;       // c_hat(p, beta) = mu + Z_beta * sigma
};

struct Query {
    int id, start_t, s, d;
    vector<Path> candidates;      // 预处理出的候选路径集 P_q
};

struct NetworkEdge {
    int u, v, id;
    double mean, std_dev, cap;
};

vector<Edge> adj[MAX_V];
int V, M;
vector<Query> queries;
vector<NetworkEdge> edges;

// Per-edge capacity: c(e) = log(trip_count)
double edge_cap[MAX_M];

// 系统状态矩阵
double x_weight[MAX_M][MAX_T];
double expected_load[MAX_M][MAX_T];
double load_variance[MAX_M][MAX_T];  // σ²(e,τ) for reliable load

// 日志输出
ofstream g_output_file;

void log_output(const string& text) {
    cout << text;
    if (g_output_file.is_open()) {
        g_output_file << text;
        g_output_file.flush();
    }
}


// ==========================================
// 4. 数据加载函数
// ==========================================
void load_network(const string& filename) {
    cout << "Loading network from " << filename << endl;
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }
    
    infile >> V >> M;
    cout << "Network: V = " << V << ", M = " << M << endl;
    
    for (int i = 0; i < M; ++i) {
        int u, v;
        double mean, std_dev, cap;
        infile >> u >> v >> mean >> std_dev >> cap;
        
        NetworkEdge ne;
        ne.u = u; ne.v = v; ne.id = i;
        ne.mean = mean; ne.std_dev = std_dev; ne.cap = cap;
        edges.push_back(ne);
        
        Edge e;
        e.to = v; e.id = i;
        e.mean = mean; e.std_dev = std_dev; e.cap = cap;
        adj[u].push_back(e);
        edge_cap[i] = max(1.0, log(cap));  // c(e) = log(trips)
    }
    
    infile.close();
    cout << "Network loaded successfully!" << endl;
}

void load_queries(const string& filename, int max_queries = -1) {
    cout << "Loading queries from " << filename << endl;
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }
    
    int num_queries;
    infile >> num_queries;
    
    // 限制加载的query数量
    int queries_to_load = (max_queries > 0 && max_queries < num_queries) ? max_queries : num_queries;
    
    for (int i = 0; i < queries_to_load; ++i) {
        Query q;
        infile >> q.id >> q.start_t >> q.s >> q.d;
        queries.push_back(q);
    }
    
    infile.close();
    cout << queries_to_load << " queries loaded (out of " << num_queries << " total)!" << endl;
}

// ==========================================
// 5. K-Shortest Paths 生成 (Yen's Algorithm 简化版)
// ==========================================
struct PathNode {
    int node;
    double cost;
    vector<int> path_edges;
    
    bool operator>(const PathNode& other) const {
        return cost > other.cost;
    }
};

// 计算路径的期望成本和方差
void compute_path_statistics(const vector<int>& path_edges, double& mu, double& sigma) {
    mu = 0.0;
    sigma = 0.0;
    for (int eid : path_edges) {
        mu += edges[eid].mean;
        sigma += edges[eid].std_dev * edges[eid].std_dev;
    }
    sigma = sqrt(sigma);
}

// Dijkstra 最短路径（用于计算下界/参考成本）
double dijkstra_shortest(int s, int d) {
    vector<double> dist(V, INF);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<pair<double,int>>> pq;
    dist[s] = 0.0;
    pq.push({0.0, s});
    while (!pq.empty()) {
        auto [c, u] = pq.top(); pq.pop();
        if (c > dist[u]) continue;
        if (u == d) return dist[d];
        for (const Edge& e : adj[u]) {
            double nc = c + e.mean;
            if (nc < dist[e.to]) {
                dist[e.to] = nc;
                pq.push({nc, e.to});
            }
        }
    }
    return dist[d];
}

// K-最短路径生成：带循环检测和成本上界剪枝
vector<Path> generate_ksp_for_query(int s, int d) {
    vector<Path> result;
    
    // 先用 Dijkstra 获取最短路径成本作为剪枝上界
    double sp_cost = dijkstra_shortest(s, d);
    if (sp_cost >= INF) {
        // s->d 不可达，返回默认路径
        Path p;
        p.mu = 100.0;
        p.sigma = 10.0;
        result.push_back(p);
        return result;
    }
    // 不设成本上界，完全依赖 expansion 限制和 K 值
    
    // 记录每个节点被取出的次数（K-shortest paths 经典方法）
    vector<int> pop_count(V, 0);
    
    priority_queue<PathNode, vector<PathNode>, greater<PathNode>> pq;
    PathNode start;
    start.node = s;
    start.cost = 0.0;
    pq.push(start);
    
    int expansions = 0;
    const int MAX_EXPANSIONS = 100000;  // 增大搜索空间
    
    while (!pq.empty() && (int)result.size() < K_PATHS && expansions < MAX_EXPANSIONS) {
        PathNode current = pq.top();
        pq.pop();
        
        pop_count[current.node]++;
        
        // 如果该节点已被取出 K 次以上，跳过
        if (pop_count[current.node] > K_PATHS) continue;
        
        if (current.node == d) {
            Path p;
            p.edges = current.path_edges;
            compute_path_statistics(p.edges, p.mu, p.sigma);
            result.push_back(p);
            continue;
        }
        
        // 检测路径中的循环：不重复访问同一节点
        // 构建已访问节点集合
        set<int> visited_nodes;
        visited_nodes.insert(s);
        for (int eid : current.path_edges) {
            visited_nodes.insert(edges[eid].v);
        }
        
        for (const Edge& e : adj[current.node]) {
            // 跳过已在路径中的节点（避免环）
            if (visited_nodes.count(e.to)) continue;
            
            double nc = current.cost + e.mean;
            
            PathNode next;
            next.node = e.to;
            next.cost = nc;
            next.path_edges = current.path_edges;
            next.path_edges.push_back(e.id);
            pq.push(next);
            expansions++;
        }
    }
    
    // 如果没找到任何路径，返回默认
    if (result.empty()) {
        Path p;
        p.mu = sp_cost;
        p.sigma = sp_cost * 0.2;  // 估计标准差
        result.push_back(p);
    }
    
    return result;
}

void generate_candidates() {
    cout << "\nGenerating candidate paths for all queries..." << endl;
    int total = queries.size();
    for (int i = 0; i < total; ++i) {
        auto& q = queries[i];
        q.candidates = generate_ksp_for_query(q.s, q.d);
        if (q.candidates.empty()) {
            cerr << "Warning: Query " << q.id << " has no candidate paths!" << endl;
        }

    }
    cout << "Candidate paths generated!" << endl;
}
// ==========================================
// 6. 梯度下降优化 Beta* (Algorithm 1 - Gradient Descent)
// ==========================================

// 计算给定 beta 下的期望最大系统负载（使用 Softmax 路径分配）
double compute_expected_max_load(const vector<Query>& Q, double beta, double alpha) {
    double Z_beta = inv_norm_cdf(beta);
    double Z_alpha = inv_norm_cdf(alpha);

    // 稀疏累积期望负载，key = eid * MAX_T + t
    unordered_map<int, double> mu_load, var_load_map;
    mu_load.reserve(200000);
    var_load_map.reserve(200000);

    for (const auto& q : Q) {
        if (q.candidates.empty()) continue;
        int np = q.candidates.size();

        double min_cost = INF;
        for (const auto& p : q.candidates)
            min_cost = min(min_cost, p.mu + Z_beta * p.sigma);

        vector<double> w(np);
        double sw = 0.0;
        for (int i = 0; i < np; i++) {
            w[i] = exp(-THETA * (q.candidates[i].mu + Z_beta * q.candidates[i].sigma - min_cost));
            sw += w[i];
        }

        for (int i = 0; i < np; i++) {
            double prob = w[i] / sw;
            int t = q.start_t;
            for (int eid : q.candidates[i].edges) {
                if (t < MAX_T && eid < M) {
                    int key = eid * MAX_T + t;
                    mu_load[key] += prob;
                    var_load_map[key] += prob * (1.0 - prob);
                }
                t++;
                if (t >= MAX_T) break;
            }
        }
    }

    double max_ratio = 0.0;
    for (auto& kv : mu_load) {
        int eid = kv.first / MAX_T;
        double mu_l = kv.second;
        double var_l = var_load_map.count(kv.first) ? var_load_map.at(kv.first) : 0.0;
        double l_hat = mu_l + Z_alpha * sqrt(var_l);
        double ratio = l_hat / edge_cap[eid];
        if (ratio > max_ratio) max_ratio = ratio;
    }
    return max_ratio;
}

double optimize_beta_gd(vector<Query>& Q, double alpha) {
    cout << "\n========== Algorithm 1: Gradient Descent Optimization of β* ==========" << endl;

    double beta = 0.75;  // 初始猜测
    double prev_beta = beta;
    const double eps = 1e-3;

    for (int iter = 0; iter < GD_ITERATIONS; iter++) {
        double f0 = compute_expected_max_load(Q, beta, alpha);
        double f1 = compute_expected_max_load(Q, min(0.9999, beta + eps), alpha);
        double grad = (f1 - f0) / eps;

        double new_beta = beta - LEARNING_RATE * grad;
        new_beta = max(0.51, min(0.99, new_beta));

        if ((iter + 1) % 10 == 0 || iter == 0)
            cout << "  Iter " << setw(3) << (iter + 1)
                 << ": β=" << fixed << setprecision(6) << new_beta
                 << "  load=" << f0 << "  grad=" << grad << endl;

        if (abs(new_beta - prev_beta) < GD_TOLERANCE) {
            cout << "  Converged at iteration " << (iter + 1) << endl;
            beta = new_beta;
            break;
        }
        prev_beta = beta;
        beta = new_beta;
    }

    cout << "Optimal User Reliability β* = " << fixed << setprecision(6) << beta << endl;
    cout << "========== Algorithm 1 Complete ==========" << endl;
    return beta;
}

// ==========================================
// 7. RS-SOR 在线路由算法 (Algorithm 2 with Linear Choice Model)
// ==========================================
double g_sor_total_mu = 0, g_sor_total_sigma = 0;

double run_rs_sor(vector<Query>& Q, double beta_star, double alpha) {
    cout << "\n========== Algorithm 2: RS-SOR Online Routing (Linear Choice Model) ==========" << endl;
    cout << "Using Beta* = " << fixed << setprecision(4) << beta_star << endl;
    cout << "System Alpha = " << alpha << endl;
    
    // 初始化 (Algorithm 2, line 1-4)
    for (int e = 0; e < M; ++e) {
        for (int t = 0; t < MAX_T; ++t) {
            expected_load[e][t] = 0.0;
            load_variance[e][t] = 0.0;
            x_weight[e][t] = 1.0 / (2.0 * M * MAX_T * edge_cap[e]);  // 1/(2mTc(e))
        }
    }
    
    // λ ← min_e 1/c(e)
    double min_cap = edge_cap[0];
    for (int e = 1; e < M; ++e) min_cap = min(min_cap, edge_cap[e]);
    double lambda = 1.0 / min_cap;
    int doubling_count = 0;
    double total_cost = 0.0;
    
    double Z_beta_star = inv_norm_cdf(beta_star);
    double Z_alpha = inv_norm_cdf(alpha);

    for (auto& q : Q) {
        // Step 1: 生成 β*-可行路径集 (Eq. 2)
        // 注意：这里我们假设候选路径已在 generate_candidates() 中生成
        
        // 计算最小成本基准 (基于 β*)
        double c_base = INF;
        for (const auto& p : q.candidates) {
            double c_p_beta = p.mu + Z_beta_star * p.sigma;
            c_base = min(c_base, c_p_beta);
        }
        
        // 过滤绕路约束内的路径
        vector<int> feasible_indices;
        for (size_t i = 0; i < q.candidates.size(); ++i) {
            double c_p_beta = q.candidates[i].mu + Z_beta_star * q.candidates[i].sigma;
            if (c_p_beta <= (1.0 + DETOUR_FACTOR) * c_base) {
                feasible_indices.push_back(i);
            }
        }
        
        if (feasible_indices.empty()) {
            // 如果没有可行路径，选择最短的
            feasible_indices.push_back(0);
        }
        
        // Step 2: 选择最优路径 p* (Eq. 8)
        // 最小化系统加权成本
        int best_path_idx = feasible_indices[0];
        double min_system_cost = INF;
        
        for (int idx : feasible_indices) {
            double sys_cost = 0.0;
            int t = q.start_t;
            for (int eid : q.candidates[idx].edges) {
                if (t < MAX_T && eid < M) {
                    sys_cost += x_weight[eid][t];
                }
                t++;
                if (t >= MAX_T) break;
            }
            
            if (sys_cost < min_system_cost) {
                min_system_cost = sys_cost;
                best_path_idx = idx;
            }
        }
        
        Path& best_path = q.candidates[best_path_idx];
        total_cost += min_system_cost;
        
        // Step 3: 翻倍检查 (Doubling Trick, Eq. 9)
        bool need_double = false;
        double max_weight = 0.0;
        int t = q.start_t;
        
        for (int eid : best_path.edges) {
            if (t < MAX_T && eid < M) {
                max_weight = max(max_weight, x_weight[eid][t]);
                if (x_weight[eid][t] > exp(0.5) / edge_cap[eid]) {  // x(e,τ) > exp(1/2)/c(e)
                    need_double = true;
                    break;
                }
            }
            t++;
            if (t >= MAX_T) break;
        }
        
        if (need_double) {
            lambda *= 2.0;
            doubling_count++;
            cout << "  Doubling triggered (#" << doubling_count << "), λ = " << lambda << endl;
            
            // 重新初始化权重 (Algorithm 2, line 12)
            for (int e = 0; e < M; ++e) {
                for (int t_step = 0; t_step < MAX_T; ++t_step) {
                    x_weight[e][t_step] = (1.0 / (2.0 * M * MAX_T * edge_cap[e])) * 
                                          exp(expected_load[e][t_step] / (lambda * edge_cap[e]));
                }
            }
            
            // 重新选择路径
            min_system_cost = INF;
            for (int idx : feasible_indices) {
                double sys_cost = 0.0;
                int t_tmp = q.start_t;
                for (int eid : q.candidates[idx].edges) {
                    if (t_tmp < MAX_T && eid < M) {
                        sys_cost += x_weight[eid][t_tmp];
                    }
                    t_tmp++;
                    if (t_tmp >= MAX_T) break;
                }
                
                if (sys_cost < min_system_cost) {
                    min_system_cost = sys_cost;
                    best_path_idx = idx;
                }
            }
            best_path = q.candidates[best_path_idx];
        }
        
        // 记录旅行时间
        g_sor_total_mu += best_path.mu;
        g_sor_total_sigma += best_path.sigma;
        
        // Step 4: 更新系统状态 (Algorithm 2 lines 15-18)
        t = q.start_t;
        for (int eid : best_path.edges) {
            if (t < MAX_T && eid < M) {
                double pi_p = 1.0;  // πp(e,τ) = 1 (deterministic occupancy)
                double delta_mu = pi_p;
                double delta_var = pi_p * (1.0 - pi_p);  // = 0 for deterministic
                expected_load[eid][t] += delta_mu;
                load_variance[eid][t] += delta_var;
                
                // 乘法更新权重 (Algorithm 2, line 18)
                // x(e,τ) *= (1 + 1/(2λc(e)))^Δµ
                double base = 1.0 + 1.0 / (2.0 * lambda * edge_cap[eid]);
                x_weight[eid][t] *= pow(base, delta_mu);
            }
            t++;
            if (t >= MAX_T) break;
        }


    }
    
    // Step 5: 计算 l_hat(e,τ,α)/c(e) (Algorithm 2, line 20-21)
    double Z_alpha_final = inv_norm_cdf(alpha);
    double max_load = 0.0;
    for (int e = 0; e < M; ++e) {
        for (int t = 0; t < MAX_T; ++t) {
            double l_hat = expected_load[e][t] + Z_alpha_final * sqrt(load_variance[e][t]);
            max_load = max(max_load, l_hat / edge_cap[e]);
        }
    }
    
    cout << "\nRS-SOR Routing Complete:" << endl;
    cout << "  Total Queries Routed: " << Q.size() << endl;
    cout << "  Doubling Events: " << doubling_count << endl;
    cout << "  Final λ: " << lambda << endl;
    cout << "  Maximum System Load: " << max_load << endl;
    cout << "  Average Query Cost: " << total_cost / max(1, (int)Q.size()) << endl;
    
    return max_load;
}

// ==========================================
// 8. 主函数
// ==========================================
int main(int argc, char* argv[]) {
    // 记录开始时间并打开输出文件
    time_t start_time = time(0);
    struct tm *tm_info = localtime(&start_time);
    char filename[256];
    strftime(filename, sizeof(filename), "results_%Y%m%d_%H%M%S.txt", tm_info);
    g_output_file.open(filename);
    
    log_output("=====================================================\n");
    log_output("  Stochastic Routing Optimization with Reliability  \n");
    log_output("=====================================================\n\n");
    
    // 参数设置 - 使用TLC数据
    string data_dir = "data2026/processed/";
    string net_file = data_dir + "network_tlc.txt";
    string query_file = data_dir + "queries_tlc.txt";
    
    // 支持命令行参数: ./main [queries] [alpha] [theta] [K] [detour]
    int max_queries = 10000;
    if (argc > 1) max_queries = atoi(argv[1]);
    if (argc > 2) ALPHA_SYS = atof(argv[2]);
    if (argc > 3) THETA = atof(argv[3]);
    if (argc > 4) K_PATHS = atoi(argv[4]);
    if (argc > 5) DETOUR_FACTOR = atof(argv[5]);
    
    if (argc <= 1) {
        cout << "Usage: ./main [queries] [alpha] [theta] [K] [detour]" << endl;
        cout << "Example: ./main 30000 0.95 0.1 5 0.3" << endl;
    }
    
    log_output("Configuration:\n");
    log_output("  Queries = " + to_string(max_queries) + "\n");
    log_output("  System Reliability (alpha) = " + to_string(ALPHA_SYS) + "\n");
    log_output("  Linear Sensitivity (theta) = " + to_string(THETA) + "\n");
    log_output("  K-Shortest Paths (K) = " + to_string(K_PATHS) + "\n");
    log_output("  Detour Factor = " + to_string(DETOUR_FACTOR) + "\n");
    log_output("  Capacity Model: c(e) = log(trips)\n");
    log_output("\n");

    // ========== Pipeline ==========
    
    // Step 1: Load Network and Queries
    log_output("Step 1: Loading Data\n");
    log_output("-------\n");
    load_network(net_file);
    load_queries(query_file, max_queries);
    log_output("\n");

    // Step 2: Generate Candidate Paths
    log_output("Step 2: Generating Candidate Paths\n");
    log_output("-------\n");
    generate_candidates();
    log_output("\n");

    // Step 3: Optimize Beta*
    log_output("Step 3: Finding Optimal User Reliability β*\n");
    log_output("-------\n");
    double beta_opt = optimize_beta_gd(queries, ALPHA_SYS);
    log_output("\n");

    // Step 4: Run RS-SOR with Optimal Beta*
    log_output("Step 4: Running Online Routing Algorithm\n");
    log_output("-------\n");
    double max_system_load = run_rs_sor(queries, beta_opt, ALPHA_SYS);
    log_output("\n");

    // ========== Results Summary ==========
    log_output("=====================================================\n");
    log_output("                   FINAL RESULTS                      \n");
    log_output("=====================================================\n");
    
    stringstream ss;
    ss << fixed << setprecision(4) << beta_opt;
    log_output("Optimal User Reliability Parameter β* = " + ss.str() + "\n");
    
    stringstream ss2;
    ss2 << fixed << setprecision(4) << max_system_load;
    log_output("Maximum System Load = " + ss2.str() + "\n");
    
    stringstream ss3;
    ss3 << fixed << setprecision(4) << ALPHA_SYS;
    log_output("System Confidence Level α = " + ss3.str() + "\n");
    log_output("\n");
    
    // 统计信息
    log_output("Statistics:\n");
    log_output("  Total Vertices: " + to_string(V) + "\n");
    log_output("  Total Edges: " + to_string(M) + "\n");
    log_output("  Total Queries: " + to_string(queries.size()) + "\n");
    
    int total_paths = 0;
    for (const auto& q : queries) {
        total_paths += q.candidates.size();
    }
    log_output("  Total Candidate Paths: " + to_string(total_paths) + "\n");
    
    stringstream ss4;
    ss4 << fixed << setprecision(4) << ((double)total_paths / max(1, (int)queries.size()));
    log_output("  Avg Paths/Query: " + ss4.str() + "\n");
    log_output("\n");
    
    // 负载分析
    double avg_load = 0.0;
    double min_load = INF;
    int nonzero_count = 0;
    
    for (int e = 0; e < M; ++e) {
        for (int t = 0; t < MAX_T; ++t) {
            if (expected_load[e][t] > 1e-6) {
                avg_load += expected_load[e][t];
                min_load = min(min_load, expected_load[e][t]);
                nonzero_count++;
            }
        }
    }
    if (nonzero_count > 0) avg_load /= nonzero_count;
    
    log_output("Load Distribution:\n");
    log_output("  Max Load: " + to_string(max_system_load) + "\n");
    
    stringstream ss5;
    ss5 << fixed << setprecision(4) << avg_load;
    log_output("  Avg Load (nonzero): " + ss5.str() + "\n");
    
    stringstream ss6;
    ss6 << fixed << setprecision(4) << (min_load == INF ? 0.0 : min_load);
    log_output("  Min Load (nonzero): " + ss6.str() + "\n");
    log_output("  Nonzero (e,τ) pairs: " + to_string(nonzero_count) + " / " + to_string(M * MAX_T) + "\n");
    log_output("\n");

    // ========== Shortest Path Baseline Comparison ==========
    log_output("\n=====================================================\n");
    log_output("  Shortest Path Baseline Comparison\n");
    log_output("=====================================================\n");
    
    // Compute SP baseline: each query picks the path with minimum mu
    static double sp_load[MAX_M][MAX_T];
    static double sp_var[MAX_M][MAX_T];
    for (int e = 0; e < M; ++e)
        for (int t = 0; t < MAX_T; ++t) {
            sp_load[e][t] = 0.0;
            sp_var[e][t] = 0.0;
        }
    
    double sp_total_mu = 0, sp_total_sigma = 0;
    for (const auto& q : queries) {
        if (q.candidates.empty()) continue;
        int best = 0;
        for (size_t i = 1; i < q.candidates.size(); ++i)
            if (q.candidates[i].mu < q.candidates[best].mu) best = i;
        const Path& bp = q.candidates[best];
        sp_total_mu += bp.mu;
        sp_total_sigma += bp.sigma;
        int t = q.start_t;
        for (int eid : bp.edges) {
            if (t < MAX_T && eid < M) {
                double pi_p = 1.0;
                sp_load[eid][t] += pi_p;
                sp_var[eid][t] += pi_p * (1.0 - pi_p);
            }
            t++; if (t >= MAX_T) break;
        }
    }
    
    double Z_alpha_sp = inv_norm_cdf(ALPHA_SYS);
    double sp_max_load = 0, sp_avg_load = 0;
    int sp_nonzero = 0;
    for (int e = 0; e < M; ++e)
        for (int t = 0; t < MAX_T; ++t) {
            double l_hat = sp_load[e][t] + Z_alpha_sp * sqrt(sp_var[e][t]);
            double ratio = l_hat / edge_cap[e];
            if (ratio > sp_max_load) sp_max_load = ratio;
            if (sp_load[e][t] > 1e-9) { sp_avg_load += sp_load[e][t]; sp_nonzero++; }
        }
    if (sp_nonzero > 0) sp_avg_load /= sp_nonzero;
    
    double sp_avg_mu = sp_total_mu / queries.size();
    double sor_avg_mu = 0;
    double sor_total_mu = 0;
    for (int e = 0; e < M; ++e)
        for (int t = 0; t < MAX_T; ++t)
            sor_total_mu += expected_load[e][t];  // not travel time, just for load
    
    // Compute RS-SOR avg travel time from expected_load (reconstructed from run)
    // We already have avg_load computed above; for travel time we need to re-trace
    // Instead, compute from the results we already have
    double sor_avg_load_v = avg_load;
    
    {
        stringstream st;
        st << "\n  " << left << setw(22) << "Method"
           << right << setw(12) << "Max Load" << setw(12) << "Avg Load"
           << setw(14) << "Nonzero(e,t)" << setw(14) << "Avg mu(min)" << "\n";
        st << "  " << string(74, '-') << "\n";
        st << "  " << left << setw(22) << "Shortest Path"
           << right << setw(12) << fixed << setprecision(2) << sp_max_load
           << setw(12) << fixed << setprecision(4) << sp_avg_load
           << setw(14) << sp_nonzero
           << setw(14) << fixed << setprecision(2) << sp_avg_mu << "\n";
        st << "  " << left << setw(22) << "RS-SOR (Ours)"
           << right << setw(12) << fixed << setprecision(2) << max_system_load
           << setw(12) << fixed << setprecision(4) << avg_load
           << setw(14) << nonzero_count
           << setw(14) << fixed << setprecision(2) << (g_sor_total_mu / queries.size()) << "\n";
        st << "  " << string(74, '-') << "\n";
        
        double load_reduction = (sp_max_load > 1e-9) ? (sp_max_load - max_system_load) / sp_max_load * 100.0 : 0.0;
        double sor_avg_mu_val = g_sor_total_mu / queries.size();
        double tt_overhead = (sp_avg_mu > 1e-9) ? (sor_avg_mu_val - sp_avg_mu) / sp_avg_mu * 100.0 : 0.0;
        st << "\n  Max Load: SP=" << fixed << setprecision(2) << sp_max_load
           << " -> RS-SOR=" << max_system_load
           << " (reduction: " << showpos << fixed << setprecision(1) << load_reduction << "%" << noshowpos << ")\n";
        st << "  Avg Travel Time: SP=" << fixed << setprecision(2) << sp_avg_mu
           << "min -> RS-SOR=" << sor_avg_mu_val
           << "min (overhead: " << showpos << fixed << setprecision(2) << tt_overhead << "%" << noshowpos << ")\n";
        
        log_output(st.str());
    }

    // 计算运行时间
    time_t end_time = time(0);
    double elapsed = difftime(end_time, start_time);
    int hours = (int)(elapsed / 3600);
    int minutes = (int)((elapsed - hours*3600) / 60);
    int seconds = (int)(elapsed - hours*3600 - minutes*60);
    
    log_output("=====================================================\n");
    log_output("Experiment completed successfully!\n");
    stringstream ss_time;
    ss_time << "Runtime: " << hours << "h " << minutes << "m " << seconds << "s\n";
    log_output(ss_time.str());
    log_output("Output file: " + string(filename) + "\n");
    log_output("=====================================================\n");

    if (g_output_file.is_open()) {
        g_output_file.close();
        cout << "\n✓ Results saved to: " << filename << endl;
    }

    return 0;
}
