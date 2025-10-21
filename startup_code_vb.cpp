// startup_code.cpp — Variational Bayes for discrete Bayesian Networks with missing data
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <iomanip>
#include <cmath>
#include <climits>
#include <limits>
#include <stdexcept>
#include <random>
#include <algorithm>

using namespace std;

// ========================= Graph_Node =========================
class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;

    // caches
    vector<int> PIdx;                 // parent indices in network order
    vector<int> PRad;                 // parent cardinalities
    vector<int> PStr;                 // mixed-radix strides (rightmost fastest)
    unordered_map<string,int> Val2Idx;

public:
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = std::move(name);
        nvalues = n;
        values = std::move(vals);
    }

    // Original API
    string get_name() { return Node_Name; }
    vector<int> get_children() { return Children; }
    vector<string> get_Parents() { return Parents; }
    vector<float> get_CPT() { return CPT; }
    int get_nvalues() { return nvalues; }
    vector<string> get_values() { return values; }
    void set_CPT(vector<float> new_CPT) { CPT.swap(new_CPT); }
    void set_Parents(vector<string> Parent_Nodes) { Parents = std::move(Parent_Nodes); }
    int add_child(int new_child_index) { for (int c : Children) if (c == new_child_index) return 0; Children.push_back(new_child_index); return 1; }

    // Caches
    void build_value_index() { Val2Idx.clear(); Val2Idx.reserve(values.size()*2+1); for (int i=0;i<(int)values.size();++i) Val2Idx[values[i]] = i; }
    inline int vindex_fast(const string& v) const { auto it = Val2Idx.find(v); return (it==Val2Idx.end()? -1: it->second); }

    void build_parent_cache(const vector<list<Graph_Node>::iterator>& idx2it,
                            const unordered_map<string,int>& name2idx) {
        PIdx.clear(); PRad.clear(); PStr.clear();
        PIdx.reserve(Parents.size()); PRad.reserve(Parents.size());
        for (const auto& pn : Parents) {
            auto it = name2idx.find(pn);
            if (it == name2idx.end()) continue;
            int pi = it->second;
            PIdx.push_back(pi);
            PRad.push_back(idx2it[pi]->get_nvalues());
        }
        PStr.assign(PRad.size(), 0);
        int stride = 1;
        for (int i=(int)PRad.size()-1; i>=0; --i) {
            PStr[i] = stride;
            if (PRad[i] > 0 && stride > INT_MAX/PRad[i]) throw runtime_error("Parent combos overflow");
            stride *= PRad[i];
        }
    }
    inline const vector<int>& parents_idx() const { return PIdx; }
    inline const vector<int>& parent_rad()  const { return PRad; }
    inline const vector<int>& parent_str()  const { return PStr; }
    inline int row_base_from_assign_code(int code) const { return code * nvalues; }
};

// ============================ network ============================
class network {
    list<Graph_Node> Pres_Graph;

    // caches
    vector<list<Graph_Node>::iterator> index_cache;
    unordered_map<string,int> name2idx;

public:
    int addNode(Graph_Node node) { Pres_Graph.push_back(std::move(node)); return 0; }
    list<Graph_Node>::iterator getNode(int i) {
        int count=0; for (auto it=Pres_Graph.begin(); it!=Pres_Graph.end(); ++it) if (count++==i) return it;
        return Pres_Graph.end();
    }
    int netSize() { return (int)Pres_Graph.size(); }
    int get_index(string val_name) { int c=0; for (auto it=Pres_Graph.begin(); it!=Pres_Graph.end(); ++it,++c) if (it->get_name()==val_name) return c; return -1; }
    list<Graph_Node>::iterator get_nth_node(int n) {
        int c=0; for (auto it=Pres_Graph.begin(); it!=Pres_Graph.end(); ++it,++c) if (c==n) return it;
        return Pres_Graph.end();
    }
    list<Graph_Node>::iterator search_node(string val_name) {
        for (auto it=Pres_Graph.begin(); it!=Pres_Graph.end(); ++it) if (it->get_name()==val_name) return it;
        cout << "node not found: " << val_name << "\n";
        return Pres_Graph.end();
    }

    // finalize caches
    void finalize_index_cache() {
        index_cache.clear(); index_cache.reserve(Pres_Graph.size());
        name2idx.clear();    name2idx.reserve(Pres_Graph.size()*2+1);
        int i=0; for (auto it=Pres_Graph.begin(); it!=Pres_Graph.end(); ++it,++i) {
            index_cache.push_back(it); name2idx[it->get_name()] = i;
        }
    }
    list<Graph_Node>::iterator fast_get(int i) { return index_cache[i]; }
    int fast_index_of(const string& name) const { auto it=name2idx.find(name); return (it==name2idx.end()? -1: it->second); }
    const vector<list<Graph_Node>::iterator>& get_index_cache() const { return index_cache; }
    const unordered_map<string,int>& get_name_map() const { return name2idx; }
};

// ====================== misc helpers & I/O ======================
static inline string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n"); if (string::npos==first) return str;
    size_t last = str.find_last_not_of(" \t\r\n"); return str.substr(first, (last-first+1));
}

network read_network(const char* filename) {
    network BayesNet;
    string line;
    ifstream myfile(filename);
    if (!myfile.is_open()) { cout << "Error: Could not open file " << filename << endl; return BayesNet; }

    while (getline(myfile, line)) {
        line = trim(line);
        if (line.empty() || line[0]=='#') continue;

        string token; { stringstream ss(line); ss >> token; }

        if (token == "variable") {
            string var_name; { stringstream ss(line); ss >> token >> var_name; }
            if (!getline(myfile, line)) break;
            stringstream ss2(line);
            string type_keyword, discrete_keyword, bracket, equals;
            int num_values; ss2 >> type_keyword >> discrete_keyword >> bracket >> num_values >> bracket >> equals >> bracket;

            vector<string> values;
            string value;
            while (ss2 >> value) {
                if (value == "};") break;
                if (!value.empty() && value.back()==',') value.pop_back();
                values.push_back(value);
            }
            BayesNet.addNode(Graph_Node(var_name, num_values, values));
        } else if (token == "probability") {
            string full = line;
            while (full.find('{') == string::npos) {
                string next_line; if (!getline(myfile, next_line)) break;
                full += " " + trim(next_line);
            }
            size_t lp=full.find('('), rp=full.find(')'), bar=full.find('|');
            if (lp==string::npos || rp==string::npos) continue;

            string inside = full.substr(lp+1, rp-lp-1);
            string node_name; { stringstream hs(inside); hs >> node_name; }

            auto listIt = BayesNet.search_node(node_name);
            int index   = BayesNet.get_index(node_name);

            vector<string> parents;
            if (bar!=string::npos && bar<rp) {
                string parents_str = full.substr(bar+1, rp-bar-1);
                stringstream ps(parents_str);
                string p;
                while (ps >> p) {
                    if (!p.empty() && p.back()==',') p.pop_back();
                    parents.push_back(p);
                    auto pit = BayesNet.search_node(p);
                    pit->add_child(index);
                }
            }
            listIt->set_Parents(parents);

            vector<float> cpt;
            while (getline(myfile, line)) {
                line = trim(line);
                if (line == "};") break;
                if (line.empty()) continue;

                size_t close_paren = line.find(')');
                string part;
                if (close_paren != string::npos) part = line.substr(close_paren + 1);
                else if (line.find("table") != string::npos) { size_t t=line.find("table"); part = line.substr(t+5); }
                else part = line;

                string tok; stringstream ss_prob(part);
                while (ss_prob >> tok) {
                    while (!tok.empty() && (tok.back()==',' || tok.back()==';')) tok.pop_back();
                    if (!tok.empty() && (isdigit((unsigned char)tok[0]) || tok[0]=='.' || tok[0]=='-' || tok[0]=='+'))
                        cpt.push_back(static_cast<float>(atof(tok.c_str())));
                }
            }
            listIt->set_CPT(cpt);
        }
    }
    myfile.close();

    BayesNet.finalize_index_cache();
    const int N = BayesNet.netSize();
    for (int i=0;i<N;++i) BayesNet.fast_get(i)->build_value_index();
    for (int i=0;i<N;++i) BayesNet.fast_get(i)->build_parent_cache(BayesNet.get_index_cache(), BayesNet.get_name_map());
    return BayesNet;
}

void write_network(const char* filename, network& BayesNet) {
    ofstream outfile(filename);
    if (!outfile.is_open()) { cout << "Error: Could not open file " << filename << " for writing" << endl; return; }

    outfile << "// Bayesian Network" << endl << endl;

    int N = BayesNet.netSize();

    for (int i=0;i<N;i++) {
        auto node = BayesNet.get_nth_node(i);
        outfile << "variable " << node->get_name() << " {\n";
        outfile << "  type discrete [ " << node->get_nvalues() << " ] = { ";
        vector<string> vals = node->get_values();
        for (int j=0;j<(int)vals.size();j++) { outfile << vals[j]; if (j<(int)vals.size()-1) outfile << ", "; }
        outfile << " };\n}\n";
    }

    outfile << std::fixed << std::setprecision(6);
    for (int i=0;i<N;i++) {
        auto node = BayesNet.get_nth_node(i);
        vector<string> parents = node->get_Parents();
        vector<string> values  = node->get_values();
        vector<float> cpt      = node->get_CPT();

        outfile << "probability ( " << node->get_name();
        if (!parents.empty()) {
            outfile << " | ";
            for (int j=0;j<(int)parents.size();j++) { outfile << parents[j]; if (j<(int)parents.size()-1) outfile << ", "; }
        }
        outfile << " ) {\n";

        // parent radices
        vector<int> radices; radices.reserve(parents.size());
        for (auto &pname : parents) {
            auto pnode = BayesNet.search_node(pname);
            radices.push_back(pnode->get_nvalues());
        }
        int parent_combinations = 1; for (int r:radices) parent_combinations *= r;

        int cpt_index = 0;
        if (parents.empty()) {
            outfile << "    table ";
            for (int k=0;k<(int)values.size();k++) {
                if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++]; else outfile << "-1";
                if (k<(int)values.size()-1) outfile << ", ";
            }
            outfile << ";\n";
        } else {
            for (int comb=0; comb<parent_combinations; ++comb) {
                vector<int> idx(parents.size(),0);
                int tmp=comb; for (int p=(int)parents.size()-1; p>=0; --p) { idx[p]=tmp%radices[p]; tmp/=radices[p]; }

                outfile << "    ( ";
                for (int p=0;p<(int)parents.size();++p) {
                    auto pnode = BayesNet.search_node(parents[p]);
                    auto pvals = pnode->get_values();
                    outfile << pvals[idx[p]];
                    if (p<(int)parents.size()-1) outfile << ", ";
                }
                outfile << " ) ";

                for (int k=0;k<(int)values.size();k++) {
                    if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++]; else outfile << "-1";
                    if (k<(int)values.size()-1) outfile << ", ";
                }
                outfile << ";\n";
            }
        }
        outfile << "};\n\n";
    }
    outfile.close();
    cout << "Network written to file: " << filename << endl;
}

// legacy helpers (kept)
int value_index(const vector<string>& values, const string& val) { for (int i=0;i<(int)values.size();++i) if (values[i]==val) return i; return -1; }
int get_cpt_index(network& net, Graph_Node& node, const vector<int>& assignment) {
    vector<string> parents = node.get_Parents();
    vector<int> radices; for (const string& p : parents) { auto pnode = net.search_node(p); radices.push_back(pnode->get_nvalues()); }
    int index=0, factor=1; for (int i=(int)parents.size()-1;i>=0;--i) { index += assignment[i]*factor; factor*=radices[i]; }
    return index * node.get_nvalues();
}

// ========================= FAST helpers =========================
inline int parent_code_fast(network& net, Graph_Node& node, const vector<string>& row) {
    const auto& PIdx = node.parents_idx(); const auto& PStr = node.parent_str();
    int code=0;
    for (int p=0;p<(int)PIdx.size();++p) {
        auto pnode = net.fast_get(PIdx[p]);
        const string& tok = row[PIdx[p]];
        if (tok == "?") return -1;
        int a = pnode->vindex_fast(tok); if (a<0) return -1;
        code += a * PStr[p];
    }
    return code;
}
inline int parent_code_with_override(network& net, Graph_Node& node, const vector<string>& row,
                                     int override_idx, int override_val) {
    const auto& PIdx = node.parents_idx(); const auto& PStr = node.parent_str();
    int code=0;
    for (int p=0;p<(int)PIdx.size();++p) {
        int idx = PIdx[p]; int a;
        if (idx == override_idx) a = override_val;
        else {
            auto pnode = net.fast_get(idx);
            const string& tok = row[idx];
            if (tok == "?") return -1;
            a = pnode->vindex_fast(tok); if (a<0) return -1;
        }
        code += a * PStr[p];
    }
    return code;
}

// ============ Initialization: complete cases + Laplace(+1) ============
void initialize_cpts_complete_case(network& net, const vector<vector<string>>& data) {
    const int N = net.netSize();
    vector<vector<float>> counts(N);
    vector<int> nvals(N), parent_combos(N);

    for (int i=0;i<N;++i) {
        auto node = net.fast_get(i);
        nvals[i] = node->get_nvalues();
        long long prod=1; for (int r: node->parent_rad()) { if (r<=0 || prod > (LLONG_MAX/r)) throw runtime_error("Parent combos too large"); prod*=r; }
        parent_combos[i] = (int)prod;
        counts[i].assign(parent_combos[i]*nvals[i], 1.0f); // Laplace +1 per cell
    }

    for (const auto& row : data) {
        if ((int)row.size()!=N) continue;
        for (int i=0;i<N;++i) {
            auto node = net.fast_get(i);
            const string& tok = row[i];
            if (tok=="?") continue;
            int x = node->vindex_fast(tok); if (x<0) continue;
            int code = parent_code_fast(net, *node, row); if (code<0) continue;
            int base = node->row_base_from_assign_code(code);
            counts[i][base+x] += 1.0f;
        }
    }

    for (int i=0;i<N;++i) {
        auto node = net.fast_get(i);
        auto& c = counts[i]; const int K = nvals[i];
        for (int off=0; off<(int)c.size(); off+=K) {
            float s=0.f; for (int k=0;k<K;++k) s += c[off+k];
            if (s<=0.f) { for (int k=0;k<K;++k) c[off+k] = 1.0f/K; }
            else { float inv=1.0f/s; for (int k=0;k<K;++k) c[off+k]*=inv; }
        }
        node->set_CPT(c);
    }
    cerr << "Initialized CPTs from complete cases with Laplace (+1)." << endl;
}

// ================== Math utils: digamma & log-sum-exp ==================
static inline double digamma(double x) {
    // Simple digamma approximation with recurrence + asymptotic expansion
    double result = 0.0;
    // Reflect for small x if needed (x>0 assumed here; our α>0)
    while (x < 6.0) { result -= 1.0/x; x += 1.0; }
    double inv = 1.0/x;
    double inv2 = inv*inv;
    // asymptotic: psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    result += std::log(x) - 0.5*inv - inv2*(1.0/12.0) + inv2*inv2*(1.0/120.0) - inv2*inv2*inv2*(1.0/252.0);
    return result;
}
static inline double logsumexp(const vector<double>& a) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : a) m = std::max(m, v);
    if (!isfinite(m)) return m;
    double s = 0.0; for (double v : a) s += std::exp(v - m);
    return m + std::log(std::max(s, 1e-300));
}

// ================== VB helpers: E[log θ] precompute ====================
struct VBDirichlet {
    // Dirichlet parameters α for each node: flattened (rows * K)
    vector<vector<double>> alpha;
    // Cached E[log θ_k] per cell: same shape
    vector<vector<double>> elog;
    // Rows and K per node
    vector<int> rows, K;

    void init_from_net(network& net, double alpha0=1.0) {
        int N = net.netSize();
        alpha.resize(N); elog.resize(N); rows.resize(N); K.resize(N);
        for (int i=0;i<N;++i) {
            auto node = net.fast_get(i);
            int k = node->get_nvalues(); K[i] = k;
            long long prod=1; for (int r : node->parent_rad()) { if (r<=0 || prod > (LLONG_MAX/r)) throw runtime_error("Parent combos too large"); prod*=r; }
            rows[i] = (int)prod;
            alpha[i].assign(rows[i]*k, alpha0); // symmetric Dirichlet prior
            elog[i].assign(rows[i]*k, 0.0);
        }
    }
    void set_from_counts_plus_prior(int i, const vector<double>& counts, double alpha0) {
        // counts must be length rows*K
        alpha[i].resize(counts.size());
        for (size_t j=0;j<counts.size();++j) alpha[i][j] = alpha0 + counts[j];
    }
    void update_elog() {
        int N = (int)alpha.size();
        for (int i=0;i<N;++i) {
            int k = K[i]; int R = rows[i];
            auto& a = alpha[i]; auto& e = elog[i];
            for (int r=0;r<R;++r) {
                double sum_a = 0.0; for (int t=0;t<k;++t) sum_a += a[r*k + t];
                double dig_sum = digamma(sum_a);
                for (int t=0;t<k;++t) e[r*k + t] = digamma(a[r*k + t]) - dig_sum;
            }
        }
    }
};

// Enumerate parents with product-of-marginals weights to take expectations.
// parent_dists: for each parent j (in node.parents_idx order) a vector<double> of size radix[j]
static void enumerate_parent_expectation(
        const vector<int>& prads,
        const vector<int>& pstr,
        const vector<vector<double>>& parent_dists,
        const vector<double>& target_row_values, // length = rows*K, but we will index row* K + fixed k
        int K, int fixed_k,
        double& out_sum)
{
    // Iterative mixed-radix enumeration
    int P = (int)prads.size();
    if (P==0) {
        out_sum += target_row_values[ /*row=0*/ 0 * K + fixed_k ];
        return;
    }
    vector<int> idx(P,0);
    while (true) {
        double w = 1.0;
        int code = 0;
        for (int j=0;j<P;++j) { w *= parent_dists[j][idx[j]]; code += idx[j]*pstr[j]; }
        out_sum += w * target_row_values[ code * K + fixed_k ];

        // increment
        int j = P-1;
        while (j>=0) {
            idx[j]++; if (idx[j] < prads[j]) break;
            idx[j]=0; --j;
        }
        if (j<0) break;
    }
}

// Same as above, but returns a vector over all k (child values) at once.
static void enumerate_parent_expectation_allk(
        const vector<int>& prads,
        const vector<int>& pstr,
        const vector<vector<double>>& parent_dists,
        const vector<double>& target_row_values, // rows*K flattened
        int K,
        vector<double>& out_sum_k)
{
    int P=(int)prads.size();
    if (P==0) {
        for (int k=0;k<K;++k) out_sum_k[k] += target_row_values[k];
        return;
    }
    vector<int> idx(P,0);
    while (true) {
        double w = 1.0; int code = 0;
        for (int j=0;j<P;++j) { w *= parent_dists[j][idx[j]]; code += idx[j]*pstr[j]; }
        int base = code * K;
        for (int k=0;k<K;++k) out_sum_k[k] += w * target_row_values[base + k];

        int j=P-1;
        while (j>=0) { idx[j]++; if (idx[j] < prads[j]) break; idx[j]=0; --j; }
        if (j<0) break;
    }
}

// Build parent distributions for a node for one record, possibly overriding one parent (override_idx->value).
// q_row: responsibilities for missing variables for this record (size N; empty vector => observed).
static void build_parent_dists(network& net, int node_idx, const vector<string>& row,
                               const vector<vector<double>>& q_row,
                               int override_idx, int override_val,
                               vector<vector<double>>& parent_dists)
{
    auto node = net.fast_get(node_idx);
    const auto& PIdx = node->parents_idx();
    const auto& PRad = node->parent_rad();
    parent_dists.clear(); parent_dists.reserve(PIdx.size());
    for (int p=0;p<(int)PIdx.size();++p) {
        int var = PIdx[p]; int r = PRad[p];
        vector<double> dist(r, 0.0);
        if (var == override_idx) {
            dist.assign(r, 0.0); dist[override_val]=1.0;
        } else if (row[var] != "?") {
            int a = net.fast_get(var)->vindex_fast(row[var]);
            if (a<0) { dist.assign(r, 0.0); } else { dist.assign(r, 0.0); dist[a]=1.0; }
        } else {
            // missing parent: use q_row[var]
            const auto& qv = q_row[var];
            if (!qv.empty()) {
                // ensure sizes match
                vector<double> tmp = qv; tmp.resize(r, 0.0);
                dist = std::move(tmp);
            } else {
                // fallback uniform if not initialized
                dist.assign(r, 1.0/r);
            }
        }
        parent_dists.push_back(std::move(dist));
    }
}

// ========================== Variational Bayes ==========================
//
// Mean-field VB with Dirichlet posteriors for CPT rows & categorical local factors:
//  - Global: α_{row,k} (Dirichlet) per node/parent-row/value
//  - Local: q_r(i = k) per record r and each missing var i
// Updates:
//  Elogθ precomputed from α via digamma.
//  Local q updates use Markov blanket terms expressed as expectations of Elogθ
//  over parent distributions (product-of-marginals factorization).
//
// Notes:
//  * Handles multiple "?" per row.
//  * Uses a few local CAVI sweeps per record each global iteration.
//  * Expected counts constructed by enumerating parent combinations.
//
void run_variational_bayes(network& net,
                           const vector<vector<string>>& data,
                           int vb_iters = 8,
                           int local_sweeps = 2,
                           double alpha0 = 1.0)
{
    const int N = net.netSize();
    const int D = (int)data.size();

    // --- Build VB state ---
    VBDirichlet vb;
    vb.init_from_net(net, alpha0);

    // --- Initialize α by complete-case counts + α0 ---
    // Build temporary real-valued counts via complete cases only.
    vector<vector<double>> cc_counts(N);
    for (int i=0;i<N;++i) {
        auto node = net.fast_get(i);
        int K = node->get_nvalues();
        long long prod=1; for (int r : node->parent_rad()) { if (r<=0 || prod>(LLONG_MAX/r)) throw runtime_error("Parent combos too large"); prod*=r; }
        cc_counts[i].assign((int)prod * K, 0.0);
    }
    for (const auto& row : data) {
        if ((int)row.size()!=N) continue;
        bool ok_all_parents=true;
        for (int i=0;i<N;++i) {
            auto node = net.fast_get(i);
            const string& tok = row[i];
            if (tok=="?") { ok_all_parents=false; continue; }
            int x = node->vindex_fast(tok); if (x<0) { ok_all_parents=false; continue; }
            int code = parent_code_fast(net, *node, row);
            if (code<0) { ok_all_parents=false; continue; }
            int base = node->row_base_from_assign_code(code);
            cc_counts[i][base+x] += 1.0;
        }
        (void)ok_all_parents;
    }
    for (int i=0;i<N;++i) vb.set_from_counts_plus_prior(i, cc_counts[i], alpha0);

    // --- Local variational distributions q for missing variables ---
    // For each record r, q_r is vector size N; q_r[i] empty => observed, else length card(X_i)
    vector<vector<vector<double>>> q(D, vector<vector<double>>(N));
    // Initialize q uniformly (or parent-mode if easy)
    for (int r=0;r<D;++r) {
        if ((int)data[r].size()!=N) continue;
        for (int i=0;i<N;++i) {
            if (data[r][i] == "?") {
                int K = net.fast_get(i)->get_nvalues();
                q[r][i].assign(K, 1.0/K);
            }
        }
    }

    // --- VB iterations ---
    for (int it=0; it<vb_iters; ++it) {
        // 1) Precompute Elogθ from α
        vb.update_elog();

        // 2) Local CAVI: update q_r(i) for each missing variable
        for (int r=0;r<D;++r) {
            const auto& row = data[r];
            if ((int)row.size()!=N) continue;

            // collect indices of missing vars in this row
            vector<int> miss;
            for (int i=0;i<N;++i) if (row[i] == "?") miss.push_back(i);
            if (miss.empty()) continue;

            for (int sweep=0; sweep<local_sweeps; ++sweep) {
                for (int i_idx=0; i_idx<(int)miss.size(); ++i_idx) {
                    int i = miss[i_idx];
                    auto node = net.fast_get(i);
                    int Ki = node->get_nvalues();

                    // Build parent distributions for node i for each v (we enforce i=v via override)
                    // Compute contributions:
                    //   log q_i(v) ∝ E[log P(X_i=v | Pa_i)] + Σ_c E[log P(X_c | Pa_c)]
                    vector<double> logq(Ki, 0.0);

                    // Parent expectations contribution for node i
                    {
                        // For each v, expectation over parents (excluding i) of Elogθ_i(row,v)
                        vector<vector<double>> parent_dists;
                        const auto& PRad = node->parent_rad();
                        const auto& PStr = node->parent_str();
                        for (int v=0; v<Ki; ++v) {
                            build_parent_dists(net, i, row, q[r], /*override*/ -1, -1, parent_dists);
                            double sum = 0.0;
                            enumerate_parent_expectation(
                                PRad, PStr, parent_dists, vb.elog[i], Ki, /*fixed_k=*/v, sum
                            );
                            logq[v] += sum;
                        }
                    }

                    // Children contributions
                    for (int ci : node->get_children()) {
                        auto child = net.fast_get(ci);
                        int Kc = child->get_nvalues();

                        // Prepare child's parent structures
                        const auto& PRadC = child->parent_rad();
                        const auto& PStrC = child->parent_str();

                        if (data[r][ci] != "?") {
                            // observed child value
                            int y = child->vindex_fast(data[r][ci]);
                            if (y >= 0) {
                                for (int v=0; v<Ki; ++v) {
                                    vector<vector<double>> parent_dists;
                                    build_parent_dists(net, ci, row, q[r], /*override_idx=*/i, /*override_val=*/v, parent_dists);
                                    double sum = 0.0;
                                    enumerate_parent_expectation(
                                        PRadC, PStrC, parent_dists, vb.elog[ci], Kc, /*fixed_k=*/y, sum
                                    );
                                    logq[v] += sum;
                                }
                            }
                        } else {
                            // child missing: expectation over child q
                            const auto& qc = q[r][ci]; // size Kc
                            if (!qc.empty()) {
                                for (int v=0; v<Ki; ++v) {
                                    vector<vector<double>> parent_dists;
                                    build_parent_dists(net, ci, row, q[r], /*override_idx=*/i, /*override_val=*/v, parent_dists);
                                    vector<double> sum_k(Kc, 0.0);
                                    enumerate_parent_expectation_allk(
                                        PRadC, PStrC, parent_dists, vb.elog[ci], Kc, sum_k
                                    );
                                    double contrib = 0.0;
                                    for (int y=0;y<Kc;++y) contrib += qc[y] * sum_k[y];
                                    logq[v] += contrib;
                                }
                            }
                        }
                    }

                    // Normalize logq -> q
                    double lse = logsumexp(logq);
                    for (int v=0; v<Ki; ++v) q[r][i][v] = std::exp(logq[v] - lse);
                }
            }
        }

        // 3) Global expected counts under current q: update α = α0 + E[counts]
        for (int i=0;i<N;++i) {
            auto node = net.fast_get(i);
            int Ki = node->get_nvalues();
            long long prod=1; for (int r : node->parent_rad()) prod *= r;
            vector<double> exp_counts((int)prod * Ki, 0.0);

            // For each record
            for (int r=0;r<D;++r) {
                const auto& row = data[r]; if ((int)row.size()!=N) continue;

                // Distribution over child value
                vector<double> child_dist(Ki, 0.0);
                if (row[i] != "?") {
                    int x = node->vindex_fast(row[i]);
                    if (x<0) continue;
                    child_dist.assign(Ki, 0.0); child_dist[x] = 1.0;
                } else {
                    const auto& qi = q[r][i];
                    if (qi.empty()) { child_dist.assign(Ki, 1.0/Ki); }
                    else { child_dist = qi; }
                }

                // Parent distributions (product-of-marginals)
                vector<vector<double>> parent_dists;
                build_parent_dists(net, i, row, q[r], /*override*/ -1, -1, parent_dists);

                // Enumerate parent combinations and distribute child mass to counts
                const auto& PRad = node->parent_rad();
                const auto& PStr = node->parent_str();
                int P = (int)PRad.size();
                if (P==0) {
                    int base = 0;
                    for (int k=0;k<Ki;++k) exp_counts[base + k] += child_dist[k];
                } else {
                    vector<int> idx(P,0);
                    while (true) {
                        double w = 1.0; int code = 0;
                        for (int j=0;j<P;++j) { w *= parent_dists[j][idx[j]]; code += idx[j]*PStr[j]; }
                        int base = code * Ki;
                        for (int k=0;k<Ki;++k) exp_counts[base+k] += w * child_dist[k];

                        int j=P-1;
                        while (j>=0) { idx[j]++; if (idx[j] < PRad[j]) break; idx[j]=0; --j; }
                        if (j<0) break;
                    }
                }
            }

            // Set α_i ← α0 + E[counts]
            vb.set_from_counts_plus_prior(i, exp_counts, alpha0);
        }

        // 4) Update CPTs to posterior means for export/use: E[θ] = α / sum α per row
        for (int i=0;i<N;++i) {
            auto node = net.fast_get(i);
            int Ki = node->get_nvalues();
            int R  = vb.rows[i];
            vector<float> cpt(R*Ki, 0.0f);
            for (int rrow=0;rrow<R;++rrow) {
                double sum = 0.0; for (int k=0;k<Ki;++k) sum += vb.alpha[i][rrow*Ki+k];
                if (sum <= 0.0) {
                    for (int k=0;k<Ki;++k) cpt[rrow*Ki+k] = 1.0f / Ki;
                } else {
                    for (int k=0;k<Ki;++k) cpt[rrow*Ki+k] = (float)(vb.alpha[i][rrow*Ki+k] / sum);
                }
            }
            node->set_CPT(cpt);
        }

        cout << "VB iteration " << (it+1) << " complete." << endl;
    }
}

// ---------- Data loading & validation ----------
vector<vector<string>> read_data(const string& filename) {
    vector<vector<string>> data;
    ifstream infile(filename);
    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        vector<string> row; string val; while (ss >> val) row.push_back(val);
        if (!row.empty()) data.push_back(std::move(row));
    }
    return data;
}
static inline void rtrim_cr(std::string& s) { if (!s.empty() && s.back()=='\r') s.pop_back(); }
static inline std::string strip_quotes(const std::string& s) {
    if (s.size()>=2 && s.front()=='"' && s.back()=='"') return s.substr(1, s.size()-2);
    return s;
}
static std::vector<std::string> parse_csv_line(const std::string& line_in) {
    std::vector<std::string> out; std::string field; bool in_quotes=false;
    for (size_t i=0;i<line_in.size();++i) {
        char c = line_in[i];
        if (c=='"') { in_quotes=!in_quotes; field.push_back(c); }
        else if (c==',' && !in_quotes) { out.push_back(strip_quotes(field)); field.clear(); }
        else { field.push_back(c); }
    }
    out.push_back(strip_quotes(field));
    return out;
}
std::vector<std::vector<std::string>> load_records_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Could not open " << path << "\n"; return {}; }
    std::vector<std::vector<std::string>> data;
    std::string line;
    while (std::getline(in, line)) {
        rtrim_cr(line); if (line.empty()) continue;
        auto row = parse_csv_line(line);
        data.push_back(std::move(row));
    }
    std::cerr << "Loaded " << data.size() << " records.\n";
    return data;
}
void validate_dataset(network& net, const vector<vector<string>>& data, int max_report=10) {
    int N = net.netSize();
    int bad_rows=0, bad_tokens=0, missing_multi=0;
    for (int r=0;r<(int)data.size();++r) {
        const auto& row = data[r];
        if ((int)row.size()!=N) { if (++bad_rows<=max_report) cerr << "[ROW LEN] r="<<r<<" has "<<row.size()<<" cols; expected "<<N<<"\n"; continue; }
        int q=0;
        for (int i=0;i<N;++i) {
            const auto node = net.get_nth_node(i);
            const auto& vals = node->get_values();
            const string& tok = row[i];
            if (tok=="?") { ++q; continue; }
            bool ok=false; for (const auto& v : vals) if (v==tok) { ok=true; break; }
            if (!ok) { if (++bad_tokens<=max_report) cerr << "[TOKEN] r="<<r<<" col="<<i<<" node="<<node->get_name()<<" value='"<<tok<<"' not in domain\n"; }
        }
        if (q>1) { ++missing_multi; if (missing_multi<=max_report) cerr << "[MULTI ?] r="<<r<<" has "<<q<<" question marks\n"; }
    }
    cerr << "Validation summary: bad_rows="<<bad_rows<<" bad_tokens="<<bad_tokens<<" multi_missing="<<missing_multi<<"\n";
}

#ifndef BN_LIB
int main() {
    network BayesNet = read_network("hailfinder.bif");
    vector<vector<string>> dataset = load_records_csv("records.dat");

    cout << "Loaded " << dataset.size() << " records." << endl;
    validate_dataset(BayesNet, dataset);

    // Strong starting point before VB
    initialize_cpts_complete_case(BayesNet, dataset);

    // === Variational Bayes (mean-field) ===
    run_variational_bayes(BayesNet, dataset,
                          /*vb_iters=*/8,
                          /*local_sweeps=*/2,
                          /*alpha0=*/1.0);

    // If you want to try alternatives:
    // run_soft_em(BayesNet, dataset, /*iters=*/7);                         // (kept earlier in your project)
    // run_multiple_imputation_em(BayesNet, dataset, 6, 5, 1, 1, 1e-3f);    // (kept earlier in your project)

    write_network("solved.bif", BayesNet);
    return 0;
}
#endif // BN_LIB
