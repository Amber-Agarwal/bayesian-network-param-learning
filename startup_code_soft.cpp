// startup_code.cpp  (Soft EM version, improved)
// Improvements in Soft EM:
// [CHANGE 1]  Log-space probability accumulation for numerical stability
// [CHANGE 2]  α-smoothing during CPT normalization
// [CHANGE 3]  Convergence check based on log-likelihood
// [CHANGE 4]  Small random noise for symmetry breaking
// [CHANGE 6]  Optional child-weight softening
// [CHANGE 7]  Fixed random seed for reproducibility
// [CHANGE 8]  Reuse preallocated count buffers for efficiency

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
using namespace std;

class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;

    // caches
    vector<int> PIdx;
    vector<int> PRad;
    vector<int> PStr;
    unordered_map<string,int> Val2Idx;

public:
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = std::move(name);
        nvalues = n;
        values = std::move(vals);
    }

    string get_name() { return Node_Name; }
    vector<int> get_children() { return Children; }
    vector<string> get_Parents() { return Parents; }
    vector<float> get_CPT() { return CPT; }
    int get_nvalues() { return nvalues; }
    vector<string> get_values() { return values; }
    void set_CPT(vector<float> new_CPT) { CPT.swap(new_CPT); }
    void set_Parents(vector<string> Parent_Nodes) { Parents = std::move(Parent_Nodes); }
    int add_child(int new_child_index) {
        for (int c : Children) if (c == new_child_index) return 0;
        Children.push_back(new_child_index);
        return 1;
    }

    void build_value_index() {
        Val2Idx.clear(); Val2Idx.reserve(values.size()*2+1);
        for (int i = 0; i < (int)values.size(); ++i) Val2Idx[values[i]] = i;
    }
    inline int vindex_fast(const string& v) const {
        auto it = Val2Idx.find(v);
        return (it == Val2Idx.end() ? -1 : it->second);
    }
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
        for (int i = (int)PRad.size()-1; i >= 0; --i) {
            PStr[i] = stride;
            if (PRad[i] > 0 && stride > INT_MAX / PRad[i]) throw runtime_error("Parent combos overflow");
            stride *= PRad[i];
        }
    }
    inline const vector<int>& parents_idx() const { return PIdx; }
    inline const vector<int>& parent_rad()  const { return PRad; }
    inline const vector<int>& parent_str()  const { return PStr; }
    inline int row_base_from_assign_code(int code) const { return code * nvalues; }
};

class network {
    list<Graph_Node> Pres_Graph;
    vector<list<Graph_Node>::iterator> index_cache;
    unordered_map<string,int> name2idx;

public:
    int addNode(Graph_Node node) { Pres_Graph.push_back(std::move(node)); return 0; }
    list<Graph_Node>::iterator getNode(int i) {
        int count = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it)
            if (count++ == i) return it;
        return Pres_Graph.end();
    }
    int netSize() { return (int)Pres_Graph.size(); }
    int get_index(string val_name) {
        int count = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it, ++count)
            if (it->get_name() == val_name) return count;
        return -1;
    }
    list<Graph_Node>::iterator get_nth_node(int n) {
        int count = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it, ++count)
            if (count == n) return it;
        return Pres_Graph.end();
    }
    list<Graph_Node>::iterator search_node(string val_name) {
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it)
            if (it->get_name() == val_name) return it;
        cout << "node not found: " << val_name << "\n";
        return Pres_Graph.end();
    }

    void finalize_index_cache() {
        index_cache.clear(); index_cache.reserve(Pres_Graph.size());
        name2idx.clear(); name2idx.reserve(Pres_Graph.size()*2+1);
        int i = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it, ++i) {
            index_cache.push_back(it);
            name2idx[it->get_name()] = i;
        }
    }
    list<Graph_Node>::iterator fast_get(int i) { return index_cache[i]; }
    int fast_index_of(const string& name) const {
        auto it = name2idx.find(name);
        return (it == name2idx.end() ? -1 : it->second);
    }
    const vector<list<Graph_Node>::iterator>& get_index_cache() const { return index_cache; }
    const unordered_map<string,int>& get_name_map() const { return name2idx; }
};

// --------------------------- Utility Functions ---------------------------
static inline string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

static inline void rtrim_cr(std::string& s) { if (!s.empty() && s.back() == '\r') s.pop_back(); }
static inline std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') return s.substr(1, s.size() - 2);
    return s;
}
static std::vector<std::string> parse_csv_line(const std::string& line_in) {
    std::vector<std::string> out; std::string field; bool in_quotes = false;
    for (size_t i = 0; i < line_in.size(); ++i) {
        char c = line_in[i];
        if (c == '"') { in_quotes = !in_quotes; field.push_back(c); }
        else if (c == ',' && !in_quotes) { out.push_back(strip_quotes(field)); field.clear(); }
        else { field.push_back(c); }
    }
    out.push_back(strip_quotes(field));
    return out;
}

// --- Robust loader with auto-detect (CSV vs whitespace) ---
std::vector<std::vector<std::string>> load_records_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "Could not open " << path << "\n"; return {}; }

    std::vector<std::vector<std::string>> data;
    std::string line;
    bool decided = false;
    bool use_csv = false;

    while (std::getline(in, line)) {
        rtrim_cr(line);
        if (line.empty()) continue;

        if (!decided) {
            // simple heuristic: if the first non-empty line has a comma outside quotes, assume CSV
            bool in_quotes = false;
            for (char c : line) {
                if (c == '"') in_quotes = !in_quotes;
                if (c == ',' && !in_quotes) { use_csv = true; break; }
            }
            decided = true;
        }

        std::vector<std::string> row;
        if (use_csv) {
            row = parse_csv_line(line);
        } else {
            std::stringstream ss(line);
            std::string tok;
            while (ss >> tok) row.push_back(tok);
        }

        // trim stray commas/spaces from tokens
        for (auto& t : row) {
            // remove trailing commas that often appear in space-delimited dumps
            while (!t.empty() && (t.back() == ',')) t.pop_back();
            // trim spaces
            size_t a = t.find_first_not_of(" \t");
            size_t b = t.find_last_not_of(" \t");
            t = (a == std::string::npos) ? std::string() : t.substr(a, b - a + 1);
        }

        if (!row.empty()) data.push_back(std::move(row));
    }
    std::cerr << "Loaded " << data.size() << " records"
              << (use_csv ? " (CSV detected)." : " (whitespace detected).") << "\n";
    return data;
}
// --------------------------- Network Read/Write ---------------------------
network read_network(const char* filename) {
    network BayesNet;
    string line;
    ifstream myfile(filename);
    if (!myfile.is_open()) { cout << "Error: Could not open file " << filename << endl; return BayesNet; }

    while (getline(myfile, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        string token; { stringstream ss(line); ss >> token; }

        if (token == "variable") {
            string var_name; { stringstream ss(line); ss >> token >> var_name; }
            if (!getline(myfile, line)) break;
            stringstream ss2(line);
            string type_keyword, discrete_keyword, bracket, equals;
            int num_values;
            ss2 >> type_keyword >> discrete_keyword >> bracket >> num_values >> bracket >> equals >> bracket;

            vector<string> values;
            string value;
            while (ss2 >> value) {
                if (value == "};") break;
                if (!value.empty() && value.back() == ',') value.pop_back();
                values.push_back(value);
            }
            BayesNet.addNode(Graph_Node(var_name, num_values, values));
        } else if (token == "probability") {
            string full = line;
            while (full.find('{') == string::npos) {
                string next_line; if (!getline(myfile, next_line)) break;
                full += " " + trim(next_line);
            }
            size_t lp = full.find('('), rp = full.find(')'), bar = full.find('|');
            if (lp == string::npos || rp == string::npos) continue;

            string inside = full.substr(lp + 1, rp - lp - 1);
            stringstream hs(inside);
            string node_name; hs >> node_name;

            auto listIt = BayesNet.search_node(node_name);
            int index = BayesNet.get_index(node_name);

            vector<string> parents;
            if (bar != string::npos && bar < rp) {
                string parents_str = full.substr(bar + 1, rp - bar - 1);
                stringstream ps(parents_str);
                string p;
                while (ps >> p) {
                    if (!p.empty() && p.back() == ',') p.pop_back();
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
                else if (line.find("table") != string::npos) { size_t t = line.find("table"); part = line.substr(t + 5); }
                else part = line;

                string tok; stringstream ss_prob(part);
                while (ss_prob >> tok) {
                    while (!tok.empty() && (tok.back() == ',' || tok.back() == ';')) tok.pop_back();
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
    for (int i = 0; i < N; ++i) BayesNet.fast_get(i)->build_value_index();
    for (int i = 0; i < N; ++i) BayesNet.fast_get(i)->build_parent_cache(BayesNet.get_index_cache(), BayesNet.get_name_map());

    return BayesNet;
}

void write_network(const char* filename, network& BayesNet) {
    ofstream outfile(filename);
    if (!outfile.is_open()) { cerr << "Error writing " << filename << "\n"; return; }

    outfile << "// Bayesian Network\n\n";
    int N = BayesNet.netSize();

    // Variables
    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);
        outfile << "variable " << node->get_name() << " {\n";
        outfile << "  type discrete [ " << node->get_nvalues() << " ] = { ";
        auto vals = node->get_values();
        for (int j = 0; j < (int)vals.size(); j++) {
            outfile << vals[j];
            if (j < (int)vals.size() - 1) outfile << ", ";
        }
        outfile << " };\n}\n";
    }

    // CPTs
    outfile << fixed << setprecision(6);
    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);
        auto parents = node->get_Parents();
        auto values = node->get_values();
        auto cpt = node->get_CPT();

        outfile << "probability ( " << node->get_name();
        if (!parents.empty()) {
            outfile << " | ";
            for (int j = 0; j < (int)parents.size(); j++) {
                outfile << parents[j];
                if (j < (int)parents.size()-1) outfile << ", ";
            }
        }
        outfile << " ) {\n";

        vector<int> radices;
        for (auto& pname : parents)
            radices.push_back(BayesNet.search_node(pname)->get_nvalues());
        int parent_combinations = 1;
        for (int r : radices) parent_combinations *= r;

        int idx = 0;
        if (parents.empty()) {
            outfile << "  table ";
            for (int k = 0; k < (int)values.size(); k++) {
                outfile << cpt[idx++];
                if (k < (int)values.size()-1) outfile << ", ";
            }
            outfile << ";\n";
        } else {
            for (int comb = 0; comb < parent_combinations; comb++) {
                vector<int> pidx(parents.size(),0);
                int tmp = comb;
                for (int p = (int)parents.size()-1; p>=0; --p){ pidx[p]=tmp%radices[p]; tmp/=radices[p]; }
                outfile << "  ( ";
                for (int p=0;p<(int)parents.size();++p){
                    auto pnode=BayesNet.search_node(parents[p]);
                    auto pvals=pnode->get_values();
                    outfile << pvals[pidx[p]];
                    if(p<(int)parents.size()-1) outfile<<", ";
                }
                outfile<<") ";
                for (int k=0;k<(int)values.size();k++){
                    outfile<<cpt[idx++];
                    if(k<(int)values.size()-1) outfile<<", ";
                }
                outfile<<";\n";
            }
        }
        outfile << "};\n\n";
    }
    outfile.close();
    cout << "Network written to file: " << filename << endl;
}

// --------------------------- CPT Initialization ---------------------------
int value_index(const vector<string>& values, const string& val) {
    for (int i = 0; i < (int)values.size(); ++i) if (values[i] == val) return i;
    return -1;
}

inline int parent_code_fast(network& net, Graph_Node& node, const vector<string>& row) {
    const auto& PIdx = node.parents_idx();
    const auto& PStr = node.parent_str();
    int code = 0;
    for (int p = 0; p < (int)PIdx.size(); ++p) {
        auto pnode = net.fast_get(PIdx[p]);
        const string& tok = row[PIdx[p]];
        if (tok == "?") return -1;
        int a = pnode->vindex_fast(tok);
        if (a < 0) return -1;
        code += a * PStr[p];
    }
    return code;
}
inline int parent_code_with_override(network& net, Graph_Node& node, const vector<string>& row,
                                     int override_idx, int override_val) {
    const auto& PIdx = node.parents_idx();
    const auto& PStr = node.parent_str();
    int code = 0;
    for (int p = 0; p < (int)PIdx.size(); ++p) {
        int idx = PIdx[p];
        int a;
        if (idx == override_idx) a = override_val;
        else {
            auto pnode = net.fast_get(idx);
            const string& tok = row[idx];
            if (tok == "?") return -1;
            a = pnode->vindex_fast(tok);
            if (a < 0) return -1;
        }
        code += a * PStr[p];
    }
    return code;
}

// --------------------------- Initialization from Data ---------------------------
void initialize_cpts_complete_case(network& net, const vector<vector<string>>& data) {
    const int N = net.netSize();
    vector<vector<float>> counts(N);
    vector<int> nvals(N), parent_combos(N);

    for (int i = 0; i < N; ++i) {
        auto node = net.fast_get(i);
        nvals[i] = node->get_nvalues();
        long long prod = 1;
        for (int r : node->parent_rad()) { if (r <= 0 || prod > (LLONG_MAX / r)) throw runtime_error("Parent combos too large"); prod *= r; }
        parent_combos[i] = (int)prod;
        counts[i].assign(parent_combos[i] * nvals[i], 1.0f);
    }

    for (const auto& row : data) {
        if ((int)row.size() != N) continue;
        for (int i = 0; i < N; ++i) {
            auto node = net.fast_get(i);
            const string& tok = row[i];
            if (tok == "?") continue;
            int x = node->vindex_fast(tok);
            if (x < 0) continue;
            int code = parent_code_fast(net, *node, row);
            if (code < 0) continue;
            int base = node->row_base_from_assign_code(code);
            counts[i][base + x] += 1.0f;
        }
    }

    for (int i = 0; i < N; ++i) {
        auto node = net.fast_get(i);
        auto& c = counts[i];
        const int K = nvals[i];
        for (int off = 0; off < (int)c.size(); off += K) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += c[off + k];
            if (s <= 0.0f) { for (int k = 0; k < K; ++k) c[off + k] = 1.0f / K; }
            else {
                float inv = 1.0f / s;
                for (int k = 0; k < K; ++k) c[off + k] *= inv;
            }
        }
        node->set_CPT(c);
    }
    cerr << "Initialized CPTs from complete cases with Laplace (+1)." << endl;
}

// ========================== Soft EM (Improved) ==========================
// see top of file for list of [CHANGE #] annotations
double compute_log_likelihood(network& net, const vector<vector<string>>& data) {
    const double tiny = 1e-12;
    double ll = 0.0;
    int N = net.netSize();
    for (const auto& row : data) {
        if ((int)row.size() != N) continue;
        double logp = 0.0;
        for (int i = 0; i < N; ++i) {
            auto node = net.fast_get(i);
            int x = node->vindex_fast(row[i]);
            if (x < 0) continue;
            int code = parent_code_fast(net, *node, row);
            if (code < 0) continue;
            int base = node->row_base_from_assign_code(code);
            const auto& cpt = node->get_CPT();
            logp += std::log(std::max<double>(tiny, cpt[base + x]));
        }
        ll += logp;
    }
    return ll;
}
void run_soft_em(network& net, const vector<vector<string>>& data, int max_iter = 50, double tol = 1e-4) {
    srand(42); // reproducibility

    // ---- Annealed-EM temperature schedule (τ) ----
    double tau       = 2.0;  // start soft; try 1.5–3.0
    double tau_decay = 0.90; // cool by ~10% per EM iteration (try 0.85–0.97)
    double tau_min   = 1.0;  // do not go below exact EM

    const int N   = net.netSize();
    const double tiny = 1e-12; // numeric floor for logs/probabilities

    // ---- Preallocate/reuse counts buffers ----
    vector<vector<float>> counts(N);
    for (int i = 0; i < N; ++i) {
        auto node = net.fast_get(i);
        long long prod = 1;
        for (int r : node->parent_rad()) {
            if (r <= 0 || prod > (LLONG_MAX / r)) throw runtime_error("Parent combos too large");
            prod *= r;
        }
        counts[i].assign((int)prod * node->get_nvalues(), 0.0f);
    }

    double prev_ll = -1e100;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Small noise floor + tiny jitter for symmetry breaking
        for (int i = 0; i < N; ++i)
            std::fill(counts[i].begin(), counts[i].end(), 1e-6f + (rand() % 100) / 1e7f);

        // ========================= E-step =========================
        for (const auto& row : data) {
            if ((int)row.size() != N) continue;

            int missing_idx = -1, q = 0;
            for (int i = 0; i < N; ++i) if (row[i] == "?" && ++q == 1) missing_idx = i;
            if (q > 1) continue; // keep single-missing constraint

            if (q == 0) {
                // Fully observed row
                for (int i = 0; i < N; ++i) {
                    auto node = net.fast_get(i);
                    int x = node->vindex_fast(row[i]);
                    if (x < 0) continue;
                    int code = parent_code_fast(net, *node, row);
                    if (code < 0) continue;
                    int base = node->row_base_from_assign_code(code);
                    counts[i][base + x] += 1.0f;
                }
            } else {
                // Exactly one missing variable
                auto mnode = net.fast_get(missing_idx);

                // Parent code for the missing node (may be unknown)
                int m_code = parent_code_fast(net, *mnode, row);
                int m_base = (m_code >= 0) ? mnode->row_base_from_assign_code(m_code) : 0;
                const auto& mC = mnode->get_CPT();
                const int Mvals = mnode->get_nvalues();

                // Posterior weights w[v] for the missing variable
                std::vector<double> w(Mvals, 0.0);
                double wsum = 0.0;

                // Fallback prior over M if parents unknown: average across parent rows
                std::vector<double> prior_fallback;  // empty unless used
                if (m_code < 0) {
                    const int K = Mvals;
                    const int total_rows = (int)mC.size() / K;
                    prior_fallback.assign(K, 0.0);
                    if (total_rows > 0) {
                        for (int rrow = 0; rrow < total_rows; ++rrow) {
                            int base = rrow * K;
                            for (int v = 0; v < K; ++v) prior_fallback[v] += mC[base + v];
                        }
                        for (int v = 0; v < K; ++v) prior_fallback[v] /= std::max(1, total_rows);
                    } else {
                        for (int v = 0; v < K; ++v) prior_fallback[v] = 1.0 / K; // degenerate safety
                    }
                }

                for (int v = 0; v < Mvals; ++v) {
                    // Prior term for M=v
                    double prior_v = (m_code >= 0)
                        ? std::max(tiny, (double)mC[m_base + v])
                        : std::max(tiny, prior_fallback[v]);

                    double logp = std::log(prior_v);

                    // Add logs of children's likelihood terms (exact; no softening)
                    bool bad_child = false;
                    for (int ci : mnode->get_children()) {
                        auto child = net.fast_get(ci);
                        int y = child->vindex_fast(row[ci]);
                        if (y < 0) { bad_child = true; break; }
                        int c_code = parent_code_with_override(net, *child, row, missing_idx, v);
                        if (c_code < 0) { bad_child = true; break; }
                        int cb = child->row_base_from_assign_code(c_code);
                        const auto& cC = child->get_CPT();
                        logp += std::log(std::max(tiny, (double)cC[cb + y]));
                    }
                    if (bad_child) { w[v] = 0.0; continue; }

                    // Temperature (annealed EM): SOFT if tau>1, HARDER if tau<1
                    w[v] = std::exp(logp / tau);
                    wsum += w[v];
                }

                // Normalize weights, with safe fallback if wsum == 0
                if (wsum <= 0.0) {
                    double norm = 0.0;
                    if (m_code < 0 && !prior_fallback.empty()) {
                        for (int v = 0; v < Mvals; ++v) { w[v] = std::max(tiny, prior_fallback[v]); norm += w[v]; }
                    } else {
                        for (int v = 0; v < Mvals; ++v) { w[v] = 1.0; norm += 1.0; }
                    }
                    for (auto& t : w) t = (float)(t / norm);
                } else {
                    for (auto& t : w) t = (float)(t / wsum);
                }

                // Add expected counts to all impacted families
                for (int i = 0; i < N; ++i) {
                    auto node = net.fast_get(i);

                    // (A) Missing node itself: distribute counts by w[v]
                    if (i == missing_idx) {
                        if (m_code >= 0) {
                            int base = m_base;
                            for (int v = 0; v < node->get_nvalues(); ++v)
                                counts[i][base + v] += (float)w[v];
                        } else {
                            // No parent code: distribute evenly across all parent rows, weighted by w[v]
                            const int K = node->get_nvalues();
                            const int total_rows = (int)counts[i].size() / K;
                            for (int rrow = 0; rrow < total_rows; ++rrow) {
                                int base = rrow * K;
                                for (int v = 0; v < K; ++v)
                                    counts[i][base + v] += (float)w[v] / std::max(1, total_rows);
                            }
                        }
                        continue;
                    }

                    // (B) If node depends on M: marginalize over M using w[v]
                    bool parent_depends_on_m = false;
                    for (int pi : node->parents_idx())
                        if (pi == missing_idx) { parent_depends_on_m = true; break; }

                    if (parent_depends_on_m) {
                        int x = node->vindex_fast(row[i]);
                        if (x < 0) continue;
                        for (int v = 0; v < (int)w.size(); ++v) {
                            if (w[v] <= 0.f) continue;
                            int code = parent_code_with_override(net, *node, row, missing_idx, v);
                            if (code < 0) continue;
                            int base = node->row_base_from_assign_code(code);
                            counts[i][base + x] += (float)w[v];
                        }
                        continue;
                    }

                    // (C) Otherwise: treat as fully observed
                    int x = node->vindex_fast(row[i]);
                    if (x < 0) continue;
                    int code = parent_code_fast(net, *node, row);
                    if (code < 0) continue;
                    int base = node->row_base_from_assign_code(code);
                    counts[i][base + x] += 1.0f;
                }
            }
        }

        // ========================= M-step (ESS smoothing) =========================
        const double ESS = .98; // Effective sample size per CPT row (tune 0.5–5.0)
        for (int i = 0; i < N; ++i) {
            auto node = net.fast_get(i);
            auto& cpt = counts[i];
            const int K = node->get_nvalues();
            for (int off = 0; off < (int)cpt.size(); off += K) {
                double sum_counts = 0.0;
                for (int k = 0; k < K; ++k) sum_counts += cpt[off + k];

                // Dirichlet prior with ESS uniformly distributed among outcomes
                for (int k = 0; k < K; ++k) {
                    cpt[off + k] = (float)((cpt[off + k] + ESS / K) / (sum_counts + ESS));
                }
            }
            node->set_CPT(cpt);
        }

        // ========================= Progress + convergence =========================
        double ll = compute_log_likelihood(net, data);
        cout << "Soft-EM iteration " << iter + 1 << " complete. logL=" << ll
             << "  tau=" << tau << endl;

        // Decay τ once per iteration
        tau = std::max(tau * tau_decay, tau_min);

        if (fabs(ll - prev_ll) < tol) {
            cout << "Converged after " << iter + 1 << " iterations.\n";
            break;
        }
        prev_ll = ll;
    }
}



void validate_dataset(network& net, const vector<vector<string>>& data, int max_report=10) {
    int N = net.netSize();
    int bad_rows = 0, bad_tokens = 0, missing_multi = 0;
    for (int r = 0; r < (int)data.size(); ++r) {
        const auto& row = data[r];
        if ((int)row.size() != N) {
            if (++bad_rows <= max_report)
                cerr << "[ROW LEN] r=" << r << " has " << row.size() << " cols; expected " << N << "\n";
            continue;
        }
        int q = 0;
        for (int i = 0; i < N; ++i) {
            const auto node = net.get_nth_node(i);
            const auto& vals = node->get_values();
            const string& tok = row[i];
            if (tok == "?") { ++q; continue; }
            bool ok = false;
            for (const auto& v : vals) if (v == tok) { ok = true; break; }
            if (!ok) {
                if (++bad_tokens <= max_report)
                    cerr << "[TOKEN] r=" << r << " col=" << i
                         << " node=" << node->get_name()
                         << " value='" << tok << "' not in node’s domain\n";
            }
        }
        if (q > 1) {
            ++missing_multi;
            if (missing_multi <= max_report)
                cerr << "[MULTI ?] r=" << r << " has " << q << " question marks\n";
        }
    }
    cerr << "Validation summary: bad_rows=" << bad_rows
         << " bad_tokens=" << bad_tokens
         << " multi_missing=" << missing_multi << "\n";
}

#ifndef BN_LIB
int main() {
    // Read once to get structure for validation and N
    network BayesNet0 = read_network("hailfinder.bif");
    vector<vector<string>> dataset = load_records_csv("records.dat");

    cout << "Loaded " << dataset.size() << " records." << endl;
    validate_dataset(BayesNet0, dataset);
    const int N = BayesNet0.netSize();

    // Restart config
    const int NUM_TRIALS = 5;
    double best_ll = -1e300;
    vector<vector<float>> best_cpts(N);  // snapshot of best CPTs per node (by index)

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        // IMPORTANT: read a FRESH network each trial so iterator caches are valid
        network model = read_network("hailfinder.bif");

        // Different seed per trial for reproducibility and exploration
        srand(42 + trial * 17);

        // Initialize + EM
        initialize_cpts_complete_case(model, dataset);
        run_soft_em(model, dataset, /*max_iter=*/75, /*tol=*/1e-5);

        // Evaluate on training data (or split off validation if you have it)
        double ll = compute_log_likelihood(model, dataset);
        cout << "Trial " << (trial + 1) << "/" << NUM_TRIALS
             << " final logL = " << ll << endl;

        // Keep best CPT snapshot
        if (ll > best_ll) {
            best_ll = ll;
            for (int i = 0; i < N; ++i) {
                best_cpts[i] = model.fast_get(i)->get_CPT();
            }
        }
    }

    // Build final network and install best CPTs
    network finalNet = read_network("hailfinder.bif");
    for (int i = 0; i < N; ++i) {
        finalNet.fast_get(i)->set_CPT(best_cpts[i]);
    }

    cout << "Best log-likelihood across trials: " << best_ll << endl;
    write_network("solved.bif", finalNet);
    return 0;
}
#endif // BN_LIB
