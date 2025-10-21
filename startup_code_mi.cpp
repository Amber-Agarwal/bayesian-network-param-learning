// startup_code.cpp  — Multiple Imputation EM (MI-EM) version
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

//
// ========================= Graph_Node =========================
//

class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;

    // caches
    vector<int> PIdx;                 // parent index in network order
    vector<int> PRad;                 // parent cardinalities
    vector<int> PStr;                 // mixed-radix strides (rightmost fastest)
    unordered_map<string,int> Val2Idx; // value -> index

public:
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = std::move(name);
        nvalues = n;
        values = std::move(vals);
    }

    // original API
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

    // caches
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

//
// ============================ network ============================
//

class network {
    list<Graph_Node> Pres_Graph;

    // caches
    vector<list<Graph_Node>::iterator> index_cache;
    unordered_map<string,int> name2idx;

public:
    int addNode(Graph_Node node) { Pres_Graph.push_back(std::move(node)); return 0; }

    list<Graph_Node>::iterator getNode(int i) {
        int count = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it) {
            if (count++ == i) return it;
        }
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

    // caches
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

//
// ====================== misc helpers and I/O ======================
//

static inline string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

network read_network(const char* filename) {
    network BayesNet;
    string line;
    ifstream myfile(filename);

    if (!myfile.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return BayesNet;
    }

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
        }
        else if (token == "probability") {
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
    if (!outfile.is_open()) {
        cout << "Error: Could not open file " << filename << " for writing" << endl;
        return;
    }

    outfile << "// Bayesian Network" << endl << endl;

    int N = BayesNet.netSize();

    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);

        outfile << "variable " << node->get_name() << " {" << endl;
        outfile << "  type discrete [ " << node->get_nvalues() << " ] = { ";

        vector<string> vals = node->get_values();
        for (int j = 0; j < (int)vals.size(); j++) {
            outfile << vals[j];
            if (j < (int)vals.size() - 1) outfile << ", ";
        }
        outfile << " };" << endl;
        outfile << "}" << endl;
    }

    outfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);
        vector<string> parents = node->get_Parents();
        vector<string> values  = node->get_values();
        vector<float> cpt      = node->get_CPT();

        outfile << "probability ( " << node->get_name();
        if (!parents.empty()) {
            outfile << " | ";
            for (int j = 0; j < (int)parents.size(); j++) {
                outfile << parents[j];
                if (j < (int)parents.size() - 1) outfile << ", ";
            }
        }
        outfile << " ) {" << endl;

        vector<int> radices; radices.reserve(parents.size());
        for (auto &pname : parents) {
            auto pnode = BayesNet.search_node(pname);
            radices.push_back(pnode->get_nvalues());
        }

        int parent_combinations = 1;
        for (int r : radices) parent_combinations *= r;

        int cpt_index = 0;

        if (parents.empty()) {
            outfile << "    table ";
            for (int k = 0; k < (int)values.size(); k++) {
                if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++];
                else outfile << "-1";
                if (k < (int)values.size() - 1) outfile << ", ";
            }
            outfile << ";" << endl;
        } else {
            for (int comb = 0; comb < parent_combinations; comb++) {
                vector<int> idx(parents.size(), 0);
                int tmp = comb;
                for (int p = (int)parents.size() - 1; p >= 0; p--) {
                    idx[p] = tmp % radices[p];
                    tmp   /= radices[p];
                }

                outfile << "    ( ";
                for (int p = 0; p < (int)parents.size(); p++) {
                    auto pnode = BayesNet.search_node(parents[p]);
                    auto pvals = pnode->get_values();
                    int vidx   = idx[p];
                    outfile << pvals[vidx];
                    if (p < (int)parents.size() - 1) outfile << ", ";
                }
                outfile << " ) ";

                for (int k = 0; k < (int)values.size(); k++) {
                    if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++];
                    else outfile << "-1";
                    if (k < (int)values.size() - 1) outfile << ", ";
                }
                outfile << ";" << endl;
            }
        }

        outfile << "};" << endl << endl;
    }

    outfile.close();
    cout << "Network written to file: " << filename << endl;
}

// legacy helpers (kept)
int value_index(const vector<string>& values, const string& val) {
    for (int i = 0; i < (int)values.size(); ++i) {
        if (values[i] == val) return i;
    }
    return -1;
}
int get_cpt_index(network& net, Graph_Node& node, const vector<int>& assignment) {
    vector<string> parents = node.get_Parents();
    vector<int> radices;
    for (const string& p : parents) {
        auto pnode = net.search_node(p);
        radices.push_back(pnode->get_nvalues());
    }
    int index = 0, factor = 1;
    for (int i = (int)parents.size() - 1; i >= 0; --i) { index += assignment[i] * factor; factor *= radices[i]; }
    return index * node.get_nvalues();
}

//
// ========================= FAST helpers =========================
//

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

//
// ============ Initialization: complete cases + Laplace(+1) ============
//

void initialize_cpts_complete_case(network& net, const vector<vector<string>>& data) {
    const int N = net.netSize();

    vector<vector<float>> counts(N);
    vector<int> nvals(N), parent_combos(N);

    for (int i = 0; i < N; ++i) {
        auto node = net.fast_get(i);
        nvals[i] = node->get_nvalues();

        long long prod = 1;
        for (int r : node->parent_rad()) {
            if (r <= 0 || prod > (LLONG_MAX / r)) throw runtime_error("Parent combos too large");
            prod *= r;
        }
        parent_combos[i] = (int)prod;
        counts[i].assign(parent_combos[i] * nvals[i], 1.0f); // Laplace +1
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
            if (s <= 0.0f) {
                for (int k = 0; k < K; ++k) c[off + k] = 1.0f / K;
            } else {
                float inv = 1.0f / s;
                for (int k = 0; k < K; ++k) c[off + k] *= inv;
            }
        }
        node->set_CPT(c);
    }
    cerr << "Initialized CPTs from complete cases with Laplace (+1)." << endl;
}

//
// ============================= Soft-EM (kept) =============================
//

void run_soft_em(network& net, const vector<vector<string>>& data, int max_iter = 7) {
    const int N = net.netSize();
    const int D = (int)data.size();
    const float tiny = 1e-12f;

    for (int iter = 0; iter < max_iter; ++iter) {
        vector<vector<float>> counts(N);
        for (int i = 0; i < N; ++i) {
            auto node = net.fast_get(i);
            long long prod = 1;
            for (int r : node->parent_rad()) { if (r <= 0 || prod > (LLONG_MAX / r)) throw runtime_error("Parent combos too large"); prod *= r; }
            counts[i].assign((int)prod * node->get_nvalues(), 1e-3f);
        }

        for (int r = 0; r < D; ++r) {
            const auto& row = data[r];
            if ((int)row.size() != N) continue;

            int missing_idx = -1, q = 0;
            for (int i = 0; i < N; ++i) if (row[i] == "?" && ++q == 1) missing_idx = i;
            if (q > 1) continue;

            if (q == 0) {
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
                auto mnode = net.fast_get(missing_idx);

                int m_code = parent_code_fast(net, *mnode, row);
                if (m_code < 0) continue;
                int m_base = mnode->row_base_from_assign_code(m_code);
                const auto mC = mnode->get_CPT();

                vector<double> w(mnode->get_nvalues(), 0.0);
                double wsum = 0.0;
                for (int v = 0; v < mnode->get_nvalues(); ++v) {
                    double p = std::max<double>(tiny, mC[m_base + v]);
                    for (int ci : mnode->get_children()) {
                        auto child = net.fast_get(ci);
                        int y = child->vindex_fast(row[ci]);
                        if (y < 0) { p = 0.0; break; }
                        int c_code = parent_code_with_override(net, *child, row, missing_idx, v);
                        if (c_code < 0) { p = 0.0; break; }
                        int cb = child->row_base_from_assign_code(c_code);
                        const auto cC = child->get_CPT();
                        p *= std::max<double>(tiny, cC[cb + y]);
                        if (p == 0.0) break;
                    }
                    w[v] = p; wsum += p;
                }
                if (wsum <= 0.0) continue;
                for (auto& t : w) t = (float)(t / wsum);

                for (int i = 0; i < N; ++i) {
                    auto node = net.fast_get(i);
                    if (i == missing_idx) {
                        int base = m_base;
                        for (int v = 0; v < node->get_nvalues(); ++v) counts[i][base + v] += (float)w[v];
                        continue;
                    }
                    bool depends_on_m = false;
                    for (int pi : node->parents_idx()) if (pi == missing_idx) { depends_on_m = true; break; }

                    if (depends_on_m) {
                        int x = node->vindex_fast(row[i]);
                        if (x < 0) continue;
                        for (int v = 0; v < (int)w.size(); ++v) {
                            if (w[v] <= 0.f) continue;
                            int code = parent_code_with_override(net, *node, row, missing_idx, v);
                            if (code < 0) continue;
                            int base = node->row_base_from_assign_code(code);
                            counts[i][base + x] += (float)w[v];
                        }
                    } else {
                        int x = node->vindex_fast(row[i]);
                        if (x < 0) continue;
                        int code = parent_code_fast(net, *node, row);
                        if (code < 0) continue;
                        int base = node->row_base_from_assign_code(code);
                        counts[i][base + x] += 1.0f;
                    }
                }
            }
        }

        // M-step
        for (int i = 0; i < N; ++i) {
            auto node = net.fast_get(i);
            auto& cpt = counts[i];
            const int K = node->get_nvalues();
            for (int off = 0; off < (int)cpt.size(); off += K) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) s += cpt[off + k];
                if (s <= 0.0f) { for (int k = 0; k < K; ++k) cpt[off + k] = 1.0f / K; }
                else {
                    float inv = 1.0f / s;
                    for (int k = 0; k < K; ++k) cpt[off + k] *= inv;
                }
            }
            node->set_CPT(cpt);
        }
        cout << "Soft-EM iteration " << iter + 1 << " complete." << endl;
    }
}

//
// ======================= Multiple Imputation EM =======================
//
// We iterate:
//  1) Using current CPTs, generate M completed datasets via Gibbs sampling per row.
//     (Burn-in + few sweeps over missing variables using Markov blanket conditionals.)
//  2) Average counts across the M datasets (with small pseudo-count).
//  3) Normalize rows to update CPTs.
//  4) Repeat for 'em_iters' iterations.
//
static inline int sample_categorical(const vector<double>& probs, std::mt19937& rng) {
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// Compute unnormalized P(X=v | Markov blanket) for node i given a row that may contain other imputed values.
// This uses CPTs and current assignments in `row`.
static inline void markov_blanket_probs(network& net, int node_idx, const vector<string>& row,
                                        vector<double>& out, double tiny=1e-12) {
    auto node = net.fast_get(node_idx);
    const int V = node->get_nvalues();
    out.assign(V, 0.0);

    // Parent code for node given current row (requires all parents observed/imputed)
    int pcode = parent_code_fast(net, *node, row);
    if (pcode < 0) { // if parents still have '?', fallback to uniform to keep sampler going
        for (int v = 0; v < V; ++v) out[v] = 1.0;
        return;
    }
    int base = node->row_base_from_assign_code(pcode);
    const auto nodeCPT = node->get_CPT();

    for (int v = 0; v < V; ++v) {
        double p = std::max<double>(tiny, nodeCPT[base + v]);
        // children likelihoods
        for (int ci : node->get_children()) {
            auto child = net.fast_get(ci);
            // build child parent code with node set to v
            int ccode = parent_code_with_override(net, *child, row, node_idx, v);
            if (ccode < 0) { p = 0.0; break; }
            int cb = child->row_base_from_assign_code(ccode);
            int y = child->vindex_fast(row[ci]);
            if (y < 0) { p = 0.0; break; }
            const auto cCPT = child->get_CPT();
            p *= std::max<double>(tiny, cCPT[cb + y]);
            if (p == 0.0) break;
        }
        out[v] = p;
    }
}

// One imputation sweep for a single row: Gibbs updates every missing variable once.
static inline void gibbs_sweep_row(network& net, vector<string>& row, const vector<int>& miss_idx,
                                   std::mt19937& rng) {
    vector<double> probs;
    for (int idx : miss_idx) {
        // compute P(X=· | MB) and sample
        markov_blanket_probs(net, idx, row, probs);
        // normalize
        double s = 0.0; for (double v : probs) s += v; if (s <= 0) {
            // uniform fallback
            int V = net.fast_get(idx)->get_nvalues();
            probs.assign(V, 1.0);
            s = (double)V;
        }
        for (double& v : probs) v /= s;
        int V = net.fast_get(idx)->get_nvalues();
        int sampled = sample_categorical(probs, rng);
        sampled = std::min(std::max(sampled, 0), V-1);
        row[idx] = net.fast_get(idx)->get_values()[sampled];
    }
}

// Produce ONE completed dataset by imputing all '?' using Gibbs sampling given current CPTs.
// - burnin_sweeps: initial sweeps discarded
// - sample_sweeps: additional sweeps; we return the final state (last sweep).
static vector<vector<string>> impute_dataset_once(network& net, const vector<vector<string>>& data,
                                                  int burnin_sweeps, int sample_sweeps,
                                                  std::mt19937& rng) {
    const int N = net.netSize();
    vector<vector<string>> imputed = data;

    // Precompute missing index lists per row and cheap initialization
    for (auto& row : imputed) {
        if ((int)row.size() != N) continue;
        vector<int> miss;
        miss.reserve(N);
        for (int i = 0; i < N; ++i) if (row[i] == "?") miss.push_back(i);

        if (miss.empty()) continue;

        // Cheap init for missing: use parent CPT row mode if parents observed; else uniform random
        for (int idx : miss) {
            auto node = net.fast_get(idx);
            int code = parent_code_fast(net, *node, row);
            int V = node->get_nvalues();
            if (code >= 0) {
                int base = node->row_base_from_assign_code(code);
                const auto cpt = node->get_CPT();
                // mode
                int argmax = 0; float best = -1.0f;
                for (int v = 0; v < V; ++v) if (cpt[base+v] > best) { best=cpt[base+v]; argmax=v; }
                row[idx] = node->get_values()[argmax];
            } else {
                std::uniform_int_distribution<int> uni(0, V-1);
                row[idx] = node->get_values()[uni(rng)];
            }
        }

        // Burn-in + sample sweeps
        for (int b = 0; b < burnin_sweeps; ++b) gibbs_sweep_row(net, row, miss, rng);
        for (int s = 0; s < sample_sweeps; ++s) gibbs_sweep_row(net, row, miss, rng);
    }

    return imputed;
}

// Multiple-Imputation EM
// em_iters     : outer EM iterations
// M            : number of imputations per iteration
// burnin       : Gibbs burn-in sweeps per row
// sweeps       : Gibbs posterior sweeps per row (return last state)
// pseudo       : small count to avoid zeros
void run_multiple_imputation_em(network& net, const vector<vector<string>>& data,
                                int em_iters = 5, int M = 5,
                                int burnin = 1, int sweeps = 1,
                                float pseudo = 1e-3f) {
    if (M < 1) M = 1;
    std::random_device rd;
    std::mt19937 rng(rd());

    const int N = net.netSize();

    for (int it = 0; it < em_iters; ++it) {
        // aggregate counts (start with pseudo-counts)
        vector<vector<float>> agg_counts(N);
        for (int i = 0; i < N; ++i) {
            auto node = net.fast_get(i);
            long long prod = 1;
            for (int r : node->parent_rad()) { if (r <= 0 || prod > (LLONG_MAX / r)) throw runtime_error("Parent combos too large"); prod *= r; }
            agg_counts[i].assign((int)prod * node->get_nvalues(), pseudo);
        }

        // Generate M completed datasets and accumulate counts
        for (int m = 0; m < M; ++m) {
            // Impute one dataset using current CPTs
            auto completed = impute_dataset_once(net, data, burnin, sweeps, rng);

            // Add complete-data counts
            for (const auto& row : completed) {
                if ((int)row.size() != N) continue;
                for (int i = 0; i < N; ++i) {
                    auto node = net.fast_get(i);
                    int x = node->vindex_fast(row[i]);
                    if (x < 0) continue;
                    int code = parent_code_fast(net, *node, row);
                    if (code < 0) continue; // shouldn’t happen after imputation
                    int base = node->row_base_from_assign_code(code);
                    agg_counts[i][base + x] += 1.0f;
                }
            }
        }

        // M-step: normalize per-row to update CPTs (equally weighted across imputations)
        for (int i = 0; i < N; ++i) {
            auto node = net.fast_get(i);
            auto& cpt = agg_counts[i];
            const int K = node->get_nvalues();
            for (int off = 0; off < (int)cpt.size(); off += K) {
                float s = 0.0f; for (int k = 0; k < K; ++k) s += cpt[off + k];
                if (s <= 0.0f) { for (int k = 0; k < K; ++k) cpt[off + k] = 1.0f / K; }
                else {
                    float inv = 1.0f / s;
                    for (int k = 0; k < K; ++k) cpt[off + k] *= inv;
                }
            }
            node->set_CPT(cpt);
        }
        cout << "MI-EM iteration " << (it+1) << " complete (M=" << M << ")." << endl;
    }
}

//
// ======================= Data loading & validation =======================
//

vector<vector<string>> read_data(const string& filename) {
    vector<vector<string>> data;
    ifstream infile(filename);
    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        vector<string> row;
        string val;
        while (ss >> val) row.push_back(val);
        if (!row.empty()) data.push_back(std::move(row));
    }
    return data;
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
    network BayesNet = read_network("hailfinder.bif");
    vector<vector<string>> dataset = load_records_csv("records.dat");

    cout << "Loaded " << dataset.size() << " records." << endl;
    validate_dataset(BayesNet, dataset);

    // Good starting point:
    initialize_cpts_complete_case(BayesNet, dataset);

    // ---- Choose ONE training strategy ----
    // 1) Soft EM:
    // run_soft_em(BayesNet, dataset, /*iters=*/7);

    // 2) Multiple-Imputation EM (recommended when rows have multiple '?'):
    run_multiple_imputation_em(BayesNet, dataset,
                               /*em_iters=*/6,
                               /*M=*/5,
                               /*burnin=*/1,
                               /*sweeps=*/1,
                               /*pseudo=*/1e-3f);

    write_network("solved.bif", BayesNet);
    return 0;
}
#endif // BN_LIB
