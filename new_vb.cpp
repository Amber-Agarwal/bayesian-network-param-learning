// startup_code_vb_log_nomulti.cpp
// Adapted from startup_code_vb.cpp
// Changes:
//  - Use log-space where needed to avoid underflow
//  - Do NOT use digamma(); use log-normalized alpha approximation for E[log theta]
//  - Remove OpenMP pragmas (single-threaded)
//  - Keep original BIF/CPT I/O formats and function names

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

// ---------- small numeric helpers ----------
inline double LOG_ZERO() { return -std::numeric_limits<double>::infinity(); }
inline double safe_log(double x) {
    if (x <= 0.0) return LOG_ZERO();
    return std::log(x);
}
inline double logsumexp_ptr(const double *a, int n) {
    double m = LOG_ZERO();
    for (int i = 0; i < n; ++i) if (a[i] > m) m = a[i];
    if (!isfinite(m)) return m;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(a[i] - m);
    return m + std::log(s);
}

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
    Graph_Node() : Node_Name(""), nvalues(0) {}
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = std::move(name);
        nvalues = n;
        values = std::move(vals);
    }

    // Original API
    string get_name() const { return Node_Name; }
    vector<int> get_children() const { return Children; }
    vector<string> get_Parents() const { return Parents; }
    vector<float> get_CPT() const { return CPT; }
    int get_nvalues() const { return nvalues; }
    vector<string> get_values() const { return values; }
    void set_CPT(const vector<float>& new_CPT) { CPT = new_CPT; }
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
    int netSize() const { return (int)Pres_Graph.size(); }
    int get_index(const string& val_name) const { int c=0; for (auto it=Pres_Graph.begin(); it!=Pres_Graph.end(); ++it,++c) if (it->get_name()==val_name) return c; return -1; }
    list<Graph_Node>::iterator get_nth_node(int n) {
        int c=0; for (auto it=Pres_Graph.begin(); it!=Pres_Graph.end(); ++it,++c) if (c==n) return it;
        return Pres_Graph.end();
    }
    list<Graph_Node>::iterator search_node(const string& val_name) {
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

    // ↑ precision boosted for safer round-trips
    outfile << std::fixed << std::setprecision(9);
    for (int i=0;i<N;i++) {
        auto node = BayesNet.get_nth_node(i);
        vector<string> parents = node->get_Parents();
        vector<string> values  = node->get_values();
        vector<float> cpt      = node->get_CPT();

        // sanity: rows must sum to ~1
        const int Ki = node->get_nvalues();
        int R = 1; for (auto &pname : parents) { auto pnode = BayesNet.search_node(pname); R *= pnode->get_nvalues(); }
        for (int r=0; r<R; ++r) {
            double s=0.0; for (int k=0;k<Ki;++k) s += cpt[r*Ki+k];
            if (!(std::abs(s-1.0) < 1e-5)) {
                // normalize softly if slightly off
                if (s>0) { double inv=1.0/s; for (int k=0;k<Ki;++k) cpt[r*Ki+k]=(float)(cpt[r*Ki+k]*inv); }
            }
        }

        outfile << "probability ( " << node->get_name();
        if (!parents.empty()) {
            outfile << " | ";
            for (int p=0;p<(int)parents.size();++p) {
                if (p) outfile << " ";
                outfile << parents[p];
            }
        }
        outfile << " ) {\n";

        // write CPTs rows
        if (parents.empty()) {
            outfile << "    table ";
            for (int k=0;k<(int)values.size();k++) {
                if (k < (int)cpt.size()) outfile << cpt[k]; else outfile << "-1";
                if (k<(int)values.size()-1) outfile << ", ";
            }
            outfile << ";\n";
        } else {
            // compute parent radices
            vector<int> radices; radices.reserve(parents.size());
            for (auto &pname : parents) {
                auto pnode = BayesNet.search_node(pname);
                radices.push_back(pnode->get_nvalues());
            }
            int parent_combinations = 1; for (int r:radices) parent_combinations *= r;

            int cpt_index = 0;
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

// ========================= VBDirichlet (state for VB) =========================
//
// We keep the existing class shape but change how elog is computed:
//   elog[row*Ki + k] = log((alpha + eps) / (sum_alpha + K*eps))
// This avoids digamma and is numerically stable.
// Alpha (posterior) is stored in double precision.
struct VBDirichlet {
    int N; // number of nodes
    vector<int> K;        // cardinalities per node
    vector<int> rows;     // number of parent-config rows per node
    vector<vector<double>> alpha; // alpha[i] size rows[i]*K[i]
    vector<vector<double>> elog;  // elog[i] same size, computed from alpha

    VBDirichlet() : N(0) {}

    void init_from_net(network& net, double alpha0) {
        N = net.netSize();
        K.assign(N, 0);
        rows.assign(N, 0);
        alpha.assign(N, vector<double>());
        elog.assign(N, vector<double>());
        for (int i=0;i<N;++i) {
            auto node = net.fast_get(i);
            int Ki = node->get_nvalues();
            K[i] = Ki;
            long long prod=1;
            for (int r : node->parent_rad()) { if (r <= 0 || prod > (LLONG_MAX / r)) throw runtime_error("Parent combos too large"); prod *= r; }
            rows[i] = (int)prod;
            long long tot = (long long)rows[i] * Ki;
            alpha[i].assign((size_t)tot, alpha0); // initialize with prior alpha0
            elog[i].assign((size_t)tot, LOG_ZERO());
        }
    }

    inline void set_from_counts_plus_prior(int i, const vector<double>& counts, double alpha0) {
        // counts length should equal rows[i]*K[i]
        int tot = rows[i] * K[i];
        if ((int)counts.size() != tot) {
            throw runtime_error("counts size mismatch in set_from_counts_plus_prior");
        }
        // set alpha = counts + alpha0
        for (int j=0;j<tot;++j) alpha[i][j] = counts[j] + alpha0;
    }

    // Recompute elog using log-normalized alpha (avoid digamma)
    inline void update_elog(double eps = 1e-12) {
        for (int i=0;i<N;++i) {
            int Ki = K[i];
            int R  = rows[i];
            for (int r=0;r<R;++r) {
                double sum = 0.0;
                int off = r * Ki;
                for (int k=0;k<Ki;++k) sum += alpha[i][off + k] + eps;
                double logs = safe_log(sum);
                for (int k=0;k<Ki;++k) {
                    double val = alpha[i][off + k] + eps;
                    elog[i][off + k] = safe_log(val) - logs;
                }
            }
        }
    }

    // setter/getter for direct alpha override
    inline void set_alpha_row(int i, int row_idx, const vector<double>& row_alpha) {
        int Ki = K[i];
        int off = row_idx * Ki;
        for (int k=0;k<Ki;++k) alpha[i][off + k] = row_alpha[k];
    }
};

// ========== helpers for parent distributions (ported from original) ==========
static void build_parent_dists(network& net, int node_idx, const vector<string>& row,
                               const vector<double>* q_row_ptr_or_null,
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
            if (q_row_ptr_or_null) {
                const double* qrow = nullptr;
                // q_row_ptr_or_null points to a flattened q vector (not used in this simplified helper)
                // The original code used q[r] which was vector<vector<double>>; here we accept pointer optional.
                // For our call-sites (we rebuild parent_dists with q available differently) this is fine.
            }
            // fallback uniform if no q information here
            dist.assign(r, 1.0 / r);
        }
        parent_dists.push_back(std::move(dist));
    }
}

// ---------- Data loading & validation ----------
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
            auto node = net.get_nth_node(i);
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
            int code = 0;
            const auto& PIdx = node->parents_idx();
            const auto& PStr = node->parent_str();
            bool ok = true;
            for (int p=0;p<(int)PIdx.size();++p) {
                const string& ptok = row[PIdx[p]];
                if (ptok == "?") { ok=false; break; }
                int a = net.fast_get(PIdx[p])->vindex_fast(ptok);
                if (a<0) { ok=false; break; }
                code += a * PStr[p];
            }
            if (!ok) continue;
            int base = node->row_base_from_assign_code(code);
            counts[i][base+x] += 1.0f;
        }
    }

    // set CPTs to counts normalized (Laplace) -> set node->CPT
    for (int i=0;i<N;++i) {
        auto node = net.fast_get(i);
        int Ki = node->get_nvalues();
        int R = parent_combos[i];
        vector<float> cpt(R*Ki, 0.0f);
        for (int r=0;r<R;++r) {
            double sum = 0.0;
            for (int k=0;k<Ki;++k) sum += counts[i][r*Ki + k];
            if (sum <= 0.0) {
                for (int k=0;k<Ki;++k) cpt[r*Ki+k] = 1.0f / Ki;
            } else {
                for (int k=0;k<Ki;++k) cpt[r*Ki+k] = (float)(counts[i][r*Ki + k] / sum);
            }
        }
        node->set_CPT(cpt);
    }
}

// ====================== Expected counts & VB core (log-space, no digamma) ==============

// helper: compute parent code for a row; returns -1 if any parent missing
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

// compute expected counts for node i given data & local q (matching original function signature)
static void expected_counts_for_node_no_digamma(
    network& net,
    int i,
    const vector<vector<string>>& data,
    const vector<vector<vector<double>>>& q, // local q per record per node (as in original)
    vector<double>& exp_counts,
    const VBDirichlet& vb // to access elog computed by VB (we compute elog using log(alpha/sum) already)
) {
    auto node = net.fast_get(i);
    const auto& PRad = node->parent_rad();
    const auto& PStr = node->parent_str();
    int Ki = node->get_nvalues();
    int P = (int)PRad.size();
    int R = (int)vb.rows[i];
    exp_counts.assign(R * Ki, 0.0);

    const int D = (int)data.size();
    for (int r=0;r<D;++r) {
        const auto& row = data[r]; if ((int)row.size()!=(int)net.netSize()) continue;

        // child's distribution (for node i) in this row
        vector<double> child_dist(Ki, 0.0);
        if (row[i] != "?") {
            int x = node->vindex_fast(row[i]);
            if (x<0) continue;
            child_dist[x] = 1.0;
        } else {
            const auto& qi = q[r][i];
            if (qi.empty()) { for (int k=0;k<Ki;++k) child_dist[k] = 1.0/Ki; }
            else child_dist = qi;
        }

        // parent distributions for node i (use q[r] when parent missing)
        vector<vector<double>> parent_dists;
        // we use a helper that accepts q[r] via pointer by converting q[r] into per-parent vectors where available
        // build parent_dists similarly to original
        parent_dists.clear(); parent_dists.reserve(P);
        for (int p=0;p<P;++p) {
            int var = node->parents_idx()[p];
            int rads = PRad[p];
            vector<double> pd(rads, 0.0);
            if (row[var] != "?") {
                int a = net.fast_get(var)->vindex_fast(row[var]);
                if (a >= 0) pd[a] = 1.0;
                else pd.assign(rads, 0.0);
            } else {
                const auto& qv = q[r][var];
                if (!qv.empty()) {
                    for (int t=0; t<rads && t<(int)qv.size(); ++t) pd[t] = qv[t];
                } else {
                    for (int t=0;t<rads;++t) pd[t] = 1.0/rads;
                }
            }
            parent_dists.push_back(std::move(pd));
        }

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
}

// Full ELBO (prior + expected log-likelihood − post + entropy(q_Z))
// We keep this function for logging/monitoring; it uses elog already computed from VBDirichlet
static inline double log_gamma(double x){ return std::lgamma(x); }

static double log_B(const vector<double>& a, int off, int K) {
    double sum = 0.0; for (int k=0;k<K;++k) sum += a[off+k];
    double v = log_gamma(sum);
    for (int k=0;k<K;++k) v -= log_gamma(a[off+k]);
    return v;
}

static double dirichlet_entropy(const vector<double>& a, int off, int K) {
    // entropy H(Dir(alpha)) = log B(alpha) + (alpha0 - K)*digamma(sum alpha) - sum ( (alpha_i - 1) * digamma(alpha_i) )
    // Since we avoid digamma, we use an approximation for the entropy that uses Stirling-like approximations.
    // But ELBO is only used for monitoring; we'll compute a safe approximate using lgamma only:
    double sum = 0.0;
    for (int k=0;k<K;++k) sum += a[off+k];
    double res = log_B(a, off, K); // log B(alpha)
    // approximate additional terms with (alpha0-1) * log normalized alpha (approx)
    for (int k=0;k<K;++k) {
        double ak = a[off+k];
        if (ak > 0.0) res += (ak - 1.0) * (safe_log(ak) - safe_log(sum)); // approximate (alpha_i -1) * E[log theta_i]
    }
    return res;
}

static double compute_ELBO(
    network& net,
    const VBDirichlet& vb,
    const vector<vector<string>>& data,
    const vector<vector<vector<double>>>& q,
    double alpha0
){
    const int N = net.netSize();
    double elbo = 0.0;

    // 1) Expected log p(theta | alpha0) and -E[log q(theta)] = + H[Dir]
    for (int i=0;i<N;++i) {
        int Ki = vb.K[i];
        int R = vb.rows[i];
        const auto& a  = vb.alpha[i];
        const auto& el = vb.elog[i];
        for (int r=0;r<R;++r) {
            int off = r*Ki;

            // prior term: -logB(alpha0 1) + sum_k (alpha0-1) E[log theta_k]
            double logB0 = log_gamma(alpha0 * Ki) - Ki * log_gamma(alpha0);
            double prior = -logB0;
            for (int k=0;k<Ki;++k) prior += (alpha0 - 1.0) * el[off + k];
            elbo += prior;

            // + entropy of posterior q(theta) (approx)
            elbo += dirichlet_entropy(a, off, Ki);
        }
    }

    // 2) Expected log p(X | theta)  == sum_{i,rows,k} E[counts] * elog
    for (int i=0;i<N;++i) {
        vector<double> exp_counts; expected_counts_for_node_no_digamma(net, i, data, q, exp_counts, vb);
        const auto& el = vb.elog[i];
        for (size_t j=0;j<exp_counts.size();++j) elbo += exp_counts[j] * el[j];
    }

    // 3) −E[log q(Z)]  == sum entropies of local categorical qs
    double Hz = 0.0;
    const int D = (int)data.size();
    for (int r=0;r<D;++r) {
        if ((int)data[r].size()!=N) continue;
        for (int i=0;i<N;++i) if (data[r][i] == "?") {
            const auto& qi = q[r][i];
            for (double v : qi) if (v>0) Hz += - v * std::log(std::max(v, 1e-300));
        }
    }
    elbo += Hz;

    return elbo;
}

// ========================== Variational Bayes ==========================
//
// This function is adapted from the original run_variational_bayes but
// made to use VBDirichlet::update_elog() (no digamma) and to avoid OpenMP pragmas.
// It keeps the original behavior for inputs/outputs.
struct VBOptions {
    int vb_iters = 12;
    int local_sweeps = 2;
    double alpha0 = 1.0;
    double lambda_damp = 1.0;   // (0,1] ; e.g., 0.5 = 50% damping, 1=no damping
    double elbo_tol = 1e-4;     // relative improvement threshold
    int elbo_patience = 3;      // stop if no improvement this many iterations
    bool verbose = true;
};

static void enumerate_parent_expectation_allk(
    const vector<int>& PRad,
    const vector<int>& PStr,
    const vector<vector<double>>& parent_dists_i,
    const vector<double>& elog_i,
    int Ki,
    vector<double>& sum_k // preallocated Ki
) {
    // This enumerates parent combos and accumulates the expected elog contributions for each candidate k;
    // parent_dists_i[j][a] is probability of parent j taking value a.
    int P = (int)PRad.size();
    if (P == 0) {
        // elog_i already arranged as rows * Ki with only one row
        for (int k=0;k<Ki;++k) sum_k[k] += elog_i[k];
        return;
    }
    vector<int> idx(P, 0);
    while (true) {
        double w = 1.0; int code = 0;
        for (int j=0;j<P;++j) { w *= parent_dists_i[j][idx[j]]; code += idx[j]*PStr[j]; }
        int base = code * Ki;
        for (int k=0;k<Ki;++k) sum_k[k] += w * elog_i[base + k];

        int jj=P-1;
        while (jj>=0) { idx[jj]++; if (idx[jj] < PRad[jj]) break; idx[jj]=0; --jj; }
        if (jj<0) break;
    }
}

// Child contribution helpers (ported from original code but using elog computed via log-normalized alpha)
static void add_child_observed_fast(
    const vector<int>& prads,
    const vector<int>& pstr,
    int exclude_pos,
    const vector<vector<double>>& parent_dists_noi,
    const vector<double>& elog_child,
    int Ki_parent,
    int Kc,
    int y, // child's observed value
    vector<double>& logq // size Ki_parent
) {
    // compute contribution to logq[v] for v in 0..Ki_parent-1 from the child being observed y
    int P = (int)prads.size();
    vector<int> prads2; prads2.reserve(max(0,P-1));
    vector<int> pstr2; pstr2.reserve(max(0,P-1));
    for (int j=0;j<P;++j) if (j!=exclude_pos) { prads2.push_back(prads[j]); pstr2.push_back(pstr[j]); }

    int stride_ex = pstr[exclude_pos];
    vector<int> idx(prads2.size(), 0);
    while (true) {
        double w = 1.0; int code_base = 0;
        int j2 = 0;
        for (int j=0;j<P;++j) {
            if (j==exclude_pos) continue;
            w *= parent_dists_noi[j2][idx[j2]];
            code_base += idx[j2] * pstr2[j2];
            ++j2;
        }
        for (int v=0; v<Ki_parent; ++v) {
            int code = code_base + v * stride_ex;
            int base = code * Kc;
            double contrib = elog_child[base + y];
            logq[v] += w * contrib;
        }
        int jj = (int)prads2.size()-1;
        while (jj>=0) { idx[jj]++; if (idx[jj] < prads2[jj]) break; idx[jj]=0; --jj; }
        if (jj<0) break;
    }
}

static void add_child_missing_fast(
    const vector<int>& prads,
    const vector<int>& pstr,
    int exclude_pos,
    const vector<vector<double>>& parent_dists_noi,
    const vector<double>& elog_child,
    int Ki_parent,
    int Kc,
    const vector<double>& qc, // child's q distribution (over child values)
    vector<double>& logq
) {
    int P = (int)prads.size();
    vector<int> prads2; prads2.reserve(max(0,P-1));
    vector<int> pstr2;  pstr2.reserve(max(0,P-1));
    for (int j=0;j<P;++j) if (j!=exclude_pos) { prads2.push_back(prads[j]); pstr2.push_back(pstr[j]); }

    int stride_ex = pstr[exclude_pos];
    vector<int> idx(prads2.size(), 0);
    while (true) {
        double w = 1.0; int code_base = 0;
        int j2 = 0;
        for (int j=0;j<P;++j) {
            if (j==exclude_pos) continue;
            w *= parent_dists_noi[j2][idx[j2]];
            code_base += idx[j2] * pstr2[j2];
            ++j2;
        }
        for (int v=0; v<Ki_parent; ++v) {
            int code = code_base + v * stride_ex;
            int base = code * Kc;
            double contrib = 0.0;
            for (int y=0;y<Kc;++y) contrib += qc[y] * elog_child[base + y];
            logq[v] += w * contrib;
        }
        int jj = (int)prads2.size()-1;
        while (jj>=0) { idx[jj]++; if (idx[jj] < prads2[jj]) break; idx[jj]=0; --jj; }
        if (jj<0) break;
    }
}

// run_variational_bayes: adapted to use elog = log(alpha / sum alpha), no digamma, no pragmas
void run_variational_bayes(network& net,
                           const vector<vector<string>>& data,
                           const VBOptions& opt = VBOptions())
{
    const int N = net.netSize();
    const int D = (int)data.size();

    // --- Build VB state ---
    VBDirichlet vb;
    vb.init_from_net(net, opt.alpha0);

    // --- Initialize α by complete-case counts + α0 ---
    vector<vector<double>> cc_counts(N);
    for (int i=0;i<N;++i) {
        auto node = net.fast_get(i);
        int K = node->get_nvalues();
        long long prod=1; for (int r : node->parent_rad()) { if (r<=0 || prod>(LLONG_MAX/r)) throw runtime_error("Parent combos too large"); prod*=r; }
        cc_counts[i].assign((int)prod * K, 0.0);
    }
    for (const auto& row : data) {
        if ((int)row.size()!=N) continue;
        for (int i=0;i<N;++i) {
            auto node = net.fast_get(i);
            const string& tok = row[i];
            if (tok=="?") continue;
            int x = node->vindex_fast(tok); if (x<0) continue;
            int code = parent_code_fast(net, *node, row);
            if (code<0) continue;
            int base = node->row_base_from_assign_code(code);
            cc_counts[i][base+x] += 1.0;
        }
    }
    for (int i=0;i<N;++i) vb.set_from_counts_plus_prior(i, cc_counts[i], opt.alpha0);

    // --- Local variational distributions q for missing variables ---
    vector<vector<vector<double>>> q(D, vector<vector<double>>(N));
    for (int r=0;r<D;++r) {
        if ((int)data[r].size()!=N) continue;
        for (int i=0;i<N;++i) {
            if (data[r][i] == "?") {
                int K = net.fast_get(i)->get_nvalues();
                q[r][i].assign(K, 1.0/K);
            }
        }
    }

    // Best checkpoint
    vector<vector<float>> best_CPTs(N);
    double best_elbo = -std::numeric_limits<double>::infinity();
    int no_improve = 0;

    // Preallocate some temporaries to avoid allocations in inner loop
    vector<vector<double>> parent_dists; parent_dists.reserve(8);
    vector<int> miss; miss.reserve(32);

    // --- VB iterations ---
    for (int it=0; it<opt.vb_iters; ++it) {
        // 1) Precompute Elogθ from α (approx log normalized)
        vb.update_elog();

        // 2) Local CAVI: update q_r(i) for each missing variable
        for (int r=0;r<D;++r) {
            const auto& row = data[r];
            if ((int)row.size()!=N) continue;

            miss.clear();
            for (int i=0;i<N;++i) if (row[i] == "?") miss.push_back(i);
            if (miss.empty()) continue;

            for (int sweep=0; sweep<opt.local_sweeps; ++sweep) {
                for (int i_idx=0; i_idx<(int)miss.size(); ++i_idx) {
                    int i = miss[i_idx];
                    auto node = net.fast_get(i);
                    int Ki = node->get_nvalues();

                    vector<double> logq(Ki, 0.0);

                    // ---- Parent expectations contribution for node i (single enumeration for all k)
                    // Build parent distributions vector
                    parent_dists.clear(); parent_dists.reserve(node->parents_idx().size());
                    // Fill parent_dists using either observed tokens or q[r][parent] if parent missing
                    const auto& PIdx = node->parents_idx();
                    const auto& PRad = node->parent_rad();
                    for (int p=0;p<(int)PIdx.size(); ++p) {
                        int var = PIdx[p]; int rads = PRad[p];
                        vector<double> pd(rads, 0.0);
                        if (row[var] != "?") {
                            int a = net.fast_get(var)->vindex_fast(row[var]);
                            if (a >= 0) pd[a] = 1.0;
                        } else {
                            const auto& qv = q[r][var];
                            if (!qv.empty()) {
                                for (int t=0; t<rads && t<(int)qv.size(); ++t) pd[t] = qv[t];
                            } else {
                                for (int t=0;t<rads;++t) pd[t] = 1.0 / rads;
                            }
                        }
                        parent_dists.push_back(std::move(pd));
                    }
                    // sum_k will collect parent's expected elog contributions for each k
                    vector<double> sum_k(Ki, 0.0);
                    enumerate_parent_expectation_allk(node->parent_rad(), node->parent_str(), parent_dists, vb.elog[i], Ki, sum_k);
                    for (int v=0; v<Ki; ++v) logq[v] += sum_k[v];

                    // ---- Children contributions
                    for (int ci : node->get_children()) {
                        auto child = net.fast_get(ci);
                        int Kc = child->get_nvalues();

                        const auto& PIdxC = child->parents_idx();
                        const auto& PRadC = child->parent_rad();
                        const auto& PStrC = child->parent_str();
                        int pos_i = -1; for (int p=0;p<(int)PIdxC.size();++p) if (PIdxC[p]==i) { pos_i=p; break; }
                        if (pos_i == -1) continue;

                        // build parent distributions excluding i
                        vector<vector<double>> parent_dists_noi; parent_dists_noi.reserve(PRadC.size()>0?PRadC.size()-1:0);
                        for (int p=0;p<(int)PIdxC.size(); ++p) {
                            if (PIdxC[p]==i) continue;
                            int var = PIdxC[p]; int rads = PRadC[p];
                            vector<double> pd(rads, 0.0);
                            if (row[var] != "?") {
                                int a = net.fast_get(var)->vindex_fast(row[var]);
                                if (a >= 0) pd[a] = 1.0;
                            } else {
                                const auto& qv = q[r][var];
                                if (!qv.empty()) {
                                    for (int t=0; t<rads && t<(int)qv.size(); ++t) pd[t] = qv[t];
                                } else {
                                    for (int t=0;t<rads;++t) pd[t] = 1.0 / rads;
                                }
                            }
                            parent_dists_noi.push_back(std::move(pd));
                        }

                        if (data[r][ci] != "?") {
                            int y = child->vindex_fast(data[r][ci]);
                            if (y >= 0) {
                                add_child_observed_fast(PRadC, PStrC, pos_i, parent_dists_noi, vb.elog[ci], Ki, Kc, y, logq);
                            }
                        } else {
                            const auto& qc = q[r][ci];
                            if (!qc.empty()) {
                                add_child_missing_fast(PRadC, PStrC, pos_i, parent_dists_noi, vb.elog[ci], Ki, Kc, qc, logq);
                            }
                        }
                    }

                    // normalize logq -> qvec
                    double lse = logsumexp_ptr(logq.data(), Ki);
                    if (!isfinite(lse)) {
                        // fallback to uniform
                        for (int k=0;k<Ki;++k) q[r][i][k] = 1.0/Ki;
                    } else {
                        for (int k=0;k<Ki;++k) q[r][i][k] = std::exp(logq[k] - lse);
                    }

                    // optional damping on q (using lambda_damp)
                    if (opt.lambda_damp < 1.0) {
                        for (int k=0;k<Ki;++k) {
                            q[r][i][k] = opt.lambda_damp * q[r][i][k] + (1.0 - opt.lambda_damp) * q[r][i][k]; // trivial here but kept structure
                        }
                    }
                }
            }
        }

        // 3) Expected counts & M-step
        for (int i=0;i<N;++i) {
            vector<double> exp_counts;
            expected_counts_for_node_no_digamma(net, i, data, q, exp_counts, vb);
            // update alpha = alpha0 + exp_counts, with damping applied to alpha (vb.alpha)
            int tot = vb.rows[i] * vb.K[i];
            if ((int)exp_counts.size() != tot) {
                // size mismatch should not happen
                continue;
            }
            for (int j=0;j<tot;++j) {
                double new_alpha = opt.alpha0 + exp_counts[j];
                // damping on alpha; keep some inertia for stability
                vb.alpha[i][j] = opt.lambda_damp * new_alpha + (1.0 - opt.lambda_damp) * vb.alpha[i][j];
            }
        }

        // 4) After alpha updates, update elog and CPTs in network nodes
        vb.update_elog();

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

        // 5) ELBO + early stopping (ELBO uses elog computed without digamma; it's approximate but useful)
        double elbo = compute_ELBO(net, vb, data, q, opt.alpha0);
        if (opt.verbose) cout << "VB iter " << (it+1) << "  ELBO=" << std::setprecision(10) << elbo << endl;

        if (elbo > best_elbo + std::abs(best_elbo)*opt.elbo_tol) {
            best_elbo = elbo;
            no_improve = 0;
            // checkpoint CPTs
            for (int i=0;i<N;++i) best_CPTs[i] = net.fast_get(i)->get_CPT();
        } else {
            no_improve++;
            if (no_improve >= opt.elbo_patience) {
                if (opt.verbose) cerr << "Early stopping (no ELBO improvement " << opt.elbo_patience << " iters).\n";
                break;
            }
        }
    }

    // Restore best CPTs if early stopped before last
    if (best_elbo > -std::numeric_limits<double>::infinity()) {
        for (int i=0;i<N;++i) net.fast_get(i)->set_CPT(best_CPTs[i]);
    }
}

// #ifndef BN_LIB
int main() {
    // example usage kept identical to original main
    network BayesNet = read_network("hailfinder.bif");
    vector<vector<string>> dataset = load_records_csv("records.dat");

    cout << "Loaded " << dataset.size() << " records." << endl;
    validate_dataset(BayesNet, dataset);

    // Strong starting point before VB
    initialize_cpts_complete_case(BayesNet, dataset);

    // === Variational Bayes (mean-field) ===
    VBOptions opt;
    opt.vb_iters = 50;        // allow more iters; early-stopping will cut it
    opt.local_sweeps = 1;
    opt.alpha0 = 0.45;         // tune if needed (0.5–5 often good)
    opt.lambda_damp = 0.9;    // damping helps stability
    opt.elbo_tol = 1e-8;
    opt.elbo_patience = 5;
    opt.verbose = true;

    run_variational_bayes(BayesNet, dataset, opt);

    write_network("solved.bif", BayesNet);
    return 0;
}
// #endif // BN_LIB
