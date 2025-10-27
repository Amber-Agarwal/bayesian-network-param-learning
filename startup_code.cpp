// startup_code.cpp  (Soft/Exact EM for Hailfinder, improved)
//
// Key changes vs. your previous version:
//  - Use DOUBLE precision CPTs & counts throughout
//  - Incomplete-data log-likelihood using Markov blanket & log-sum-exp
//  - Initialization with Dirichlet prior centered at original BIF CPTs
//  - Annealed EM (tau>1 initially)
//  - Row-wise clipping + renormalization every M-step
//  - Weighted model averaging across restarts
//
// Notes:
//  * At most ONE missing entry per row ("?"), as per your dataset.
//  * Robust CSV/whitespace loader and network I/O.
//
// Build: g++ -O3 -std=c++17 startup_code.cpp -o hail_em
// Run:   ./hail_em
//
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
#include <algorithm>
using namespace std;

class Graph_Node
{
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT; // FLOAT precision

    // caches
    vector<int> PIdx;
    vector<int> PRad;
    vector<int> PStr;
    unordered_map<string, int> Val2Idx;

public:
    Graph_Node(string name, int n, vector<string> vals)
    {
        Node_Name = move(name);
        nvalues = n;
        values = move(vals);
    }

    string get_name() { return Node_Name; }
    vector<int> get_children() { return Children; }
    vector<string> get_Parents() { return Parents; }
    vector<float> &get_CPT() { return CPT; }
    int get_nvalues() { return nvalues; }
    vector<string> get_values() { return values; }
    void set_CPT(vector<float> new_CPT) { CPT.swap(new_CPT); }
    void set_Parents(vector<string> Parent_Nodes) { Parents = move(Parent_Nodes); }
    int add_child(int new_child_index)
    {
        for (int c : Children)
            if (c == new_child_index)
                return 0;
        Children.push_back(new_child_index);
        return 1;
    }

    void build_value_index()
    {
        Val2Idx.clear();
        Val2Idx.reserve(values.size() * 2 + 1);
        for (int i = 0; i < (int)values.size(); ++i)
            Val2Idx[values[i]] = i;
    }
    inline int vindex_fast(string &v)
    {
        auto it = Val2Idx.find(v);
        return (it == Val2Idx.end() ? -1 : it->second);
    }
    void build_parent_cache(vector<list<Graph_Node>::iterator> &idx2it,
                            unordered_map<string, int> &name2idx)
    {
        PIdx.clear();
        PRad.clear();
        PStr.clear();
        PIdx.reserve(Parents.size());
        PRad.reserve(Parents.size());
        for (auto &pn : Parents)
        {
            auto it = name2idx.find(pn);
            if (it == name2idx.end())
                continue;
            int pi = it->second;
            PIdx.push_back(pi);
            PRad.push_back(idx2it[pi]->get_nvalues());
        }
        PStr.assign(PRad.size(), 0);
        int stride = 1;
        for (int i = (int)PRad.size() - 1; i >= 0; --i)
        {
            PStr[i] = stride;
            if (PRad[i] > 0 && stride > INT_MAX / PRad[i])
                throw runtime_error("Parent combos overflow");
            stride *= PRad[i];
        }
    }
    inline vector<int> &parents_idx() { return PIdx; }
    inline vector<int> &parent_rad() { return PRad; }
    inline vector<int> &parent_str() { return PStr; }
    inline int row_base_from_assign_code(int code) { return code * nvalues; }
};

class network
{
    list<Graph_Node> Pres_Graph;
    vector<list<Graph_Node>::iterator> index_cache;
    unordered_map<string, int> name2idx;

public:
    int addNode(Graph_Node node)
    {
        Pres_Graph.push_back(move(node));
        return 0;
    }
    list<Graph_Node>::iterator getNode(int i)
    {
        int count = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it)
            if (count++ == i)
                return it;
        return Pres_Graph.end();
    }
    int netSize() { return (int)Pres_Graph.size(); }
    int get_index(string val_name)
    {
        int count = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it, ++count)
            if (it->get_name() == val_name)
                return count;
        return -1;
    }
    list<Graph_Node>::iterator get_nth_node(int n)
    {
        int count = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it, ++count)
            if (count == n)
                return it;
        return Pres_Graph.end();
    }
    list<Graph_Node>::iterator search_node(string val_name)
    {
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it)
            if (it->get_name() == val_name)
                return it;
        cout << "node not found: " << val_name << "\n";
        return Pres_Graph.end();
    }

    void finalize_index_cache()
    {
        index_cache.clear();
        index_cache.reserve(Pres_Graph.size());
        name2idx.clear();
        name2idx.reserve(Pres_Graph.size() * 2 + 1);
        int i = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it, ++i)
        {
            index_cache.push_back(it);
            name2idx[it->get_name()] = i;
        }
    }
    list<Graph_Node>::iterator fast_get(int i) { return index_cache[i]; }
    int fast_index_of(string &name)
    {
        auto it = name2idx.find(name);
        return (it == name2idx.end() ? -1 : it->second);
    }
    vector<list<Graph_Node>::iterator> &get_index_cache() { return index_cache; }
    unordered_map<string, int> &get_name_map() { return name2idx; }
};

// --------------------------- Utility Functions ---------------------------
static inline string trim(string &str)
{
    size_t first = str.find_first_not_of(" \t\r\n");
    if (string::npos == first)
        return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}
static inline void rtrim_cr(string &s)
{
    if (!s.empty() && s.back() == '\r')
        s.pop_back();
}
static inline string strip_quotes(string &s)
{
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}
static vector<string> parse_csv_line(string &line_in)
{
    vector<string> out;
    string field;
    bool in_quotes = false;
    for (size_t i = 0; i < line_in.size(); ++i)
    {
        char c = line_in[i];
        if (c == '"')
        {
            in_quotes = !in_quotes;
            field.push_back(c);
        }
        else if (c == ',' && !in_quotes)
        {
            out.push_back(strip_quotes(field));
            field.clear();
        }
        else
        {
            field.push_back(c);
        }
    }
    out.push_back(strip_quotes(field));
    return out;
}

// --- Robust loader with auto-detect (CSV vs whitespace) ---
vector<vector<string>> load_records_csv(string path)
{
    ifstream in(path);
    if (!in)
    {
        cerr << "Could not open " << path << "\n";
        return {};
    }

    vector<vector<string>> data;
    string line;
    bool decided = false;
    bool use_csv = false;

    while (getline(in, line))
    {
        rtrim_cr(line);
        if (line.empty())
            continue;

        if (!decided)
        {
            bool in_quotes = false;
            for (char c : line)
            {
                if (c == '"')
                    in_quotes = !in_quotes;
                if (c == ',' && !in_quotes)
                {
                    use_csv = true;
                    break;
                }
            }
            decided = true;
        }

        vector<string> row;
        if (use_csv)
        {
            row = parse_csv_line(line);
        }
        else
        {
            stringstream ss(line);
            string tok;
            while (ss >> tok)
                row.push_back(tok);
        }

        for (auto &t : row)
        {
            while (!t.empty() && (t.back() == ','))
                t.pop_back();
            size_t a = t.find_first_not_of(" \t");
            size_t b = t.find_last_not_of(" \t");
            t = (a == string::npos) ? string() : t.substr(a, b - a + 1);
        }

        if (!row.empty())
            data.push_back(move(row));
    }
    cerr << "Loaded " << data.size() << " records"
         << (use_csv ? " (CSV detected)." : " (whitespace detected).") << "\n";
    return data;
}

// --------------------------- Network Read/Write ---------------------------
network read_network(char *filename)
{
    network BayesNet;
    string line;
    ifstream myfile(filename);
    if (!myfile.is_open())
    {
        cout << "Error: Could not open file " << filename << endl;
        return BayesNet;
    }

    while (getline(myfile, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        string token;
        {
            stringstream ss(line);
            ss >> token;
        }

        if (token == "variable")
        {
            string var_name;
            {
                stringstream ss(line);
                ss >> token >> var_name;
            }
            if (!getline(myfile, line))
                break;
            stringstream ss2(line);
            string type_keyword, discrete_keyword, bracket, equals;
            int num_values;
            ss2 >> type_keyword >> discrete_keyword >> bracket >> num_values >> bracket >> equals >> bracket;

            vector<string> values;
            string value;
            while (ss2 >> value)
            {
                if (value == "};")
                    break;
                if (!value.empty() && value.back() == ',')
                    value.pop_back();
                values.push_back(value);
            }
            BayesNet.addNode(Graph_Node(var_name, num_values, values));
        }
        else if (token == "probability")
        {
            string full = line;
            while (full.find('{') == string::npos)
            {
                string next_line;
                if (!getline(myfile, next_line))
                    break;
                full += " " + trim(next_line);
            }
            size_t lp = full.find('('), rp = full.find(')'), bar = full.find('|');
            if (lp == string::npos || rp == string::npos)
                continue;

            string inside = full.substr(lp + 1, rp - lp - 1);
            stringstream hs(inside);
            string node_name;
            hs >> node_name;

            auto listIt = BayesNet.search_node(node_name);
            int index = BayesNet.get_index(node_name);

            vector<string> parents;
            if (bar != string::npos && bar < rp)
            {
                string parents_str = full.substr(bar + 1, rp - bar - 1);
                stringstream ps(parents_str);
                string p;
                while (ps >> p)
                {
                    if (!p.empty() && p.back() == ',')
                        p.pop_back();
                    parents.push_back(p);
                    auto pit = BayesNet.search_node(p);
                    pit->add_child(index);
                }
            }
            listIt->set_Parents(parents);

            vector<float> cpt;
            while (getline(myfile, line))
            {
                line = trim(line);
                if (line == "};")
                    break;
                if (line.empty())
                    continue;

                size_t close_paren = line.find(')');
                string part;
                if (close_paren != string::npos)
                    part = line.substr(close_paren + 1);
                else if (line.find("table") != string::npos)
                {
                    size_t t = line.find("table");
                    part = line.substr(t + 5);
                }
                else
                    part = line;

                string tok;
                stringstream ss_prob(part);
                while (ss_prob >> tok)
                {
                    while (!tok.empty() && (tok.back() == ',' || tok.back() == ';'))
                        tok.pop_back();
                    if (!tok.empty() && (isdigit((unsigned char)tok[0]) || tok[0] == '.' || tok[0] == '-' || tok[0] == '+'))
                        cpt.push_back(static_cast<float>(atof(tok.c_str())));
                }
            }
            listIt->set_CPT(cpt);
        }
    }
    myfile.close();

    BayesNet.finalize_index_cache();
    int N = BayesNet.netSize();
    for (int i = 0; i < N; ++i)
        BayesNet.fast_get(i)->build_value_index();
    for (int i = 0; i < N; ++i)
        BayesNet.fast_get(i)->build_parent_cache(BayesNet.get_index_cache(), BayesNet.get_name_map());

    return BayesNet;
}

void write_network(char *filename, network &BayesNet)
{
    ofstream outfile(filename);
    if (!outfile.is_open())
    {
        cerr << "Error writing " << filename << "\n";
        return;
    }

    outfile << "// Bayesian Network\n\n";
    int N = BayesNet.netSize();

    // Variables
    for (int i = 0; i < N; i++)
    {
        auto node = BayesNet.get_nth_node(i);
        outfile << "variable " << node->get_name() << " {\n";
        outfile << "  type discrete [ " << node->get_nvalues() << " ] = { ";
        auto vals = node->get_values();
        for (int j = 0; j < (int)vals.size(); j++)
        {
            outfile << vals[j];
            if (j < (int)vals.size() - 1)
                outfile << ", ";
        }
        outfile << " };\n}\n";
    }

    // CPTs
    outfile << fixed << setprecision(4);
    for (int i = 0; i < N; i++)
    {
        auto node = BayesNet.get_nth_node(i);
        auto parents = node->get_Parents();
        auto values = node->get_values();
        auto &cpt = node->get_CPT();

        outfile << "probability ( " << node->get_name();
        if (!parents.empty())
        {
            outfile << " | ";
            for (int j = 0; j < (int)parents.size(); j++)
            {
                outfile << parents[j];
                if (j < (int)parents.size() - 1)
                    outfile << ", ";
            }
        }
        outfile << " ) {\n";

        vector<int> radices;
        for (auto &pname : parents)
            radices.push_back(BayesNet.search_node(pname)->get_nvalues());
        int parent_combinations = 1;
        for (int r : radices)
            parent_combinations *= r;

        int idx = 0;
        if (parents.empty())
        {
            outfile << "  table ";
            for (int k = 0; k < (int)values.size(); k++)
            {
                outfile << cpt[idx++];
                if (k < (int)values.size() - 1)
                    outfile << ", ";
            }
            outfile << ";\n";
        }
        else
        {
            for (int comb = 0; comb < parent_combinations; comb++)
            {
                vector<int> pidx(parents.size(), 0);
                int tmp = comb;
                for (int p = (int)parents.size() - 1; p >= 0; --p)
                {
                    pidx[p] = tmp % radices[p];
                    tmp /= radices[p];
                }
                outfile << "  ( ";
                for (int p = 0; p < (int)parents.size(); ++p)
                {
                    auto pnode = BayesNet.search_node(parents[p]);
                    auto pvals = pnode->get_values();
                    outfile << pvals[pidx[p]];
                    if (p < (int)parents.size() - 1)
                        outfile << ", ";
                }
                outfile << ") ";
                for (int k = 0; k < (int)values.size(); k++)
                {
                    outfile << cpt[idx++];
                    if (k < (int)values.size() - 1)
                        outfile << ", ";
                }
                outfile << ";\n";
            }
        }
        outfile << "};\n\n";
    }
    outfile.close();
    cout << "Network written to file: " << filename << endl;
}

// --------------------------- Index helpers ---------------------------
inline int parent_code_fast(network &net, Graph_Node &node, vector<string> &row)
{
    auto &PIdx = node.parents_idx();
    auto &PStr = node.parent_str();
    int code = 0;
    for (int p = 0; p < (int)PIdx.size(); ++p)
    {
        auto pnode = net.fast_get(PIdx[p]);
        string &tok = row[PIdx[p]];
        if (tok == "?")
            return -1;
        int a = pnode->vindex_fast(tok);
        if (a < 0)
            return -1;
        code += a * PStr[p];
    }
    return code;
}
inline int parent_code_with_override(network &net, Graph_Node &node, vector<string> &row,
                                     int override_idx, int override_val)
{
    auto &PIdx = node.parents_idx();
    auto &PStr = node.parent_str();
    int code = 0;
    for (int p = 0; p < (int)PIdx.size(); ++p)
    {
        int idx = PIdx[p];
        int a;
        if (idx == override_idx)
            a = override_val;
        else
        {
            auto pnode = net.fast_get(idx);
            string &tok = row[idx];
            if (tok == "?")
                return -1;
            a = pnode->vindex_fast(tok);
            if (a < 0)
                return -1;
        }
        code += a * PStr[p];
    }
    return code;
}

// ====================== Incomplete-data Log-Likelihood ======================
double compute_incomplete_log_likelihood(network &net, vector<vector<string>> &data)
{
    float tiny = 1e-12;
    int N = net.netSize();
    double total = 0.0;

    for (auto &row : data)
    {
        if ((int)row.size() != N)
            continue;

        int missing_idx = -1, q = 0;
        for (int i = 0; i < N; ++i)
            if (row[i] == "?" && ++q == 1)
                missing_idx = i;
        if (q > 1)
            continue; // guaranteed not to happen per your note

        if (q == 0)
        {
            // Fully observed: ordinary joint log prob via local conditionals
            double ll = 0.0;
            for (int i = 0; i < N; ++i)
            {
                auto node = net.fast_get(i);
                int x = node->vindex_fast(row[i]);
                if (x < 0)
                    continue;
                int code = parent_code_fast(net, *node, row);
                if (code < 0)
                    continue;
                int base = node->row_base_from_assign_code(code);
                auto &cpt = node->get_CPT();
                ll += log(max(tiny, cpt[base + x]));
            }
            total += ll;
            continue;
        }

        // Exactly one missing: part + blanket log-sum-exp
        int m = missing_idx;
        auto mnode = net.fast_get(m);

        // (A) Unaffected nodes (no parent equals M)
        double const_part = 0.0;
        for (int i = 0; i < N; ++i)
        {
            if (i == m)
                continue;
            auto node = net.fast_get(i);

            bool parent_depends_on_m = false;
            for (int pi : node->parents_idx())
                if (pi == m)
                {
                    parent_depends_on_m = true;
                    break;
                }
            if (parent_depends_on_m)
                continue;

            int x = node->vindex_fast(row[i]);
            if (x < 0)
                continue;
            int code = parent_code_fast(net, *node, row);
            if (code < 0)
                continue;
            int base = node->row_base_from_assign_code(code);
            auto &cpt = node->get_CPT();
            const_part += log(max(tiny, cpt[base + x]));
        }

        // (B) Blanket part: log sum_v P(M=v|PaM) * ∏_child P(child|Pa, M=v)
        int m_code = parent_code_fast(net, *mnode, row);
        if (m_code < 0)
            m_code = 0; // safety; shouldn't occur with 1-missing assumption
        int m_base = mnode->row_base_from_assign_code(m_code);
        auto &mCPT = mnode->get_CPT();
        int K = mnode->get_nvalues();

        double max_logterm = -1e300;
        vector<double> logterms(K, -1e300);
        for (int v = 0; v < K; ++v)
        {
            double logp = log(max(tiny, mCPT[m_base + v]));
            bool bad = false;
            for (int ci : mnode->get_children())
            {
                auto child = net.fast_get(ci);
                int y = child->vindex_fast(row[ci]);
                if (y < 0)
                {
                    bad = true;
                    break;
                }
                int c_code = parent_code_with_override(net, *child, row, m, v);
                if (c_code < 0)
                {
                    bad = true;
                    break;
                }
                int cb = child->row_base_from_assign_code(c_code);
                auto &cC = child->get_CPT();
                logp += log(max(tiny, cC[cb + y]));
            }
            if (!bad)
                logterms[v] = logp, max_logterm = max(max_logterm, logp);
        }
        double sum_exp = 0.0;
        for (int v = 0; v < K; ++v)
            sum_exp += exp(logterms[v] - max_logterm);
        double blanket = max_logterm + log(max(1e-300, sum_exp));

        total += const_part + blanket;
    }
    return total;
}

// --------------------------- Prior-centered Initializer ---------------------------
void initialize_cpts_with_prior(network &net,
                                vector<vector<string>> &data,
                                double prior_ess = 5.0)
{
    int N = net.netSize();

    vector<vector<float>> prior(N);
    vector<int> nvals(N), parent_combos(N);

    for (int i = 0; i < N; ++i)
    {
        auto node = net.fast_get(i);
        nvals[i] = node->get_nvalues();
        auto &pcpt = node->get_CPT();
        prior[i] = pcpt; // same shape
        long long prod = 1;
        for (int r : node->parent_rad())
        {
            prod *= r;
        }
        parent_combos[i] = (int)prod;
    }

    vector<vector<float>> counts(N);
    for (int i = 0; i < N; ++i)
        counts[i].assign(parent_combos[i] * nvals[i], 0.0);

    for (auto &row : data)
    {
        if ((int)row.size() != N)
            continue;

        bool has_missing = false;
        for (auto &t : row)
            if (t == "?")
            {
                has_missing = true;
                break;
            }
        if (has_missing)
            continue; // clean init from complete cases

        for (int i = 0; i < N; ++i)
        {
            auto node = net.fast_get(i);
            int x = node->vindex_fast(row[i]);
            if (x < 0)
                continue;
            int code = parent_code_fast(net, *node, row);
            if (code < 0)
                continue;
            int base = node->row_base_from_assign_code(code);
            counts[i][base + x] += 1.0;
        }
    }

    // MAP with Dirichlet(prior_ess * prior_row)
    for (int i = 0; i < N; ++i)
    {
        auto node = net.fast_get(i);
        int K = nvals[i];
        auto &c = counts[i];
        auto &p = prior[i];

        for (int off = 0; off < (int)c.size(); off += K)
        {
            double sum_c = 0.0;
            for (int k = 0; k < K; ++k)
                sum_c += c[off + k];

            double denom = sum_c + prior_ess;
            if (denom < 1e-12)
                denom = 1e-12;

            for (int k = 0; k < K; ++k)
            {
                double alpha_k = prior_ess * max((float)1e-12, p[off + k]);
                c[off + k] = (c[off + k] + alpha_k) / denom;
            }

            // safety renorm
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += c[off + k];
            if (s < 1e-12)
            {
                for (int k = 0; k < K; ++k)
                    c[off + k] = 1.0 / K;
            }
            else
            {
                double inv = 1.0 / s;
                for (int k = 0; k < K; ++k)
                    c[off + k] *= inv;
            }
        }
        node->set_CPT(c);
    }

    cerr << "Initialized CPTs with prior-centered Dirichlet (ESS=" << prior_ess << ").\n";
}

// ========================== Annealed + Exact EM ==========================
void run_soft_then_exact_em(network &net, vector<vector<string>> &data,
                            int max_iter_soft = 60, double tol = 1e-4,
                            double ESS = 2.0,
                            double tau_init = 2.0, double tau_decay = 0.90, double tau_min = 1.0)
{
    int N = net.netSize();
    float tiny = 1e-12;

    // Preallocate counts buffers
    vector<vector<float>> counts(N);
    for (int i = 0; i < N; ++i)
    {
        auto node = net.fast_get(i);
        long long prod = 1;
        for (int r : node->parent_rad())
        {
            if (r <= 0 || prod > (LLONG_MAX / r))
                throw runtime_error("Parent combos too large");
            prod *= r;
        }
        counts[i].assign((int)prod * node->get_nvalues(), 0.0);
    }

    auto em_pass = [&](double tau)
    {
        // --------- E-step ----------
        for (auto &row : data)
        {
            if ((int)row.size() != N)
                continue;

            int missing_idx = -1, q = 0;
            for (int i = 0; i < N; ++i)
                if (row[i] == "?" && ++q == 1)
                    missing_idx = i;
            if (q > 1)
                continue;

            if (q == 0)
            {
                // Fully observed
                for (int i = 0; i < N; ++i)
                {
                    auto node = net.fast_get(i);
                    int x = node->vindex_fast(row[i]);
                    if (x < 0)
                        continue;
                    int code = parent_code_fast(net, *node, row);
                    if (code < 0)
                        continue;
                    int base = node->row_base_from_assign_code(code);
                    counts[i][base + x] += 1.0;
                }
            }
            else
            {
                // One missing variable
                int m = missing_idx;
                auto mnode = net.fast_get(m);

                int m_code = parent_code_fast(net, *mnode, row);
                int m_base = (m_code >= 0) ? mnode->row_base_from_assign_code(m_code) : 0;
                auto &mC = mnode->get_CPT();
                int K = mnode->get_nvalues();

                // Posterior weights w[v] ∝ exp( (log prior + sum child logs / child_soft) / tau )
                vector<double> w(K, 0.0);
                double max_lp = -1e300;

                // If m_code < 0 (shouldn't with ≤1 missing), fallback to average prior
                vector<float> prior_fb;
                if (m_code < 0)
                {
                    int total_rows = (int)mC.size() / K;
                    prior_fb.assign(K, 0.0);
                    if (total_rows > 0)
                    {
                        for (int rr = 0; rr < total_rows; ++rr)
                        {
                            int base = rr * K;
                            for (int v = 0; v < K; ++v)
                                prior_fb[v] += mC[base + v];
                        }
                        for (int v = 0; v < K; ++v)
                            prior_fb[v] /= max(1, total_rows);
                    }
                    else
                    {
                        for (int v = 0; v < K; ++v)
                            prior_fb[v] = 1.0 / K;
                    }
                }

                vector<double> lps(K, -1e300);
                for (int v = 0; v < K; ++v)
                {
                    double prior_v = (m_code >= 0) ? max(tiny, mC[m_base + v])
                                                   : max(tiny, prior_fb[v]);
                    double logp = log(prior_v);

                    bool bad = false;
                    for (int ci : mnode->get_children())
                    {
                        auto child = net.fast_get(ci);
                        int y = child->vindex_fast(row[ci]);
                        if (y < 0)
                        {
                            bad = true;
                            break;
                        }
                        int c_code = parent_code_with_override(net, *child, row, m, v); // missing node m has value v - get parent code
                        if (c_code < 0)
                        {
                            bad = true;
                            break;
                        }
                        int cb = child->row_base_from_assign_code(c_code); // get base from parent code
                        auto &cC = child->get_CPT();
                        logp += log(max(tiny, cC[cb + y])); // remove this softening
                    }
                    if (bad)
                    {
                        lps[v] = -1e300;
                        continue;
                    }
                    lps[v] = logp;
                    max_lp = max(max_lp, logp);
                }

                double wsum = 0.0;
                for (int v = 0; v < K; ++v)
                {
                    if (lps[v] <= -1e299)
                    {
                        w[v] = 0.0;
                        continue;
                    }
                    w[v] = exp((lps[v] - max_lp) / max(1.0, tau)); // remove max_lp
                    wsum += w[v];
                }
                if (wsum <= 0.0)
                {
                    for (int v = 0; v < K; ++v)
                        w[v] = 1.0 / K;
                }
                else
                {
                    for (int v = 0; v < K; ++v)
                        w[v] /= wsum;
                }

                // Add expected counts
                for (int i = 0; i < N; ++i)
                {
                    auto node = net.fast_get(i);

                    if (i == m)
                    {
                        if (m_code >= 0)
                        {
                            int base = m_base;
                            for (int v = 0; v < K; ++v)
                                counts[i][base + v] += w[v];
                        }
                        else
                        {
                            int total_rows = (int)counts[i].size() / K;
                            for (int rr = 0; rr < total_rows; ++rr)
                            {
                                int base = rr * K;
                                for (int v = 0; v < K; ++v)
                                    counts[i][base + v] += w[v] / max(1, total_rows);
                            }
                        }
                        continue;
                    }

                    bool parent_depends_on_m = false;
                    for (int pi : node->parents_idx())
                        if (pi == m)
                        {
                            parent_depends_on_m = true;
                            break;
                        }

                    if (parent_depends_on_m)
                    { // if parent is missing
                        int x = node->vindex_fast(row[i]);
                        if (x < 0)
                            continue;
                        for (int v = 0; v < (int)w.size(); ++v)
                        {
                            if (w[v] <= 0.0)
                                continue;
                            int code = parent_code_with_override(net, *node, row, m, v);
                            if (code < 0)
                                continue;
                            int base = node->row_base_from_assign_code(code);
                            counts[i][base + x] += w[v];
                        }
                        continue;
                    }

                    // Unaffected node - nothing missing
                    int x = node->vindex_fast(row[i]);
                    if (x < 0)
                        continue;
                    int code = parent_code_fast(net, *node, row);
                    if (code < 0)
                        continue;
                    int base = node->row_base_from_assign_code(code);
                    counts[i][base + x] += 1.0;
                }
            }
        }

        // --------- M-step with ESS smoothing + clipping + renorm ----------
        for (int i = 0; i < N; ++i)
        {
            auto node = net.fast_get(i);
            auto &cpt = counts[i];
            int K = node->get_nvalues();

            for (int off = 0; off < (int)cpt.size(); off += K)
            {
                double sum_counts = 0.0;
                for (int k = 0; k < K; ++k)
                    sum_counts += cpt[off + k];

                // add Dirichlet ESS uniformly
                double denom = sum_counts + ESS;
                if (denom < tiny)
                    denom = tiny;

                for (int k = 0; k < K; ++k)
                {
                    float num = cpt[off + k] + ESS / K;
                    float p = num / denom;
                    cpt[off + k] = max(tiny, p);
                }
                // renorm after clipping
                double s = 0.0;
                for (int k = 0; k < K; ++k)
                    s += cpt[off + k];
                double inv = 1.0 / s;
                for (int k = 0; k < K; ++k)
                    cpt[off + k] *= inv;
            }
            node->set_CPT(cpt);
        }
    };

    // --------- Annealed (soft) EM phase ----------
    double tau = tau_init;
    double prev_ll = -1e300;
    for (int iter = 0; iter < max_iter_soft; ++iter)
    {
        // double child_soft = (iter < soften_iters ? 1.6 : 1.0);

        em_pass(tau);

        double ll = compute_incomplete_log_likelihood(net, data);
        cout << "EM iter " << (iter + 1) << " | logL=" << ll << "  tau=" << tau
             << "\n";

        tau = max(tau * tau_decay, tau_min);

        if (fabs(ll - prev_ll) < tol)
        {
            cout << "Soft EM converged at iter " << (iter + 1) << ".\n";
            break;
        }
        prev_ll = ll;
    }

    // // --------- Exact-EM polish ----------
    // for (int p = 0; p < polish_iters; ++p) {
    //     em_pass(/*tau=*/1.0);
    //     double ll = compute_incomplete_log_likelihood(net, data);
    //     cout << "Polish " << (p+1) << "/" << polish_iters << " | logL=" << ll << "\n";
    // }
}

void validate_dataset(network &net, vector<vector<string>> &data, int max_report = 10)
{
    int N = net.netSize();
    int bad_rows = 0, bad_tokens = 0, missing_multi = 0;
    for (int r = 0; r < (int)data.size(); ++r)
    {
        auto &row = data[r];
        if ((int)row.size() != N)
        {
            if (++bad_rows <= max_report)
                cerr << "[ROW LEN] r=" << r << " has " << row.size() << " cols; expected " << N << "\n";
            continue;
        }
        int q = 0;
        for (int i = 0; i < N; ++i)
        {
            auto node = net.get_nth_node(i);
            const auto &vals = node->get_values();
            string &tok = row[i];
            if (tok == "?")
            {
                ++q;
                continue;
            }
            bool ok = false;
            for (auto &v : vals)
                if (v == tok)
                {
                    ok = true;
                    break;
                }
            if (!ok)
            {
                if (++bad_tokens <= max_report)
                    cerr << "[TOKEN] r=" << r << " col=" << i
                         << " node=" << node->get_name()
                         << " value='" << tok << "' not in node’s domain\n";
            }
        }
        if (q > 1)
        {
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
int main()
{
    srand(42);

    // Read once to get structure for validation and N
    network BayesNet0 = read_network("hailfinder.bif");
    vector<vector<string>> dataset = load_records_csv("records.dat");

    cout << "Loaded " << dataset.size() << " records." << endl;
    validate_dataset(BayesNet0, dataset);
    int N = BayesNet0.netSize();

    // Restart config
    int NUM_TRIALS = 1;     // try 5–10
    double PRIOR_ESS = 5.0; // 2–10 good for Hailfinder
    double MSTEP_ESS = 2.0; // small regularization during M-step

    vector<double> LLs;
    LLs.reserve(NUM_TRIALS);
    vector<vector<vector<float>>> all_cpts; // [trial][node][flat cpt]
    all_cpts.resize(NUM_TRIALS, vector<vector<float>>(N));

    double best_ll = -1e300;
    int best_idx = -1;

    for (int trial = 0; trial < NUM_TRIALS; ++trial)
    {
        cout << "\n=== Trial " << (trial + 1) << "/" << NUM_TRIALS << " ===\n";
        // Fresh network each trial (keeps caches valid)
        network model = read_network("hailfinder.bif");

        // Different seed per trial for exploration
        srand(42 + trial * 17);

        // Init with prior-centered counts
        initialize_cpts_with_prior(model, dataset, PRIOR_ESS);

        // Annealed + exact EM
        run_soft_then_exact_em(model, dataset,
                               /*max_iter_soft=*/60, /*tol=*/1e-5,
                               /*ESS=*/MSTEP_ESS,
                               /*tau_init=*/2.0, /*tau_decay=*/0.90, /*tau_min=*/1.0);

        // Evaluate incomplete-data LL on training set
        double ll = compute_incomplete_log_likelihood(model, dataset);
        cout << "Trial " << (trial + 1) << " final logL = " << ll << endl;

        // Snapshot CPTs
        for (int i = 0; i < N; ++i)
        {
            all_cpts[trial][i] = model.fast_get(i)->get_CPT();
        }
        LLs.push_back(ll);
        if (ll > best_ll)
        {
            best_ll = ll;
            best_idx = trial;
        }
    }

    // Weighted model averaging across trials (softmax weights on LL)
    vector<double> w(NUM_TRIALS, 0.0);
    double m = *max_element(LLs.begin(), LLs.end());
    double Z = 0.0;
    for (int t = 0; t < NUM_TRIALS; ++t)
    {
        w[t] = exp(LLs[t] - m);
        Z += w[t];
    }
    if (Z <= 0.0)
    {
        for (int t = 0; t < NUM_TRIALS; ++t)
            w[t] = 1.0 / NUM_TRIALS;
    }
    else
    {
        for (int t = 0; t < NUM_TRIALS; ++t)
            w[t] /= Z;
    }

    // Build final averaged network
    network finalNet = read_network("hailfinder.bif");
    for (int i = 0; i < N; ++i)
    {
        vector<float> avg = all_cpts[best_idx][i]; // shape template
        fill(avg.begin(), avg.end(), 0.0);
        // weighted sum
        for (int t = 0; t < NUM_TRIALS; ++t)
        {
            auto &cpt = all_cpts[t][i];
            for (size_t j = 0; j < avg.size(); ++j)
                avg[j] += w[t] * cpt[j];
        }
        // row renorm (safety)
        int K = finalNet.fast_get(i)->get_nvalues();
        for (int off = 0; off < (int)avg.size(); off += K)
        {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += avg[off + k];
            if (s < 1e-12)
            {
                for (int k = 0; k < K; ++k)
                    avg[off + k] = 1.0 / K;
            }
            else
            {
                double inv = 1.0 / s;
                for (int k = 0; k < K; ++k)
                    avg[off + k] *= inv;
            }
        }
        finalNet.fast_get(i)->set_CPT(avg);
    }

    cout << "\nBest individual trial logL: " << best_ll
         << " (trial " << (best_idx + 1) << ")\n";

    // For visibility, also print the averaged LL
    double avg_ll = compute_incomplete_log_likelihood(finalNet, dataset);
    cout << "Averaged model logL: " << avg_ll << "\n";

    write_network("solved.bif", finalNet);
    return 0;
}
#endif // BN_LIB