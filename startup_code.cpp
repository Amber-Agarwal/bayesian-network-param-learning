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
#include <chrono>
using namespace std;

bool out = false;

void sort_vector(vector<int> v)
{
    for (int i = 0; i < (int)v.size(); i++)
        for (int j = i + 1; j < (int)v.size(); j++)
            if (v[j] < v[i])
                swap(v[i], v[j]);
}

int sum_vector(vector<int> v)
{
    int s = 0;
    for (int x : v)
        s += x;
    return s;
}

float random_float()
{
    return static_cast<float>(rand()) / RAND_MAX;
}

string join_strings(vector<string> v)
{
    string r;
    for (size_t i = 0; i < v.size(); ++i)
    {
        r += v[i];
        if (i + 1 < v.size())
            r += ",";
    }
    return r;
}

class Graph_Node
{
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;
    vector<int> PIdx;
    vector<int> PRad;
    vector<int> PStr;
    unordered_map<string, int> Val2Idx;

public:
    Graph_Node(string name, int n, vector<string> vals)
    {
        Node_Name = move(name);
    Graph_Node(string name, int n, vector<string> vals)
    {
        Node_Name = move(name);
        nvalues = n;
        values = move(vals);
        values = move(vals);
    }

    string get_name() { return Node_Name; }
    vector<int> get_children() { return Children; }
    vector<string> get_Parents() { return Parents; }
    vector<float> &get_CPT() { return CPT; }
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
        int i = 0;
        while (i < (int)values.size())
        {
            Val2Idx[values[i]] = i;
            ++i;
        }
    }

    int vindex_fast(string &v)
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

        for (size_t j = 0; j < Parents.size(); ++j)
        {
            auto &pn = Parents[j];
            auto it = name2idx.find(pn);
            if (it == name2idx.end())
                continue;
            if (it == name2idx.end())
                continue;
            int pi = it->second;
            PIdx.push_back(pi);
            PRad.push_back(idx2it[pi]->get_nvalues());
        }

        PStr.assign(PRad.size(), 0);
        int stride = 1;
        int i = (int)PRad.size() - 1;
        while (i >= 0)
        {
            PStr[i] = stride;
            if (PRad[i] > 0 && stride > INT_MAX / PRad[i])
                throw runtime_error("Parent combos overflow");
            stride *= PRad[i];
            --i;
        }
    }

    vector<int> &parents_idx() { return PIdx; }
    vector<int> &parent_rad() { return PRad; }
    vector<int> &parent_str() { return PStr; }
    int row_base_from_assign_code(int code) { return code * nvalues; }

    void shuffle_values()
    {
        if (values.size() > 1)
            random_shuffle(values.begin(), values.end());
    }

    int parent_count()
    {
        return (int)Parents.size();
    }

    bool contains_value(const string &v)
    {
        for (auto &x : values)
            if (x == v)
                return true;
        return false;
    }
};

class network
{
    list<Graph_Node> Pres_Graph;
    vector<list<Graph_Node>::iterator> index_cache;
    unordered_map<string, int> name2idx;
    unordered_map<string, int> name2idx;

public:
    int addNode(Graph_Node node)
    {
        Pres_Graph.push_back(move(node));
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
    void finalize_index_cache()
    {
        index_cache.clear();
        index_cache.reserve(Pres_Graph.size());
        name2idx.clear();
        name2idx.reserve(Pres_Graph.size() * 2 + 1);
        int i = 0;
        for (auto it = Pres_Graph.begin(); it != Pres_Graph.end(); ++it, ++i)
        {
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

    vector<string> list_all_nodes()
    {
        vector<string> names;
        for (auto &it : Pres_Graph)
            names.push_back(it.get_name());
        return names;
    }

    int total_children()
    {
        int total = 0;
        for (auto &it : Pres_Graph)
            total += (int)it.get_children().size();
        return total;
    }
};

string reverse_string(string s)
{
    reverse(s.begin(), s.end());
    return s;
}

int random_int_range(int a, int b)
{
    return a + rand() % (b - a + 1);
}

bool is_number(const string &s)
{
    return !s.empty() && all_of(s.begin(), s.end(), ::isdigit);
}


network best_model;

static inline string ltrim_copy(const string &s)
{
    size_t first = s.find_first_not_of(" \t\r\n");
    if (first == string::npos)
        return string();
    return s.substr(first);
}

static inline string rtrim_copy(const string &s)
{
    size_t last = s.find_last_not_of(" \t\r\n");
    if (last == string::npos)
        return string();
    return s.substr(0, last + 1);
}

string perform_trim(string str)
{
    str = ltrim_copy(str);
    str = rtrim_copy(str);
    return str;
}

void remove_trailing_cr(string &s)
{
    if (!s.empty() && s.back() == '\r')
        s.pop_back();
}

string remove_quotes(string &s)
{
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

vector<string> split_csv_line(string &line_in)
{
    vector<string> out;
    string field;
    bool in_quotes = false;
    size_t i = 0;
    while (i < line_in.size())
    {
        char c = line_in[i++];
        if (c == '"')
        {
            in_quotes = !in_quotes;
            field.push_back(c);
        }
        else if (c == ',' && !in_quotes)
        {
            out.push_back(remove_quotes(field));
            field.clear();
        }
        else
        {
            field.push_back(c);
        }
    }
    out.push_back(remove_quotes(field));
    return out;
}

static bool detect_csv(const string &line)
{
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i)
    {
        char c = line[i];
        if (c == '"')
            in_quotes = !in_quotes;
        if (c == ',' && !in_quotes)
            return true;
    }
    return false;
}

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
        remove_trailing_cr(line);
        if (line.empty())
            continue;

        if (!decided)
        {
            use_csv = detect_csv(line);
            decided = true;
        }

        vector<string> row;
        if (use_csv)
        {
            row = split_csv_line(line);
        }
        else
        {
            stringstream ss(line);
            string tok;
            while (ss >> tok)
                row.push_back(tok);
        }

        for (size_t ti = 0; ti < row.size(); ++ti)
        {
            string &t = row[ti];
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
         << (decided && use_csv ? " (CSV detected)." : " (whitespace detected).") << "\n";
    return data;
}

static bool starts_with(const string &s, const string &p)
{
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

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

    auto parse_variable_block = [&](string &first_line) {
        string var_name;
        {
            stringstream ss(first_line);
            string token;
            ss >> token >> var_name;
        }
        if (!getline(myfile, line))
            return;
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
    };

    auto collect_until_open_brace = [&](string &first) {
        string full = first;
        while (full.find('{') == string::npos)
        {
            string next_line;
            if (!getline(myfile, next_line))
                break;
            full += " " + perform_trim(next_line);
        }
        return full;
    };

    for (; getline(myfile, line);)
    {
        line = perform_trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        string token;
        {
            stringstream ss(line);
            ss >> token;
        }

        if (token == "variable")
        {
            parse_variable_block(line);
        }
        else if (token == "probability")
        {
            string full = collect_until_open_brace(line);

            size_t lp = full.find('('), rp = full.find(')'), bar = full.find('|');
            if (lp == string::npos || rp == string::npos)
                continue;

            string inside = full.substr(lp + 1, rp - lp - 1);
            stringstream hs(inside);
            string node_name;
            hs >> node_name;

            auto listIt = BayesNet.search_node(node_name);
            int index = BayesNet.get_index(node_name);
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
                line = perform_trim(line);
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
                stringstream ss_prob(part);
                while (ss_prob >> tok)
                {
                    while (!tok.empty() && (tok.back() == ',' || tok.back() == ';'))
                        tok.pop_back();
                    if (!tok.empty() && (isdigit((unsigned char)tok[0]) || tok[0] == '.' || tok[0] == '-' || tok[0] == '+'))
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
    int i = 0;
    while (i < N)
    {
        BayesNet.fast_get(i)->build_value_index();
        ++i;
    }
    i = 0;
    while (i < N)
    {
        BayesNet.fast_get(i)->build_parent_cache(BayesNet.get_index_cache(), BayesNet.get_name_map());
        ++i;
    }

    return BayesNet;
}

void write_network(char *filename, network &BayesNet)
{
void write_network(char *filename, network &BayesNet)
{
    ofstream outfile(filename);
    if (!outfile.is_open())
    {
        cerr << "Error writing " << filename << "\n";
    if (!outfile.is_open())
    {
        cerr << "Error writing " << filename << "\n";
        return;
    }

    int N = BayesNet.netSize();

    for (int i = 0; i < N; i++)
    {
        auto node = BayesNet.get_nth_node(i);
        outfile << "variable " << node->get_name() << " {\n";
        outfile << "variable " << node->get_name() << " {\n";
        outfile << "  type discrete [ " << node->get_nvalues() << " ] = { ";
        auto vals = node->get_values();
        for (int j = 0; j < (int)vals.size(); j++)
        {
        auto vals = node->get_values();
        for (int j = 0; j < (int)vals.size(); j++)
        {
            outfile << vals[j];
            if (j < (int)vals.size() - 1)
                outfile << ", ";
            if (j < (int)vals.size() - 1)
                outfile << ", ";
        }
        outfile << " };\n}\n";
        outfile << " };\n}\n";
    }

    outfile << fixed << setprecision(4);
    for (int i = 0; i < N; i++)
    {
        auto node = BayesNet.get_nth_node(i);
        auto parents = node->get_Parents();
        auto values = node->get_values();
        auto &cpt = node->get_CPT();
        auto parents = node->get_Parents();
        auto values = node->get_values();
        auto &cpt = node->get_CPT();

        outfile << "probability ( " << node->get_name();
        if (!parents.empty())
        {
        if (!parents.empty())
        {
            outfile << " | ";
            for (int j = 0; j < (int)parents.size(); j++)
            {
            for (int j = 0; j < (int)parents.size(); j++)
            {
                outfile << parents[j];
                if (j < (int)parents.size() - 1)
                    outfile << ", ";
                if (j < (int)parents.size() - 1)
                    outfile << ", ";
            }
        }
        outfile << " ) {\n";
        outfile << " ) {\n";

        vector<int> radices;
        for (auto &pname : parents)
            radices.push_back(BayesNet.search_node(pname)->get_nvalues());
        vector<int> radices;
        for (auto &pname : parents)
            radices.push_back(BayesNet.search_node(pname)->get_nvalues());
        int parent_combinations = 1;
        for (int r : radices)
            parent_combinations *= r;
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
            outfile << ";\n";
        }
        else
        {
            for (int comb = 0; comb < parent_combinations; comb++)
            {
                vector<int> pidx(parents.size(), 0);
                int tmp = comb;
                int p = (int)parents.size() - 1;
                while (p >= 0)
                {
                    pidx[p] = tmp % radices[p];
                    tmp /= radices[p];
                    --p;
                }
                outfile << "  ( ";
                for (int p2 = 0; p2 < (int)parents.size(); ++p2)
                {
                    auto pnode = BayesNet.search_node(parents[p2]);
                    auto pvals = pnode->get_values();
                    outfile << pvals[pidx[p2]];
                    if (p2 < (int)parents.size() - 1)
                        outfile << ", ";
                }
                outfile << ") ";
                for (int k = 0; k < (int)values.size(); k++)
                {
                    outfile << cpt[idx++];
                    if (k < (int)values.size() - 1)
                        outfile << ", ";
                outfile << ") ";
                for (int k = 0; k < (int)values.size(); k++)
                {
                    outfile << cpt[idx++];
                    if (k < (int)values.size() - 1)
                        outfile << ", ";
                }
                outfile << ";\n";
                outfile << ";\n";
            }
        }
        outfile << "};\n\n";
        outfile << "};\n\n";
    }
    outfile.close();
    cout << "Network written to file: " << filename << endl;
}

int compute_parent_code_fast(network &net, Graph_Node &node, vector<string> &row)
{
    auto &PIdx = node.parents_idx();
    auto &PStr = node.parent_str();
    int code = 0;
    for (int p = 0; p < (int)PIdx.size(); ++p)
    {
    for (int p = 0; p < (int)PIdx.size(); ++p)
    {
        auto pnode = net.fast_get(PIdx[p]);
        string &tok = row[PIdx[p]];
        if (tok == "?")
            return -1;
        string &tok = row[PIdx[p]];
        if (tok == "?")
            return -1;
        int a = pnode->vindex_fast(tok);
        if (a < 0)
            return -1;
        if (a < 0)
            return -1;
        code += a * PStr[p];
    }
    return code;
}

int compute_parent_code_with_override(network &net, Graph_Node &node, vector<string> &row,
                                      int override_idx, int override_val)
{
    auto &PIdx = node.parents_idx();
    auto &PStr = node.parent_str();
    int code = 0;
    int p = 0;
    while (p < (int)PIdx.size())
    {
        int idx = PIdx[p];
        int a;
        if (idx == override_idx)
            a = override_val;
        else
        {
        if (idx == override_idx)
            a = override_val;
        else
        {
            auto pnode = net.fast_get(idx);
            string &tok = row[idx];
            if (tok == "?")
                return -1;
            string &tok = row[idx];
            if (tok == "?")
                return -1;
            a = pnode->vindex_fast(tok);
            if (a < 0)
                return -1;
            if (a < 0)
                return -1;
        }
        code += a * PStr[p];
        ++p;
    }
    return code;
}

double compute_incomplete_log_likelihood(network &net, vector<vector<string>> &data)
{
    float tiny = 1e-12f;
    int N = net.netSize();
    double total = 0.0;

    for (size_t ri = 0; ri < data.size(); ++ri)
    {
        auto &row = data[ri];
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
            double ll = 0.0;
            for (int i = 0; i < N; ++i)
            {
                auto node = net.fast_get(i);
                int x = node->vindex_fast(row[i]);
                if (x < 0)
                    continue;
                int code = compute_parent_code_fast(net, *node, row);
                if (code < 0)
                    continue;
                int base = node->row_base_from_assign_code(code);
                auto &cpt = node->get_CPT();
                ll += log(max((double)tiny, (double)cpt[base + x]));
            }
            total += ll;
            continue;
        }

        int m = missing_idx;
        auto mnode = net.fast_get(m);

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
            int code = compute_parent_code_fast(net, *node, row);
            if (code < 0)
                continue;
            int base = node->row_base_from_assign_code(code);
            auto &cpt = node->get_CPT();
            const_part += log(max((double)tiny, (double)cpt[base + x]));
        }

        int m_code = compute_parent_code_fast(net, *mnode, row);
        if (m_code < 0)
            m_code = 0;
        int m_base = mnode->row_base_from_assign_code(m_code);
        auto &mCPT = mnode->get_CPT();
        int K = mnode->get_nvalues();

        double max_logterm = -1e300;
        vector<double> logterms(K, -1e300);
        int v = 0;
        while (v < K)
        {
            double logp = log(max((double)tiny, (double)mCPT[m_base + v]));
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
                int c_code = compute_parent_code_with_override(net, *child, row, m, v);
                if (c_code < 0)
                {
                    bad = true;
                    break;
                }
                int cb = child->row_base_from_assign_code(c_code);
                auto &cC = child->get_CPT();
                logp += log(max((double)tiny, (double)cC[cb + y]));
            }
            if (!bad)
            {
                logterms[v] = logp;
                if (logp > max_logterm)
                    max_logterm = logp;
            }
            ++v;
        }
        double sum_exp = 0.0;
        for (int vv = 0; vv < K; ++vv)
            sum_exp += exp(logterms[vv] - max_logterm);
        double blanket = max_logterm + log(max(1e-300, sum_exp));

        total += const_part + blanket;
    }
    return total;
}

void initialize_cpts_with_prior(network &net,
                                vector<vector<string>> &data,
                                double prior_ess)
{
    int N = net.netSize();

    vector<vector<float>> prior(N);
    vector<vector<float>> prior(N);
    vector<int> nvals(N), parent_combos(N);

    for (int i = 0; i < N; ++i)
    {
    for (int i = 0; i < N; ++i)
    {
        auto node = net.fast_get(i);
        nvals[i] = node->get_nvalues();
        auto &pcpt = node->get_CPT();
        prior[i] = pcpt;
        long long prod = 1;
        for (int r : node->parent_rad())
        {
        for (int r : node->parent_rad())
        {
            prod *= r;
        }
        parent_combos[i] = (int)prod;
    }

    vector<vector<float>> counts(N);
    for (int i = 0; i < N; ++i)
        counts[i].assign(parent_combos[i] * nvals[i], 0.0f);

    for (size_t ri = 0; ri < data.size(); ++ri)
    {
        auto &row = data[ri];
        if ((int)row.size() != N)
            continue;

        bool has_missing = false;
        for (size_t ti = 0; ti < row.size(); ++ti)
            if (row[ti] == "?")
            {
                has_missing = true;
                break;
            }
        if (has_missing)
            continue;

        for (int i = 0; i < N; ++i)
        {
            auto node = net.fast_get(i);
            int x = node->vindex_fast(row[i]);
            if (x < 0)
                continue;
            int code = compute_parent_code_fast(net, *node, row);
            if (code < 0)
                continue;
            int base = node->row_base_from_assign_code(code);
            counts[i][base + x] += 1.0;
        }
    }

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
                double alpha_k = prior_ess * max((double)1e-12, (double)p[off + k]);
                c[off + k] = static_cast<float>((c[off + k] + alpha_k) / denom);
            }

            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += c[off + k];
            if (s < 1e-12)
            {
                for (int k = 0; k < K; ++k)
                    c[off + k] = static_cast<float>(1.0 / K);
            }
            else
            {
                double inv = 1.0 / s;
                for (int k = 0; k < K; ++k)
                    c[off + k] = static_cast<float>(c[off + k] * inv);
            }
        }
        node->set_CPT(c);
    }

    cerr << "Initialized CPTs with prior-centered Dirichlet (ESS=" << prior_ess << ").\n";

    cerr << "Initialized CPTs with prior-centered Dirichlet (ESS=" << prior_ess << ").\n";
}

void initialize_uniform_cpt(network &net)
{
    int N = net.netSize();
    for (int i = 0; i < N; ++i)
    {
        auto node = net.fast_get(i);
        int K = node->get_nvalues();
        long long prod = 1;
        for (int r : node->parent_rad())
        {
            prod *= r;
        }
        int rows = (int)prod;
        vector<float> cpt(rows * K, 1.0f / K);
        node->set_CPT(cpt);
    }
    cerr << "Initialized CPTs with uniform distributions.\n";
}

void run_soft_then_exact_em(network &net, vector<vector<string>> &data, chrono::steady_clock::time_point deadline,
                            int max_iter_soft, double tol,
                            double ESS,
                            double tau_init, double tau_decay, double tau_min)
{
    int N = net.netSize();
    float tiny = 1e-12f;

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
        counts[i].assign((int)prod * node->get_nvalues(), 0.0f);
    }

    auto em_pass = [&](double tau)
    {
        for (size_t ri = 0; ri < data.size(); ++ri)
        {
            auto &row = data[ri];
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
                for (int i = 0; i < N; ++i)
                {
                    auto node = net.fast_get(i);
                    int x = node->vindex_fast(row[i]);
                    if (x < 0)
                        continue;
                    int code = compute_parent_code_fast(net, *node, row);
                    if (code < 0)
                        continue;
                    int base = node->row_base_from_assign_code(code);
                    counts[i][base + x] += 1.0;
                }
            }
            else
            {
                int m = missing_idx;
                auto mnode = net.fast_get(m);

                int m_code = compute_parent_code_fast(net, *mnode, row);
                int m_base = (m_code >= 0) ? mnode->row_base_from_assign_code(m_code) : 0;
                auto &mC = mnode->get_CPT();
                int K = mnode->get_nvalues();

                vector<double> w(K, 0.0);
                double max_lp = -1e300;

                vector<float> prior_fb;
                if (m_code < 0)
                {
                    int total_rows = (int)mC.size() / K;
                    prior_fb.assign(K, 0.0f);
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
                            prior_fb[v] = 1.0f / K;
                    }
                }

                vector<double> lps(K, -1e300);
                for (int v = 0; v < K; ++v)
                {
                    double prior_v = (m_code >= 0) ? max((double)tiny, (double)mC[m_base + v])
                                                   : max((double)tiny, (double)prior_fb[v]);
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
                        int c_code = compute_parent_code_with_override(net, *child, row, m, v);
                        if (c_code < 0)
                        {
                            bad = true;
                            break;
                        }
                        int cb = child->row_base_from_assign_code(c_code);
                        auto &cC = child->get_CPT();
                        logp += log(max((double)tiny, (double)cC[cb + y]));
                    }
                    if (bad)
                    {
                        lps[v] = -1e300;
                        continue;
                    }
                    lps[v] = logp;
                    if (logp > max_lp)
                        max_lp = logp;
                }

                double wsum = 0.0;
                for (int v = 0; v < K; ++v)
                {
                    if (lps[v] <= -1e299)
                    {
                        w[v] = 0.0;
                        continue;
                    }
                    w[v] = exp((lps[v] - max_lp) / max(1.0, tau));
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

                for (int i = 0; i < N; ++i)
                {
                    auto node = net.fast_get(i);

                    if (i == m)
                    {
                        if (m_code >= 0)
                        {
                            int base = m_base;
                            for (int v = 0; v < K; ++v)
                                counts[i][base + v] += (float)w[v];
                        }
                        else
                        {
                            int total_rows = (int)counts[i].size() / K;
                            for (int rr = 0; rr < total_rows; ++rr)
                            {
                                int base = rr * K;
                                for (int v = 0; v < K; ++v)
                                    counts[i][base + v] += (float)(w[v] / max(1, total_rows));
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
                    {
                        int x = node->vindex_fast(row[i]);
                        if (x < 0)
                            continue;
                        for (size_t vv = 0; vv < w.size(); ++vv)
                        {
                            if (w[vv] <= 0.0)
                                continue;
                            int code = compute_parent_code_with_override(net, *node, row, m, (int)vv);
                            if (code < 0)
                                continue;
                            int base = node->row_base_from_assign_code(code);
                            counts[i][base + x] += (float)w[vv];
                        }
                        continue;
                    }

                    int x = node->vindex_fast(row[i]);
                    if (x < 0)
                        continue;
                    int code = compute_parent_code_fast(net, *node, row);
                    if (code < 0)
                        continue;
                    int base = node->row_base_from_assign_code(code);
                    counts[i][base + x] += 1.0f;
                }
            }
        }

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

                double denom = sum_counts + ESS;
                if (denom < tiny)
                    denom = tiny;

                for (int k = 0; k < K; ++k)
                {
                    float num = cpt[off + k] + ESS / K;
                    float p = num / denom;
                    cpt[off + k] = max(tiny, p);
                }
                double s = 0.0;
                for (int k = 0; k < K; ++k)
                    s += cpt[off + k];
                double inv = 1.0 / s;
                for (int k = 0; k < K; ++k)
                    cpt[off + k] = (float)(cpt[off + k] * inv);
            }
            node->set_CPT(cpt);
        }
    };

    double tau = tau_init;
    double prev_ll = -1e300;
    for (int iter = 0; iter < max_iter_soft; ++iter)
    {
        if (chrono::steady_clock::now() > deadline)
        {
            best_model = net;
            out = true;
            cout << "Reached time limit during EM; stopping early at iter " << (iter + 1) << ".\n";
            break;
        }
        em_pass(tau);

        double ll = compute_incomplete_log_likelihood(net, data);
        cout << "EM iter " << (iter + 1) << " | logL=" << ll << "  tau=" << tau << "\n";

        tau = max(tau * tau_decay, tau_min);

        if (fabs(ll - prev_ll) < tol)
        {
            cout << "Soft EM converged at iter " << (iter + 1) << ".\n";
            break;
        }
        prev_ll = ll;
    }
}

void validate_dataset(network &net, vector<vector<string>> &data, int max_report)
{
    int N = net.netSize();
    int bad_rows = 0, bad_tokens = 0, missing_multi = 0;
    for (int r = 0; r < (int)data.size(); ++r)
    {
        auto &row = data[r];
        if ((int)row.size() != N)
        {
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
            auto vals = node->get_values();
            string &tok = row[i];
            if (tok == "?")
            {
                ++q;
                continue;
            }
            bool ok = false;
            for (size_t vi = 0; vi < vals.size(); ++vi)
                if (vals[vi] == tok)
                {
                    ok = true;
                    break;
                }
            if (!ok)
            {
                if (++bad_tokens <= max_report)
                    cerr << "[TOKEN] r=" << r << " col=" << i
                         << " node=" << node->get_name()
                         << " value='" << tok << "' not in node's domain\n";
            }
        }
        if (q > 1)
        {
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
int main(int argv,char** argc)
{
    auto start_time = chrono::steady_clock::now();
    auto deadline = start_time + chrono::minutes(2) - chrono::seconds(5);
    char* bif_path = argc[1];
    char* datapath = argc[2];
    network BayesNet0 = read_network(bif_path);
    vector<vector<string>> dataset = load_records_csv(datapath);

    cout << "Loaded " << dataset.size() << " records." << endl;
    validate_dataset(BayesNet0, dataset, 10);
    int N = BayesNet0.netSize();

    int NUM_TRIALS = 1;
    double PRIOR_ESS = 5.0;
    double MSTEP_ESS = 2.0;

    vector<double> LLs;
    LLs.reserve(NUM_TRIALS);
    vector<vector<vector<float>>> all_cpts;
    all_cpts.resize(NUM_TRIALS, vector<vector<float>>(N));

    double best_ll = -1e300;
    int best_idx = -1;

    for (int trial = 0; trial < NUM_TRIALS; ++trial)
    {
        cout << "\n=== Trial " << (trial + 1) << "/" << NUM_TRIALS << " ===\n";
        network model = read_network(bif_path);

        initialize_cpts_with_prior(model, dataset, PRIOR_ESS);

        run_soft_then_exact_em(model, dataset, deadline, 60, 1e-5, MSTEP_ESS, 2.0, 0.90, 1.0);

        best_model = model;
        if (!out)
            initialize_uniform_cpt(model);

        run_soft_then_exact_em(model, dataset, deadline, 30, 1e-5, MSTEP_ESS, 1.0, 0.95, 1.0);

        model = best_model;
        double ll = compute_incomplete_log_likelihood(model, dataset);
        cout << "Trial " << (trial + 1) << " final logL = " << ll << endl;

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

    network finalNet = read_network(bif_path);
    for (int i = 0; i < N; ++i)
    {
        vector<float> avg = all_cpts[best_idx][i];
        fill(avg.begin(), avg.end(), 0.0f);
        for (int t = 0; t < NUM_TRIALS; ++t)
        {
            auto &cpt = all_cpts[t][i];
            for (size_t j = 0; j < avg.size(); ++j)
                avg[j] += (float)(w[t] * cpt[j]);
        }
        int K = finalNet.fast_get(i)->get_nvalues();
        for (int off = 0; off < (int)avg.size(); off += K)
        {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += avg[off + k];
            if (s < 1e-12)
            {
                for (int k = 0; k < K; ++k)
                    avg[off + k] = static_cast<float>(1.0 / K);
            }
            else
            {
                double inv = 1.0 / s;
                for (int k = 0; k < K; ++k)
                    avg[off + k] = static_cast<float>(avg[off + k] * inv);
            }
        }
        finalNet.fast_get(i)->set_CPT(avg);
    }

    cout << "\nBest individual trial logL: " << best_ll
         << " (trial " << (best_idx + 1) << ")\n";

    double avg_ll = compute_incomplete_log_likelihood(finalNet, dataset);
    cout << "Averaged model logL: " << avg_ll << "\n";

    write_network("solved.bif", finalNet);
    write_network("solved.bif", finalNet);
    return 0;
}
#endif
