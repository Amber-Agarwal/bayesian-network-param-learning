// Orchestrated hybrid pipeline combining Soft EM, Hard EM, Multiple Imputation, and Variational Bayes
// Robust against invalid CPT rows (-1 / zeros) -> no accidental uniformization.

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

using std::string;
using std::vector;

// ------- Import Soft-EM implementation (macro-renamed) -------
#define Graph_Node Soft_Graph_Node
#define network Soft_network
#define read_network Soft_read_network
#define write_network Soft_write_network
#define read_data Soft_read_data
#define load_records_csv Soft_load_records_csv
#define validate_dataset Soft_validate_dataset
#define initialize_cpts_complete_case Soft_initialize_cpts_complete_case
#define run_soft_em Soft_run_soft_em
#define get_cpt_index Soft_get_cpt_index
#define value_index Soft_value_index
#define trim Soft_trim
#define rtrim_cr Soft_rtrim_cr
#define strip_quotes Soft_strip_quotes
#define parse_csv_line Soft_parse_csv_line
#define BN_LIB
#include "startup_code_soft.cpp"
#undef BN_LIB
#undef Graph_Node
#undef network
#undef read_network
#undef write_network
#undef read_data
#undef load_records_csv
#undef validate_dataset
#undef initialize_cpts_complete_case
#undef run_soft_em
#undef get_cpt_index
#undef value_index
#undef trim
#undef rtrim_cr
#undef strip_quotes
#undef parse_csv_line

// ------- Import Hard-EM implementation (macro-renamed) -------
#define Graph_Node Hard_Graph_Node
#define network Hard_network
#define read_network Hard_read_network
#define write_network Hard_write_network
#define read_data Hard_read_data
#define load_records_csv Hard_load_records_csv
#define validate_dataset Hard_validate_dataset
#define initialize_cpts_complete_case Hard_initialize_cpts_complete_case
#define run_hard_em Hard_run_hard_em
#define get_cpt_index Hard_get_cpt_index
#define value_index Hard_value_index
#define trim Hard_trim
#define rtrim_cr Hard_rtrim_cr
#define strip_quotes Hard_strip_quotes
#define parse_csv_line Hard_parse_csv_line
#define BN_LIB
#include "startup_code.cpp"
#undef BN_LIB
#undef Graph_Node
#undef network
#undef read_network
#undef write_network
#undef read_data
#undef load_records_csv
#undef validate_dataset
#undef initialize_cpts_complete_case
#undef run_hard_em
#undef get_cpt_index
#undef value_index
#undef trim
#undef rtrim_cr
#undef strip_quotes
#undef parse_csv_line

// ------- Import Multiple-Imputation implementation (macro-renamed) -------
#define Graph_Node MI_Graph_Node
#define network MI_network
#define read_network MI_read_network
#define write_network MI_write_network
#define read_data MI_read_data
#define load_records_csv MI_load_records_csv
#define validate_dataset MI_validate_dataset
#define run_multiple_imputation_em MI_run_multiple_imputation_em
#define trim MI_trim
#define rtrim_cr MI_rtrim_cr
#define strip_quotes MI_strip_quotes
#define parse_csv_line MI_parse_csv_line
#define value_index MI_value_index
#define BN_LIB
#include "startup_code_mi.cpp"
#undef BN_LIB
#undef Graph_Node
#undef network
#undef read_network
#undef write_network
#undef read_data
#undef load_records_csv
#undef validate_dataset
#undef run_multiple_imputation_em
#undef trim
#undef rtrim_cr
#undef strip_quotes
#undef parse_csv_line
#undef value_index

// ------- Import Variational-Bayes implementation (macro-renamed) -------
#define Graph_Node VB_Graph_Node
#define network VB_network
#define read_network VB_read_network
#define write_network VB_write_network
#define read_data VB_read_data
#define load_records_csv VB_load_records_csv
#define validate_dataset VB_validate_dataset
#define initialize_cpts_complete_case VB_initialize_cpts_complete_case
#define run_variational_bayes VB_run_variational_bayes
#define trim VB_trim
#define rtrim_cr VB_rtrim_cr
#define strip_quotes VB_strip_quotes
#define parse_csv_line VB_parse_csv_line
#define value_index VB_value_index
#define BN_LIB
#include "startup_code_vb.cpp"
#undef BN_LIB
#undef Graph_Node
#undef network
#undef read_network
#undef write_network
#undef read_data
#undef load_records_csv
#undef validate_dataset
#undef initialize_cpts_complete_case
#undef run_variational_bayes
#undef trim
#undef rtrim_cr
#undef strip_quotes
#undef parse_csv_line
#undef value_index

// ---------- Shared data helpers ----------
static int count_missing(const vector<string>& row) {
    int c = 0; for (const auto& t : row) if (t == "?") ++c; return c;
}
static vector<vector<string>> filter_rows_leq1_missing(const vector<vector<string>>& data) {
    vector<vector<string>> out; out.reserve(data.size());
    for (const auto& r : data) if (count_missing(r) <= 1) out.push_back(r);
    return out;
}
static bool is_fully_observed(const vector<string>& row) {
    for (const auto& t : row) if (t == "?") return false; return true;
}
static vector<vector<string>> build_validation_set(const vector<vector<string>>& data, double frac, int expected_cols) {
    size_t target = static_cast<size_t>(data.size() * frac);
    vector<vector<string>> val; val.reserve(target);
    for (const auto& r : data) {
        if ((int)r.size() != expected_cols) continue;
        if (!is_fully_observed(r)) continue;
        val.push_back(r);
        if (val.size() >= target) break;
    }
    return val;
}

// ---------- CPT sanitization & smoothing ----------
static void sanitize_network(Hard_network& net, float eps=1e-12f) {
    const int N = net.netSize();
    for (int i = 0; i < N; ++i) {
        auto node = net.get_nth_node(i);
        auto cpt  = node->get_CPT();
        const int K = node->get_nvalues();
        if (K <= 0) continue;
        for (int off = 0; off < (int)cpt.size(); off += K) {
            bool bad = false;
            for (int k = 0; k < K; ++k) {
                float &x = cpt[off + k];
                if (!std::isfinite(x) || x < 0.0f) { x = 0.0f; bad = true; }
            }
            float s = 0.0f; for (int k = 0; k < K; ++k) s += cpt[off + k];
            if (s <= eps) {
                // No signal: uniform
                for (int k = 0; k < K; ++k) cpt[off + k] = 1.0f / K;
            } else {
                float inv = 1.0f / s;
                for (int k = 0; k < K; ++k) cpt[off + k] *= inv;
            }
        }
        node->set_CPT(cpt);
    }
}
static void smooth_cpts_uniform(Hard_network& net, float lambda = 0.005f) {
    const int N = net.netSize();
    for (int i = 0; i < N; ++i) {
        auto node = net.get_nth_node(i);
        auto cpt = node->get_CPT();
        const int K = node->get_nvalues();
        for (int off = 0; off < (int)cpt.size(); off += K) {
            for (int k = 0; k < K; ++k) {
                float uniform = 1.0f / K;
                cpt[off + k] = (1.0f - lambda) * cpt[off + k] + lambda * uniform;
            }
            float s = 0.0f; for (int k = 0; k < K; ++k) s += cpt[off + k];
            if (s > 0.0f) { float inv = 1.0f / s; for (int k = 0; k < K; ++k) cpt[off + k] *= inv; }
        }
        node->set_CPT(cpt);
    }
}

// ---------- Validation LL ----------
static double compute_log_likelihood(Hard_network& net, const vector<vector<string>>& val) {
    const int N = net.netSize();
    const double tiny = 1e-12;
    double ll = 0.0;
    for (const auto& row : val) {
        if ((int)row.size() != N) continue;
        double s = 0.0;
        for (int i = 0; i < N; ++i) {
            auto node = net.get_nth_node(i);
            int x = Hard_value_index(node->get_values(), row[i]);
            if (x < 0) { s += std::log(tiny); continue; }
            vector<string> parents = node->get_Parents();
            vector<int> pidx; pidx.reserve(parents.size());
            for (const auto& p : parents) {
                int pi = net.get_index(p);
                if (pi < 0) { pidx.clear(); break; }
                int v = Hard_value_index(net.get_nth_node(pi)->get_values(), row[pi]);
                if (v < 0) { pidx.clear(); break; }
                pidx.push_back(v);
            }
            if (!parents.empty() && pidx.empty()) { s += std::log(tiny); continue; }
            int base = Hard_get_cpt_index(net, *node, pidx);
            const auto cpt = node->get_CPT();
            double p = (base + x >= 0 && base + x < (int)cpt.size()) ? cpt[base + x] : 0.0;
            s += std::log(std::max(p, tiny));
        }
        ll += s;
    }
    return ll;
}

int main() {
    const char* bif_in   = "hailfinder.bif";
    const char* data_in  = "records.dat";
    const char* bif_soft = "soft_ind.bif";
    const char* bif_hard = "hard_ind.bif";
    const char* bif_mi   = "mi_ind.bif";
    const char* bif_vb   = "vb_ind.bif";
    const char* bif_out  = "solved_mix.bif";

    // Load dataset once (whitespace separated to match your parsers)
auto data = Hard_load_records_csv(data_in);
    std::cout << "Loaded " << data.size() << " records." << std::endl;

    // Build validation set of fully observed rows (15%)
    int Ncols = 0; { Hard_network tmp = Hard_read_network(bif_in); Ncols = tmp.netSize(); }
    auto val = build_validation_set(data, 0.15, Ncols);
    if (val.empty()) std::cout << "[mix] Warning: validation set is empty; weights will be equal.\n";

    // --- SOFT EM ---
    {
        Soft_network net = Soft_read_network(bif_in);
        Soft_initialize_cpts_complete_case(net, data);
        Soft_run_soft_em(net, data, /*iters=*/10);
        // sanitize then write
        Hard_network tmp = Hard_read_network(bif_in);
        // copy CPTs across for sanitization via Hard_network
        for (int i=0;i<tmp.netSize();++i) tmp.get_nth_node(i)->set_CPT( Soft_read_network(bif_in).get_nth_node(i)->get_CPT() );
        // reload trained net and sanitize
        Hard_network hs = Hard_read_network(bif_in);
        for (int i=0;i<hs.netSize();++i) hs.get_nth_node(i)->set_CPT( net.get_nth_node(i)->get_CPT() );
        sanitize_network(hs);
        Hard_write_network(bif_soft, hs);
        std::cout << "[mix] Wrote Soft-EM model to " << bif_soft << std::endl;
    }

    // --- HARD EM (≤1-missing rows, Laplace init) ---
    {
        auto data_leq1 = filter_rows_leq1_missing(data);
        Hard_network net = Hard_read_network(bif_in);
        Hard_initialize_cpts_complete_case(net, data_leq1);
        Hard_run_hard_em(net, data_leq1, /*iters=*/10);
        sanitize_network(net);
        Hard_write_network(bif_hard, net);
        std::cout << "[mix] Wrote Hard-EM model to " << bif_hard << std::endl;
    }

    // --- MULTIPLE IMPUTATION EM ---
    {
        MI_network net = MI_read_network(bif_in);
        MI_run_multiple_imputation_em(net, data,
                                      /*em_iters=*/6, /*M=*/8,
                                      /*burnin=*/2, /*sweeps=*/1,
                                      /*pseudo=*/1e-3f);
        // sanitize via Hard_network adapter
        Hard_network hm = Hard_read_network(bif_in);
        for (int i=0;i<hm.netSize();++i) hm.get_nth_node(i)->set_CPT( net.get_nth_node(i)->get_CPT() );
        sanitize_network(hm);
        Hard_write_network(bif_mi, hm);
        std::cout << "[mix] Wrote MI-EM model to " << bif_mi << std::endl;
    }

    // --- VARIATIONAL BAYES ---
    {
        VB_network net = VB_read_network(bif_in);
        VB_initialize_cpts_complete_case(net, data);
        VB_run_variational_bayes(net, data, /*vb_iters=*/8, /*local_sweeps=*/2, /*alpha0=*/1.0);
        // sanitize via Hard_network adapter
        Hard_network hv = Hard_read_network(bif_in);
        for (int i=0;i<hv.netSize();++i) hv.get_nth_node(i)->set_CPT( net.get_nth_node(i)->get_CPT() );
        sanitize_network(hv);
        Hard_write_network(bif_vb, hv);
        std::cout << "[mix] Wrote VB model to " << bif_vb << std::endl;
    }

    // --- Evaluate on validation set ---
    Hard_network eval_soft = Hard_read_network(bif_soft); sanitize_network(eval_soft);
    Hard_network eval_hard = Hard_read_network(bif_hard); sanitize_network(eval_hard);
    Hard_network eval_mi   = Hard_read_network(bif_mi);   sanitize_network(eval_mi);
    Hard_network eval_vb   = Hard_read_network(bif_vb);   sanitize_network(eval_vb);

    double ll_soft = val.empty() ? 0.0 : compute_log_likelihood(eval_soft, val);
    double ll_hard = val.empty() ? 0.0 : compute_log_likelihood(eval_hard, val);
    double ll_mi   = val.empty() ? 0.0 : compute_log_likelihood(eval_mi,   val);
    double ll_vb   = val.empty() ? 0.0 : compute_log_likelihood(eval_vb,   val);
    std::cout << "[mix] Validation LLs => soft=" << ll_soft << " hard=" << ll_hard << " mi=" << ll_mi << " vb=" << ll_vb << std::endl;

    // --- Stabilized weights ---
    double m = std::max(std::max(ll_soft, ll_hard), std::max(ll_mi, ll_vb));
    double w1 = std::exp(ll_soft - m);
    double w2 = std::exp(ll_hard - m);
    double w3 = std::exp(ll_mi   - m);
    double w4 = std::exp(ll_vb   - m);
    double ws = w1 + w2 + w3 + w4;
    if (!(ws > 0.0)) { w1 = w2 = w3 = w4 = 1.0; ws = 4.0; }
    w1 /= ws; w2 /= ws; w3 /= ws; w4 /= ws;
    std::cout << "[mix] Weights => soft=" << w1 << " hard=" << w2 << " mi=" << w3 << " vb=" << w4 << std::endl;

    // --- Blend CPTs per row, skipping invalid model rows ---
    Hard_network blended = Hard_read_network(bif_in);
    const int N = blended.netSize();
    for (int i = 0; i < N; ++i) {
        auto node      = blended.get_nth_node(i);
        const int K    = node->get_nvalues();

        auto cpt_soft  = eval_soft.get_nth_node(i)->get_CPT();
        auto cpt_hard  = eval_hard.get_nth_node(i)->get_CPT();
        auto cpt_mi    = eval_mi.get_nth_node(i)->get_CPT();
        auto cpt_vb    = eval_vb.get_nth_node(i)->get_CPT();
        size_t L = node->get_CPT().size();

        if (cpt_soft.size()!=L || cpt_hard.size()!=L || cpt_mi.size()!=L || cpt_vb.size()!=L) {
            std::cerr << "[mix] CPT size mismatch at node " << i << " (skipping blend for this node).\n";
            continue;
        }

        vector<float> out(L, 0.0f);
        for (size_t off = 0; off < L; off += K) {
            // check each model's row validity
            auto row_ok = [&](const vector<float>& cpt)->bool{
                double s=0.0; for (int k=0;k<K;++k) {
                    float x = cpt[off+k];
                    if (!std::isfinite(x) || x < 0.0f) return false;
                    s += x;
                }
                return s > 1e-15;
            };
            bool ok1=row_ok(cpt_soft), ok2=row_ok(cpt_hard), ok3=row_ok(cpt_mi), ok4=row_ok(cpt_vb);
            double ww = 0.0;
            for (int k=0;k<K;++k) {
                double v = 0.0;
                if (ok1) v += w1 * cpt_soft[off+k];
                if (ok2) v += w2 * cpt_hard[off+k];
                if (ok3) v += w3 * cpt_mi[off+k];
                if (ok4) v += w4 * cpt_vb[off+k];
                out[off + k] = (float)std::max(0.0, v);
                ww += out[off + k];
            }
            if (ww <= 1e-15) {
                // all invalid for this row -> uniform fallback only for this row
                for (int k=0;k<K;++k) out[off+k] = 1.0f / K;
            } else {
                float inv = (float)(1.0 / ww);
                for (int k=0;k<K;++k) out[off+k] *= inv;
            }
        }
        node->set_CPT(out);
    }

    // Final very light smoothing + sanitize once more
    smooth_cpts_uniform(blended, /*lambda=*/0.0025f);
    sanitize_network(blended);
    Hard_write_network(bif_out, blended);
    std::cout << "[mix] Wrote ensembled output to " << bif_out << std::endl;
    return 0;
}
