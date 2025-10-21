// Pull in implementations from startup_code.cpp but disable its own main
#define BN_LIB
#include "startup_code.cpp"

int main() {
    // Use existing dataset files in the workspace
    network BayesNet = read_network("hailfinder.bif");
    vector<vector<string>> dataset = load_records_csv("records.dat");

    cout << "Loaded " << dataset.size() << " records." << endl;
    validate_dataset(BayesNet, dataset);
        initialize_cpts_complete_case(BayesNet, dataset);

    run_hard_em(BayesNet, dataset, 5);
    write_network("solved.bif", BayesNet);
    return 0;
}
