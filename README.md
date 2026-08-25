# Bayesian Network Parameter Learning (Hailfinder)

Learns the missing conditional probability tables (CPTs) of a Bayesian
network from partially-observed data, using an annealed EM algorithm.
Written for the Hailfinder weather-forecasting network as a course
assignment (see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)).

## Problem

You're given:

- `hailfinder.bif` — the network structure (nodes, parents, value domains)
  with some CPT entries replaced by `-1`, meaning "unknown, learn me".
- `records.dat` — a CSV-style dataset over the network's variables, where
  some cells are `?` (missing at random, at most one missing value per row).

The task is to fill in the `-1` entries so the resulting CPTs are valid
probability distributions that explain the data well, and write the result
to `solved.bif`.

## Approach

All of the logic lives in [startup_code.cpp](startup_code.cpp):

1. **Parse** the `.bif` network and the CSV dataset (`read_network`,
   `load_records_csv`), tracking which CPT entries are fixed vs. learnable.
2. **Initialize** the learnable entries from a Dirichlet-style prior blended
   with complete-case counts from the data (`initialize_cpts_with_prior`),
   rather than starting from a uniform distribution.
3. **Expectation-Maximization** (`run_soft_then_exact_em`): for each row with
   a missing value, compute the posterior over that value given its Markov
   blanket, accumulate soft counts, and re-normalize the CPTs (M-step). The
   E-step uses a temperature parameter that anneals from soft/smoothed
   assignments down towards hard assignments across iterations, which helps
   the optimizer avoid poor local optima early on before committing to sharp
   probabilities later.
4. **Multiple trials + weighting**: several EM runs (different
   initializations) are scored by incomplete-data log-likelihood
   (`compute_incomplete_log_likelihood`) and combined into a final CPT set.
5. The whole run is bounded by a wall-clock deadline (2 minutes minus a
   safety margin) so it always terminates with a usable answer.
6. **Write** the learned network back out to `solved.bif`.

[Format_Checker.cpp](Format_Checker.cpp) is a standalone tool that validates
the output (every row sums to 1, no `-1` left, structure unchanged from the
input skeleton) and reports the total absolute CPT error against a
ground-truth network, `gold_hailfinder.bif`.

## Repository layout

```
startup_code.cpp   Main program: parsing, EM training, writing solved.bif
Format_Checker.cpp Validates solved.bif and scores it against the gold network
compile.sh         Builds both of the above
run.sh             Runs the solver then the checker on the sample case
tester.py          Runs the full pipeline across every case in test_case_checker/tc
test_case_checker/ Sample datasets (tc1 .. tc10, plus "given") for tester.py
hailfinder.bif, gold_hailfinder.bif, records.dat
                   The default ("given") sample case, usable directly at the repo root
```

## Usage

Build the solver and checker:

```bash
bash compile.sh
```

Run on the sample case (produces `solved.bif` in the repo root):

```bash
bash run.sh hailfinder.bif records.dat
```

Run the full test suite (compiles fresh, runs every case under
`test_case_checker/tc/`, and prints a pass/fail summary with per-case
timing and total CPT error):

```bash
python tester.py
```

## Acknowledgments

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for the course-required
collaboration disclosure.
