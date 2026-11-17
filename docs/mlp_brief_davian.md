# MLP brief (Davian): instructions for Claude Code

Build my main model for PR1: a **plain MLP (PyTorch)** that predicts `target` (1 = the customer buys between
2024-10-01 and 2024-10-30). Every decision below comes from my EDA in `notebooks/02_eda_davian.ipynb` and is final.
Follow the repo rules in `README.md` and `notebooks/README.md`.

## Step 0: read before coding
Read `README.md`, `notebooks/README.md`, `src/data_prep.py`, `src/shopsphere/config.py`, `src/shopsphere/split.py`,
`01_baselines.ipynb` (skim), `results/baseline_results.csv`, and **Section 8 of `notebooks/02_eda_davian.ipynb`**.
Then tell me in 3–5 lines what you'll build before writing code.

## Data: already prepared, so don't redo it
Running `notebooks/02_eda_davian.ipynb` (Section 8) creates `data/processed/mlp_prepared/`:

| File | Rows | Contents |
|---|---|---|
| `train_prepared.csv` | 44,800 | 77 scaled inputs + `customer_id` + `target` (team **train** split) |
| `val_prepared.csv` | 9,600 | same (team **val** split) |
| `test_prepared.csv` | 9,600 | same (team **test** split: use once, at the end) |
| `submission_prepared.csv` | 16,000 | 77 inputs + `customer_id`, no target (`test_features.csv`) |
| `mlp_preprocessor.joblib` | | the fitted pipeline (fitted on train only) |

- These use the team's one shared split (`data_prep.get_splits`, 70/15/15, stratified, seed 42), the same rows as
  `results/baseline_results.csv`. **Don't make a new split.** Buy rate is 14.57% in every part.
- Inputs are already log-transformed and scaled, missing `avg_csat` is filled with a flag, duplicates are dropped and
  categoricals are one-hot encoded. **Don't add another scaler**, and drop `customer_id` and `target` before training.
- If the folder is missing, run `02_eda_davian.ipynb` first.

## Decisions from my EDA that the model must follow
| EDA finding | What the code must do |
|---|---|
| Only 14.6% buy | Never use accuracy as the headline. Use `BCEWithLogitsLoss(pos_weight=5.86)` (negatives ÷ positives in train). |
| Rare target | **Main metric: PR-AUC**, with ROC-AUC, precision, recall, F1, precision@top10% and lift. Use `data_prep.evaluate()` so the numbers match the team's. |
| 0.5 isn't the right cut-off | Choose the threshold on **val** with `data_prep.best_f1_threshold()`, and reuse it on test. |
| Recency is today's CRM rule | Targets to beat (val PR-AUC): recency rule **0.366**, logistic regression **0.562**, LightGBM **0.563**. |
| Country barely matters (13.8–15.0%) | After training, report recall and precision per `country_code` and per age group on test (Ethical AI). Get the raw `country_code`, `age` and `age_missing` by joining `customer_id` to `data/processed/train_features.csv`. |
| Leakage was removed in cleaning | Never add raw `delivery_status` or any data after 2024-09-30. |
| `loyalty_score` is the top feature but opaque | In explanations, cite the behaviour behind it (recent orders/visits), not just "high loyalty score". |

## Build `notebooks/04_mlp_davian.ipynb`
Each section gets a short markdown cell explaining **what and why in simple words**. I must explain every line in
a live walkthrough with notes closed, so keep the code simple and readable over clever.

1. **Setup:** find the project root like `01_baselines.ipynb` does, import from `data_prep` and `shopsphere.config`
   (`SEED`, `MODELS_DIR`, `FIGURES_DIR`), fix all seeds (numpy, torch, random), choose CPU/GPU, load the prepared CSVs,
   and confirm shapes and buy rate.
2. **Baselines:** load `results/baseline_results.csv` and show the val rows as the targets to beat.
3. **Baseline MLP:** 77 → 128 ReLU → Dropout 0.3 → 64 ReLU → Dropout 0.3 → 1 logit; Adam lr 1e-3; batch 256;
   `pos_weight` 5.86; early stopping on **val PR-AUC** (patience 10, restore the best weights); max ~100 epochs.
   Write one reusable `train_model(config)` function so tuning is just a loop.
4. **Systematic tuning (evidence of optimisation):** a small, documented search (about 12–20 runs, not a huge grid)
   over layers/units, dropout, weight decay (L2), learning rate and batch size. Change one or two things at a time and
   say why. Log every run to `results/mlp_tuning_davian.csv` (config, best epoch, val PR-AUC, ROC-AUC, F1, train time)
   and show it sorted. Optuna is in the dependencies and is optional, but keep it explainable.
5. **Final model:** retrain the best config with a fixed seed, then plot train/val loss curves, PR curves of the MLP
   vs logistic regression and LightGBM on val (refit those two with the team preprocessor like `01_baselines.ipynb`,
   or just compare against the PR-AUC numbers if refitting is heavy), and the confusion matrix.
6. **Threshold:** `best_f1_threshold` on val, plus a threshold vs precision/recall plot. Explain it in plain words,
   e.g. "at threshold X we catch Y% of buyers, and Z% of the customers we flag really buy".
7. **Test, once:** `evaluate()` on the test split with that threshold. Save the val and test rows in the same columns as
   `results/baseline_results.csv` to `results/mlp_results_davian.csv` (**don't overwrite the team's file**), and show
   one comparison table of every model against every metric.
8. **Fairness:** recall and precision per `country_code` and per age group (18–24, 25–34, 35–44, 45–54, 55+, and
   "unknown" where `age_missing` = 1) on test, with a short comment on any gap.
9. **Save:** the model to `models/mlp_davian.pt` (state_dict + config + threshold), figures to
   `reports/figures/mlp_davian/`, and submission probabilities for the 16,000 customers to
   `results/mlp_submission_davian.csv` (`customer_id,target`, the same format as `data/raw/sample_submission.csv`).
10. **Summary:** five bullets covering the best config, val/test PR-AUC and F1 vs the baselines, the threshold, any
    fairness concerns and the next steps.

## Rules
- Use the repo's `.venv` (`uv sync` installs CPU PyTorch on Windows). If something is missing, tell me the exact
  command and wait for my approval before installing.
- Never fit anything on val or test. Val is only for early stopping, tuning and the threshold, and test is used once.
- Don't edit teammates' files (`01_baselines.ipynb`, `src/data_prep.py`, `results/baseline_results.csv`).
- Restart and run all at the end, fix every error, and only then tell me it's done with the summary from step 10.
- Don't commit or push; I'll commit myself.
