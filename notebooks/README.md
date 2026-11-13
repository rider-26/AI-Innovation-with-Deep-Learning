# Notebooks

Name each notebook `NN_topic_owner.ipynb` so the order and the author are obvious:

| Notebook | Contents |
|---|---|
| `01_data_audit_<owner>.ipynb` | Types, true/false encodings, numbers stored as text, duplicates, missing values, outliers, join keys |
| `02_eda_<owner>.ipynb` | EDA with plots; the hypotheses in the project plan (Phase 2) |
| `03_baselines_<owner>.ipynb` | Always-0, recency rule, logistic regression, LightGBM |
| `04_mlp_<owner>.ipynb` | One per member: your own deep learning variant, tuning log, evaluation |
| `05_comparison.ipynb` | One table: every model against every metric on the test split |

Rules:

- Import the seed, paths and split from `shopsphere.config` and `shopsphere.split`. Don't redefine them.
- Write markdown as you go: what you are checking, what you found, what you decided. Source Codes
  (5 marks at PR1 and PR2) is marked on structure and explanations.
- Fit imputers and scalers on the train split only. Tune on validation. Touch the test split once.
- Restart and run all before committing, so the saved outputs match the code.
