# KNIME workflows

Data cleaning and feature building for ShopSphere were done in KNIME (Analytics Platform 5.12)
by Davian. Each file is a standard KNIME export: in KNIME, use **File > Import KNIME Workflow**
and pick the `.knwf`.

| File | What it does | Reads | Writes |
|---|---|---|---|
| `ShopSphere_Cleaning_final_v1.0.knwf` | Cleans each of the 17 data-pack tables, one metanode per table: types, true/false encodings, numbers stored as text, duplicates, missing values, and the 510 `delivered_date` values after the 30 Sep 2024 cutoff | The raw data pack (`customers.csv`, `transactions.csv`, `sessions.csv`, ...) | `<table>_clean.csv`, one per table |
| `ShopSphere_Features.knwf` | Joins the cleaned tables into one row per customer, with recency, frequency, session, delivery, ticket and contact features dated on or before 30 Sep 2024 | The `*_clean.csv` tables | `features.csv`, `train_features.csv`, `test_features.csv` |
| `ShopSphere_EDA_Davian.knwf` | First-pass EDA on the feature table: statistics, value counts and bar charts by target. The full EDA is in `notebooks/02_eda_davian.ipynb` | `features.csv`, `train_features.csv` | Views only |

Run order: Cleaning, then Features, then EDA.

## Before running

- **The workflows were exported reset, without data.** Every node is configured but not
  executed, so the files are small (under 300 KB) and contain none of the data pack. Run them
  to regenerate the outputs.
- **File paths are absolute** and point to Davian's laptop (`C:\Users\Davian\OneDrive\School\AIDL\Data\`
  for inputs, `...\AIDL\cleaned data\` for outputs). On another machine, open each CSV Reader and
  CSV Writer and point it at your copy of `data/raw/` and `data/interim/`.
- Earlier drafts (`ShopSphere_Cleaning`, `ShopSphere_Cleaning_import`, `ShopSphere_Cleaning_final_v1.1`)
  are not included. `ShopSphere_Cleaning_final_v1.0` is the version whose outputs the features
  workflow and notebooks use.
