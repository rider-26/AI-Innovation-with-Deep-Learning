# Task board

Starter backlog taken from the project plan. Move these into GitHub Projects (or keep this file
current) and put a name and a date on each one. Unowned tasks don't get done.

## Now: before Progress Review 1 (Week 5, w/c 16 Nov 2026, 20%)

### Team set-up

- [ ] Elect a team leader; fill in the team table in the README
- [ ] Every member clones the repo and makes a first commit from their own GitHub account
      (Davian: done, `0822d10`, 13 Nov)
- [ ] Every member creates their AI Use Log in `docs/ai-use-log/`
      (Davian: done, `docs/ai-use-log/dav_1ann.md`)
- [ ] Every member starts a running reflection log (Reflection is 10 marks at the Final)
- [ ] Email Mr Wee the open questions (project plan, Section 0): which dates apply, the PR1
      submission time, how individual marking works on a shared solution, whether PR-AUC or F1
      counts as the "industry benchmark", approved LLM APIs and budget
- [ ] Agree who owns which model variant and which agent tool

### Data (PR1 Input Data, 5 marks)

- [ ] Put the data pack in `data/raw/` and confirm everyone has identical files
- [x] Audit every table: types, true/false encodings, numbers stored as text, duplicates, missing
      values, outliers, join keys, `noise_*` columns (Davian, KNIME
      `knime/ShopSphere_Cleaning_final_v1.0.knwf`)
- [x] Clean tables into `data/interim/` (blank the 510 `delivered_date` values after the cutoff;
      never use the original `delivery_status` as a feature) (Davian, KNIME, 17 `*_clean.csv` tables)
- [ ] Check the pre-aggregated columns in `customers.csv` (`loyalty_score`, `email_open_rate`,
      `discount_usage_rate`, `avg_review_score`) for post-cutoff information
- [x] Build `data/processed/features.csv`: one row per customer for all 80,000 customers, nothing
      dated after 30 Sep 2024, keep average order value (Davian, KNIME
      `knime/ShopSphere_Features.knwf`)
- [x] EDA notebook with plots and written findings (Davian, `notebooks/02_eda_davian.ipynb`, `22a9a37`)

### Model (PR1 Supervised Model, 5 marks)

- [x] Baselines: always-0, recency rule, logistic regression, LightGBM (Davian, `01_baselines.ipynb`,
      `e6e97b3`, 16 Nov)
- [ ] Each member: own deep learning variant (plain MLP / MLP with embeddings / GRU or LSTM over
      weekly activity / wide-and-deep or TabNet)
      (Davian: plain MLP done, `notebooks/04_mlp_davian.ipynb`, `a356bd0`)
- [ ] Tuning log for every run: this is the evidence of optimisation
      (Davian: done, `results/mlp_tuning_davian.csv`, 5 stages plus a 3-seed repeat of the top 3)
- [ ] Evaluation on the shared split: PR-AUC (main), ROC-AUC, precision, recall, F1, confusion
      matrix, precision in the top 10%. Accuracy only next to the 85.4% floor
      (Davian: done, `results/mlp_results_davian.csv`. Test PR-AUC 0.574, a tie with LightGBM 0.576
      and logistic regression 0.577)
- [ ] Leakage checks: shuffled-label test, no single dominant feature, scaler and imputer fitted
      on train only
      (Davian: scalers and imputer fitted on train only. Shuffled-label test still to do)

### PR1 presentation (group, 5 marks) and live defence

- [ ] Progress slides (Davian: own slides done)
- [ ] Mock walkthrough: each member opens a random file of theirs and explains what it does, what
      depends on it and what breaks without it, with notes and AI closed

## Next: before Progress Review 2 (Week 14, w/c 18 Jan 2027, 30%)

- [ ] Agree who owns which agent tool and fill in the README team table (Davian: not yet assigned)
- [ ] Davian: add the shuffled-label leakage check to the MLP notebook
- [ ] Davian: look into the MLP's lower recall for customers aged 55+ (55% against 73% for 18–24)

- [ ] Calibrate probabilities; set band cut-offs on validation
- [ ] `predict(customer_ids)` returning probability, band and top reasons, with unit tests
- [ ] Policy, offer, cost and budget functions as plain Python with unit tests
- [ ] Agent in LangGraph that calls those tools
- [ ] App: weekly list review and approval, customer detail, ask the agent, dashboard (W6)
- [ ] Scenario test suite (project plan, Section 5.2)
- [ ] Model behind a FastAPI endpoint, in Docker, deployed remotely (needed for the A band)
- [ ] Mock interview (10%): date to confirm

## Later: Final (Week 18/19, w/c 15 Feb 2027, 40%)

- [ ] Report (max 25 pages), 20-minute demo where every member presents
- [ ] Ethical AI Checklist, individual reflection, peer review, final AI Use Log
- [ ] Reproducible repo: a fresh clone runs from the README
