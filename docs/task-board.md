# Task board

Starter backlog taken from the project plan. Move these into GitHub Projects (or keep this file
current) and put a name and a date on each one. Unowned tasks don't get done.

## Now: before Progress Review 1 (Week 5, w/c 16 Nov 2026, 20%)

### Team set-up

- [ ] Elect a team leader; fill in the team table in the README
- [ ] Every member clones the repo and makes a first commit from their own GitHub account
- [ ] Every member creates their AI Use Log in `docs/ai-use-log/`
- [ ] Every member starts a running reflection log (Reflection is 10 marks at the Final)
- [ ] Email Mr Wee the open questions (project plan, Section 0): which dates apply, the PR1
      submission time, how individual marking works on a shared solution, whether PR-AUC or F1
      counts as the "industry benchmark", approved LLM APIs and budget
- [ ] Agree who owns which model variant and which agent tool

### Data (PR1 Input Data, 5 marks)

- [ ] Put the data pack in `data/raw/` and confirm everyone has identical files
- [ ] Audit every table: types, true/false encodings, numbers stored as text, duplicates, missing
      values, outliers, join keys, `noise_*` columns
- [ ] Clean tables into `data/interim/` (blank the 510 `delivered_date` values after the cutoff;
      never use the original `delivery_status` as a feature)
- [ ] Check the pre-aggregated columns in `customers.csv` (`loyalty_score`, `email_open_rate`,
      `discount_usage_rate`, `avg_review_score`) for post-cutoff information
- [ ] Build `data/processed/features.csv`: one row per customer for all 80,000 customers, nothing
      dated after 30 Sep 2024, keep average order value
- [ ] EDA notebook with plots and written findings

### Model (PR1 Supervised Model, 5 marks)

- [ ] Baselines: always-0, recency rule, logistic regression, LightGBM
- [ ] Each member: own deep learning variant (plain MLP / MLP with embeddings / GRU or LSTM over
      weekly activity / wide-and-deep or TabNet)
- [ ] Tuning log for every run: this is the evidence of optimisation
- [ ] Evaluation on the shared split: PR-AUC (main), ROC-AUC, precision, recall, F1, confusion
      matrix, precision in the top 10%. Accuracy only next to the 85.4% floor
- [ ] Leakage checks: shuffled-label test, no single dominant feature, scaler and imputer fitted
      on train only

### PR1 presentation (group, 5 marks) and live defence

- [ ] Progress slides
- [ ] Mock walkthrough: each member opens a random file of theirs and explains what it does, what
      depends on it and what breaks without it, with notes and AI closed

## Next: before Progress Review 2 (Week 14, w/c 18 Jan 2027, 30%)

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
