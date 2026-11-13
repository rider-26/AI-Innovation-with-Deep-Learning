# data/raw: the ShopSphere data pack

Put the data pack here, exactly as issued. It is git-ignored, so each member downloads their own copy.

**Never edit, re-save or overwrite anything in this folder.** The brief forbids altering the data pack.
Cleaning writes new files to `data/interim/`; the feature table goes to `data/processed/`;
anything we generate ourselves goes to `data/synthetic/` and is labelled as synthetic.

Expected contents (17 tables, 4 policy documents, `DATA_README.md`):

| Group | Files |
|---|---|
| Customer | `customers.csv`, `customer_preferences.csv`, `geo_data.csv` |
| Behaviour | `sessions.csv`, `transactions.csv`, `transaction_items.csv` |
| Marketing | `marketing_campaigns.csv`, `campaign_daily_performance.csv`, `retention_contact_history.csv` |
| Operations | `products.csv`, `deliveries.csv`, `support_tickets.csv` |
| Finance and CRM | `offer_catalogue.csv`, `retention_budget_2024Q4.csv` |
| Labels | `train.csv` (64,000 labelled), `test.csv` (16,000 to predict), `sample_submission.csv` |
| Policies | `retention_offer_policy.md`, `shipping_and_returns_policy.md`, `customer_service_sla.md`, `loyalty_programme.md` |

The data is an extract as at **30 September 2024**. The target is 1 if the customer bought at least
once between 1 and 30 October 2024. The column-level data dictionary is in Appendix A of the brief
(in the repo root) and in `DATA_README.md` in the pack.
