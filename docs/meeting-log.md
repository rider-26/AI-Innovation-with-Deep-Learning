# Weekly meeting log

Newest meeting at the top. One entry per meeting, written during the meeting.

## Template

### Week N: DD Mon YYYY

- **Present:**
- **Absent:**

**Done since last meeting** (one line per member, with the commit or notebook)

-

**Decisions** (what was decided and why)

-

**Actions** (owner, due date)

- [ ]

**Blockers and questions for Mr Wee**

-

---

<!-- Add meetings below this line -->

### Week 5: progress update for PR1 (written 22 Nov 2026, not a meeting)

Built from each member's branch and commit history. Members: check your own line.

**Done so far**

- Jadon (leader): wide-and-deep model module and unit tests (`ea2e36f`), EDA with an interaction
  screen for the wide part's crossed features, and the model notebook run with findings
  (`554f551`, `43642c9`). Test PR-AUC 0.581 (wide only 0.487, deep only 0.585).
- Leon: embedding MLP module and 14 unit tests (`4e8736e`), notebook comparing learned
  embeddings with one-hot encoding (`cf251c8`), 20-trial search per network (`9a1806a`), AI Use
  Log (`eea7156`). Test PR-AUC 0.575 with embeddings, 0.583 with one-hot.
- Chad: weekly activity sequences (104 weeks × 4 numbers per customer), weekly EDA and GRU
  notebook (`c2ac22f`), GRU conclusions and AI Use Log (`c1df34b`). GRU hybrid test PR-AUC 0.587,
  the highest in the team; a shuffled-label test found no leakage.
- Davian: cleaned the 17 data-pack tables and built the feature table in KNIME (`knime/`, Nov
  2026); repo set-up (`0822d10`); baselines, shared split and `data_prep.py` (`e6e97b3`); EDA
  (`22a9a37`); plain MLP with tuning, seed check, test results and fairness (`a356bd0`,
  `1918c7c`). Test PR-AUC 0.574, a tie with the baselines (LightGBM 0.576, logistic regression
  0.577).

**Changes to the plan**

- Davian: the plain MLP does not beat the baselines on PR-AUC, so the case for deep learning
  has to rest on the team's other variants or a combined model. Recall for customers aged 55+ is
  lower (55% against 73% for 18–24) and is flagged for review before the model feeds the agent.
- Team: all four variants are within about 0.01 test PR-AUC of the baselines (0.574 to 0.587,
  against 0.576 and 0.577). The revised plan is to choose the agent's model on calibration,
  explanations and cost as well as PR-AUC, and to consider combining the variants before PR2.
- Team: agent tools were not assigned at PR1. Assign them before work on the agent starts.

**Actions** (owner, due date)

- [ ] Davian: add the shuffled-label leakage check to the MLP notebook, before PR2
- [ ] Davian: agree an agent tool with the team, before PR2
- [ ] Jadon: add an AI Use Log, before PR1 submission
- [ ] Leon, Jadon: add the shuffled-label leakage check, before PR2
- [ ] All: merge the four branches into `main` by pull request
- [ ] All: assign agent tools and choose the model for the agent, before PR2
