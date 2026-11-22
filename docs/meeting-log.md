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

Each member adds their own line.

**Done so far**

- Davian: cleaned the 17 data-pack tables and built the feature table in KNIME (`knime/`, Nov
  2026); repo set-up (`0822d10`); baselines, shared split and `data_prep.py` (`e6e97b3`); EDA
  (`22a9a37`); plain MLP with tuning, seed check, test results and fairness (`a356bd0`,
  `1918c7c`). Test PR-AUC 0.574, a tie with the baselines (LightGBM 0.576, logistic regression
  0.577).

**Changes to the plan**

- Davian: the plain MLP does not beat the baselines on PR-AUC, so the case for deep learning
  has to rest on the team's other variants or a combined model. Recall for customers aged 55+ is
  lower (55% against 73% for 18–24) and is flagged for review before the model feeds the agent.

**Actions** (owner, due date)

- [ ] Davian: add the shuffled-label leakage check to the MLP notebook, before PR2
- [ ] Davian: agree an agent tool with the team, before PR2
